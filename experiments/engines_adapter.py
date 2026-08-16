"""Step-by-step replay adapters for both the ChainCaps and ElasticCap engines.

The role of this module is to take an ``ElasticScenario`` (defined in
:mod:`experiments.scenarios`) and run its tool-call sequence through:
  * :func:`run_elasticap` — the ElasticCap engine (scope-isolated DAG with
    optional DeepSeek/Mock auditor and auto-recovery on DISPLAY sinks).
  * :func:`run_chaincaps` — the stock ChainCaps engine from
    ``src/chaincaps-code``. Used for in-repo cross-checks against the
    frozen ``raid_v3_results.json`` so we can trust the comparison.

Two cases of data-dependency resolution are supported, mirroring
``eval/ablation_runner._resolve_deps``:

* ``deps is None``         -> fill with *every* prior output node (so the
  engine's conservative context applies).
* ``deps == []``           -> leave empty (intentional; engine falls back
  to the context budget / all nodes).
* ``deps == ['_idx:N']``   -> resolved to the output node id of step N.
* ``deps`` real node ids   -> used verbatim.

The module also exposes :class:`ReplayOutcome` which carries per-step
verdicts, the engine report, and a small component trajectory for
root-cause diagnostics; both experiment-1 (which only inspects top-line
metrics) and experiment-2 (which inspects the trajectory) consume this.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from elasticap.budget import Budget, SinkPrivilege, SinkType
from elasticap.engine import (
    ElasticCapEngine, ToolCall as ECToolCall, ToolResult as ECToolResult,
)
from elasticap.manifest import ToolManifest
from elasticap.auditor import DeclassificationAuditor
from elasticap.dag import ElasticDAG

from .scenarios import (
    ElasticScenario,
    _CHAINCAPS_ROOT,
)

# Ensure the ChainCaps baseline codebase is importable.
if str(_CHAINCAPS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAINCAPS_ROOT))

from chaincaps.proxy.engine import (  # type: ignore
    ChainCapsEngine, ToolCall as CCToolCall, ToolResult as CCToolResult,
)
from chaincaps.core.manifest import ToolManifest as CCToolManifest  # type: ignore
from chaincaps.core.dag import DataflowDAG  # type: ignore
from chaincaps.core.budget import (  # type: ignore
    Budget as CCBudget, SinkPrivilege as CCSinkPrivilege, SinkType as CCSinkType,
)


# ---------------------------------------------------------------------------
# Cross-package budget / privilege translation
# ---------------------------------------------------------------------------

# elasticap and chaincaps both define DISPLAY/SEND_HTTP/SEND_EMAIL/WRITE_FILE/
# EXECUTE/DB_WRITE/MEMORY_WRITE with identical names but in disjoint Enum
# classes. ``SinkPrivilege`` equality requires the *same* enum object, so a
# budget authored with elasticap's SinkType does NOT authorise anything when
# handed to the chaincaps engine. We bridge that by rebuilding each budget
# with chaincaps' own SinkPrivilege objects.

_EC_SINKTYPE_TO_CC = {
    SinkType.DISPLAY:    CCSinkType.DISPLAY,
    SinkType.SEND_HTTP:  CCSinkType.SEND_HTTP,
    SinkType.SEND_EMAIL: CCSinkType.SEND_EMAIL,
    SinkType.WRITE_FILE: CCSinkType.WRITE_FILE,
    SinkType.EXECUTE:    CCSinkType.EXECUTE,
    SinkType.DB_WRITE:   CCSinkType.DB_WRITE,
    SinkType.MEMORY_WRITE: CCSinkType.MEMORY_WRITE,
}


def _translate_budget(b: Budget) -> CCBudget:
    """Rebuild a (elasticap) ``Budget`` as a chaincaps ``Budget``.

    Uses by-symbol name lookup so the mapping is robust even if the two
    Enum classes add members in differing orders in future.
    """
    if b is None:
        return None  # type: ignore[return-value]
    cc_privs = []
    for p in b.privileges:
        cc_op = _EC_SINKTYPE_TO_CC.get(p.operation, getattr(
            CCSinkType, p.operation.name, None))
        if cc_op is None:
            continue
        cc_privs.append(CCSinkPrivilege(cc_op, p.scope))
    return CCBudget(frozenset(cc_privs))


def _translate_source_budgets(
    budgets: Dict[str, Budget],
) -> Dict[str, CCBudget]:
    return {k: _translate_budget(v) for k, v in budgets.items()}


# ---------------------------------------------------------------------------
# Custom manifest sets shared across both engines
# ---------------------------------------------------------------------------

def _shared_transform_manifests() -> Dict[str, ToolManifest]:
    """Custom manifests for transform-only tools used by the new scenarios."""
    return {
        "compute_average": ToolManifest(
            name="compute_average", description="Compute average of data",
        ),
        "compute_stats": ToolManifest(
            name="compute_stats", description="Compute statistics from data",
        ),
        "aggregate_stats": ToolManifest(
            name="aggregate_stats", description="Compute aggregate statistics",
        ),
        "summarize": ToolManifest(
            name="summarize", description="Summarize text content",
        ),
        "generate_report": ToolManifest(
            name="generate_report", description="Generate a formatted report",
        ),
        "compose_email": ToolManifest(
            name="compose_email", description="Compose an email draft",
        ),
        "edit_file": ToolManifest(
            name="edit_file", description="Edit file content",
        ),
        "format_data": ToolManifest(
            name="format_data", description="Format data for presentation",
        ),
        # Sanitization / aggregation transforms (critical for Auditor GREEN path)
        "anonymize": ToolManifest(
            name="anonymize", description="Anonymize sensitive fields in data",
        ),
        "redact": ToolManifest(
            name="redact", description="Redact sensitive fields from data",
        ),
        "mask": ToolManifest(
            name="mask", description="Mask sensitive fields in data",
        ),
        "hash_data": ToolManifest(
            name="hash_data", description="Hash data for integrity verification",
        ),
        "sanitize": ToolManifest(
            name="sanitize", description="Sanitize data by removing PII",
        ),
        # Database query (source-like transform)
        "db_query": ToolManifest(
            name="db_query", description="Query a database",
            is_source=True,
            default_source_budget=Budget.from_sinks(
                SinkPrivilege(SinkType.DISPLAY),
            ),
        ),
        # Sink tools
        "git_commit": ToolManifest(
            name="git_commit", is_sink=True,
            exec_privileges=[SinkPrivilege(SinkType.EXECUTE, "*")],
            description="git commit",
        ),
        "git_push": ToolManifest(
            name="git_push", is_sink=True,
            exec_privileges=[SinkPrivilege(SinkType.SEND_HTTP, "github.com/*")],
            description="git push to remote",
        ),
    }


SHARED_MANIFESTS = _shared_transform_manifests()


def _translate_sinkpriv(p: SinkPrivilege) -> CCSinkPrivilege:
    cc_op = _EC_SINKTYPE_TO_CC.get(p.operation, getattr(
        CCSinkType, p.operation.name, None))
    return CCSinkPrivilege(cc_op, p.scope)


def _to_chaincaps_manifests(manifests: Dict[str, ToolManifest]
                           ) -> Dict[str, CCToolManifest]:
    out = {}
    for name, m in manifests.items():
        cc_exec = [_translate_sinkpriv(p) for p in m.exec_privileges]
        cc_pass = _translate_budget(m.pass_through) if m.pass_through is not None else None
        cc_default = _translate_budget(m.default_source_budget) if m.default_source_budget is not None else None
        out[name] = CCToolManifest(
            name=m.name,
            exec_privileges=cc_exec,
            is_sink=m.is_sink,
            is_source=m.is_source,
            default_source_budget=cc_default,
            pass_through=cc_pass,
            description=m.description,
        )
    return out


_CC_MANIFESTS = _to_chaincaps_manifests(SHARED_MANIFESTS)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

class _GlobalContextDAG(ElasticDAG):
    """A no-scope DAG: behave like ChainCaps (one global context budget)."""

    def __init__(self, signing_key=None):
        super().__init__(signing_key=signing_key)
        self._global_context_budget: Budget = Budget.top()

    def add_source(self, label, init_budget, tool_name=None):
        node_id = super().add_source(label, init_budget, tool_name)
        # Globally tighten the single context budget (ChainCaps behaviour).
        self._global_context_budget = self._global_context_budget.meet(init_budget)
        return node_id

    def add_transform(self, label, input_node_ids, tool_name, pass_through=None):
        node_id = super().add_transform(label, input_node_ids, tool_name, pass_through)
        node = self.nodes[node_id]
        self._global_context_budget = self._global_context_budget.meet(node.budget)
        return node_id

    def get_context_budget(self, input_node_ids: List[str]) -> Budget:
        return self._global_context_budget

    def mark_component_complete(self, node_id: str):
        # No-op in -Scope-off configuration: budgets never release.
        return


class _NoScopeElasticCapEngine(ElasticCapEngine):
    """An ElasticCap engine whose DAG is the single global budget (-Scope)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace the DAG; preserve the signing key so tokens still verify.
        signing_key = self.dag._signing_key
        self.dag = _GlobalContextDAG(signing_key=signing_key)


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

def _resolve_deps(deps: Optional[List[str]],
                  all_node_ids: List[Optional[str]],
                  engine_nodes: Dict[str, Any]) -> List[str]:
    """Resolve ``deps`` against the raw per-step output-node list.

    ``all_node_ids`` MUST be the *original* (unfiltered) list — sink steps
    produce ``None`` output ids and so their entries must be preserved
    to keep ``_idx:N`` step-index references aligned.
    """
    if deps is None:
        return [n for n in all_node_ids if n]
    if len(deps) == 0:
        return []
    resolved = []
    for d in deps:
        if isinstance(d, str) and d.startswith("_idx:"):
            try:
                idx = int(d.split(":")[1])
            except ValueError:
                continue
            if 0 <= idx < len(all_node_ids) and all_node_ids[idx] is not None:
                resolved.append(all_node_ids[idx])
        elif isinstance(d, str) and d in engine_nodes:
            if d in engine_nodes:
                resolved.append(d)
    # NOTE: do NOT blanket-fallback to all nodes on an un-resolved _idx —
    # that would merge otherwise-independent dependency chains. Instead
    # leave the call with its (possibly empty) explicit deps.
    return resolved


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------

@dataclass
class ReplayOutcome:
    scenario_name: str
    engine: str
    outcomes: List[bool] = field(default_factory=list)
    deny_count: int = 0
    attack_blocked: Optional[bool] = None
    benign_completed: Optional[bool] = None
    final_outcome: str = ""
    stats: Dict = field(default_factory=dict)
    block_reasons: List[Optional[str]] = field(default_factory=list)
    component_trajectory: List[Dict] = field(default_factory=list)
    audit_verdicts: List[Dict] = field(default_factory=list)
    latency_ms_total: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "scenario_name": self.scenario_name,
            "engine": self.engine,
            "outcomes": self.outcomes,
            "deny_count": self.deny_count,
            "attack_blocked": self.attack_blocked,
            "benign_completed": self.benign_completed,
            "final_outcome": self.final_outcome,
            "stats": self.stats,
            "block_reasons": self.block_reasons,
            "latency_ms_total": self.latency_ms_total,
            "audit_verdicts": self.audit_verdicts,
        }


# ---------------------------------------------------------------------------
# Engine replay core (shared by both engines to keep semantics identical)
# ---------------------------------------------------------------------------

def _replay_one(engine,
                scenario: ElasticScenario,
                tool_call_factory: Callable,
                engine_label: str,
                get_stats: Callable[[], Dict],
                capture_trajectory: bool = False) -> ReplayOutcome:
    """Drive a scenario through an already-constructed engine.

    ``tool_call_factory(tool_name, args, deps)`` returns the per-engine
    ToolCall object (ChainCaps vs ElasticCap differ by type).
    ``get_stats()`` returns the engine's final stats dict.
    """
    outcome = ReplayOutcome(scenario_name=scenario.name, engine=engine_label)
    all_node_ids: List[Optional[str]] = []
    t0 = time.perf_counter()

    for step_idx, (tool_name, args, deps) in enumerate(scenario.steps):
        resolved = _resolve_deps(deps, all_node_ids, engine.dag.nodes)
        call = tool_call_factory(tool_name, dict(args), resolved)
        result = engine.process_tool_call(call)

        outcome.outcomes.append(result.allowed)
        outcome.block_reasons.append(result.block_reason if not result.allowed else None)
        if result.allowed:
            all_node_ids.append(result.output_node_id)
        else:
            all_node_ids.append(all_node_ids[-1] if all_node_ids else None)
            outcome.deny_count += 1
            if (scenario.attack_step is not None
                    and step_idx == scenario.attack_step):
                outcome.attack_blocked = True
            if scenario.task_type == "benign":
                outcome.benign_completed = False

        if capture_trajectory and hasattr(engine.dag, "_components"):
            outcome.component_trajectory.append({
                "step": step_idx,
                "tool": tool_name,
                "allowed": result.allowed,
                "components": {
                    int(k): repr(v) for k, v in engine.dag._components.items()
                },
                "completed": sorted(engine.dag._completed_components)
                            if hasattr(engine.dag, "_completed_components") else [],
                "node_to_comp": {
                    k: v for k, v in engine.dag._node_to_component.items()
                } if hasattr(engine.dag, "_node_to_component") else {},
            })
        if capture_trajectory and hasattr(result, "audit_result") and result.audit_result:
            outcome.audit_verdicts.append({
                "step": step_idx,
                "tool": tool_name,
                "verdict": result.audit_result.verdict,
                "reason": result.audit_result.reason,
            })

    outcome.latency_ms_total = (time.perf_counter() - t0) * 1000.0
    outcome.stats = get_stats()

    if scenario.task_type == "adversarial":
        if scenario.attack_step is not None:
            outcome.attack_blocked = (
                bool(outcome.attack_blocked)
                or not outcome.outcomes[scenario.attack_step]
            )
        outcome.benign_completed = None
        outcome.final_outcome = "blocked" if outcome.attack_blocked else "succeeded"
    else:
        if outcome.benign_completed is None:
            outcome.benign_completed = (outcome.deny_count == 0)
        outcome.final_outcome = "success" if outcome.benign_completed else "failed"
    return outcome


# ---------------------------------------------------------------------------
# ElasticCap run
# ---------------------------------------------------------------------------

def run_elasticap(
    scenario: ElasticScenario,
    auditor: Optional[DeclassificationAuditor] = None,
    use_recovery: bool = True,
    use_scope: bool = True,
    manifests: Optional[Dict[str, ToolManifest]] = None,
) -> ReplayOutcome:
    """Run ``scenario`` through the ElasticCap engine.

    Optional toggles exist for the ablation experiment (exp5):
      * ``use_recovery`` - when False, automatic mark-component-complete on
        DISPLAY sinks is disabled (the `-TTL` configuration).
      * ``use_scope``    - when False, the ElasticDAG is replaced by a plain
        ChainCaps-style global-context-budget DAG (the `-Scope` configuration).
      * ``auditor``      - when ``None``, suspends the Auditor path entirely.
    """
    if use_scope:
        engine = ElasticCapEngine(
            manifests=manifests or dict(SHARED_MANIFESTS),
            source_budget_overrides=dict(scenario.source_budgets),
            auditor=auditor,
        )
    else:
        engine = _NoScopeElasticCapEngine(
            manifests=manifests or dict(SHARED_MANIFESTS),
            source_budget_overrides=dict(scenario.source_budgets),
            auditor=auditor,
        )
    # Allow toggling the engine's automatic-recovery hook at replay time.
    engine._auto_recovery_enabled = use_recovery  # type: ignore[attr-defined]
    engine.set_user_intent(scenario.user_intent)

    def factory(name, args, deps):
        return ECToolCall(tool_name=name, arguments=args, data_dependencies=deps)

    def get_stats():
        return engine.get_report()

    return _replay_one(engine, scenario, factory, "elasticap", get_stats,
                       capture_trajectory=True)


# ---------------------------------------------------------------------------
# ChainCaps run (baseline cross-check)
# ---------------------------------------------------------------------------

def run_chaincaps(scenario: ElasticScenario,
                  manifests: Optional[Dict[str, CCToolManifest]] = None
                  ) -> ReplayOutcome:
    """Run ``scenario`` through the stock ChainCaps engine.

    Used as an in-repo cross-check against ``raid_v3_results.json`` so we
    can validate that the scenarios still reproduce the recorded FPs.
    """
    engine = ChainCapsEngine(
        manifests=manifests or dict(_CC_MANIFESTS),
        source_budget_overrides=_translate_source_budgets(scenario.source_budgets),
    )

    def factory(name, args, deps):
        return CCToolCall(tool_name=name, arguments=args, data_dependencies=deps)

    def get_stats():
        return engine.get_report()

    return _replay_one(engine, scenario, factory, "chaincaps", get_stats,
                       capture_trajectory=False)


__all__ = [
    "ReplayOutcome",
    "SHARED_MANIFESTS",
    "run_elasticap",
    "run_chaincaps",
]
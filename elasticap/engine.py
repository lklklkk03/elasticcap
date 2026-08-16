"""
ElasticCap Proxy: Runtime Enforcement Engine with Auditor Routing

Key improvement over ChainCapsEngine:
- Uses ElasticDAG (scope-isolated budget) instead of DataflowDAG
- _handle_sink routes blocked calls to Auditor instead of directly rejecting
- Supports "suspend → audit → re-inject" closed loop
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .budget import Budget, SinkPrivilege, SinkType
from .dag import ElasticDAG, SinkEvent
from .manifest import ToolManifest, get_manifest
from .auditor import DeclassificationAuditor, AuditResult


@dataclass
class ToolCall:
    """A tool call intercepted by the proxy."""
    tool_name: str
    arguments: Dict[str, Any]
    data_dependencies: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result of a tool call after proxy enforcement."""
    tool_call: ToolCall
    allowed: bool
    output: Optional[Any] = None
    output_node_id: Optional[str] = None
    sink_event: Optional[SinkEvent] = None
    block_reason: Optional[str] = None
    latency_ms: float = 0.0
    audit_result: Optional[AuditResult] = None  # ElasticCap: audit trail


@dataclass
class EnforcementStats:
    """Aggregate statistics for evaluation."""
    total_calls: int = 0
    allowed_calls: int = 0
    blocked_calls: int = 0
    declassified_calls: int = 0
    audited_calls: int = 0  # ElasticCap: calls routed to auditor
    auditor_green: int = 0
    auditor_red: int = 0
    auditor_yellow: int = 0
    source_reads: int = 0
    transforms: int = 0
    sink_attempts: int = 0
    total_latency_ms: float = 0.0
    dag_nodes: int = 0

    @property
    def block_rate(self) -> float:
        return self.blocked_calls / max(self.total_calls, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_calls, 1)


class ElasticCapEngine:
    """The ElasticCap runtime enforcement engine.

    Uses ElasticDAG for scope-isolated budget tracking and
    DeclassificationAuditor for automated declassification decisions.
    """

    def __init__(self,
                 manifests: Optional[Dict[str, ToolManifest]] = None,
                 source_budget_overrides: Optional[Dict[str, Budget]] = None,
                 auditor: Optional[DeclassificationAuditor] = None):
        self.dag = ElasticDAG()
        self.manifests = manifests or {}
        self.source_budget_overrides = source_budget_overrides or {}
        self.stats = EnforcementStats()
        self._auditor = auditor

        # Store current user intent for auditor (set before processing)
        self._current_user_intent: str = ""
        # Ablation toggle: when False, the automatic mark_component_complete
        # on DISPLAY sinks is disabled (the "-TTL" configuration).
        self._auto_recovery_enabled: bool = True

    def set_user_intent(self, intent: str):
        """Set the current user's original request for audit context.

        IMPORTANT: This should be set from the system frontend API,
        NOT from the Agent's context, to prevent injection attacks.
        """
        self._current_user_intent = intent

    def _filter_active_nodes(self, node_ids: List[str]) -> List[str]:
        """Drop nodes whose dependency chain has already been released.

        Used by the conservative "depend on every known node" fallbacks in
        ``_handle_transform`` / ``_handle_sink``. Once a dependency chain is
        marked complete (its terminal consumer was a DISPLAY sink), its
        nodes should no longer poison the implicit-context fallback —
        this is the core of the ElasticCap scope-isolation FP fix
        (design_doc §4.3 Step 3). Explicit data-dep flows still use the
        original budget and are NOT affected.
        """
        completed = getattr(self.dag, "_completed_components", set())
        node_to_comp = getattr(self.dag, "_node_to_component", {})
        if not completed:
            return list(node_ids)
        active = [nid for nid in node_ids
                  if node_to_comp.get(nid) not in completed]
        # Fall back to all nodes if EVERYTHING is completed (don't
        # accidentally authorise a real attack because the filter left no
        # controlled inputs — keep it conservative).
        return active if active else list(node_ids)

    def get_manifest(self, tool_name: str) -> ToolManifest:
        if tool_name in self.manifests:
            return self.manifests[tool_name]
        return get_manifest(tool_name)

    def process_tool_call(self, call: ToolCall,
                          tool_executor: Optional[Callable] = None,
                          declassification_token: Optional[str] = None,
                          ) -> ToolResult:
        """Process a tool call through the ElasticCap enforcement pipeline.

        Pipeline: parse → check → (allow | suspend→audit→re-inject | block)
        """
        start_time = time.time()
        self.stats.total_calls += 1

        manifest = self.get_manifest(call.tool_name)
        result = ToolResult(tool_call=call, allowed=False)

        if manifest.is_source and manifest.is_sink:
            has_data_deps = bool(call.data_dependencies) or bool(self.dag.nodes)
            if has_data_deps and manifest.exec_privileges:
                result = self._handle_sink(call, manifest, declassification_token)
            else:
                result = self._handle_source(call, manifest)
        elif manifest.is_source:
            result = self._handle_source(call, manifest)
        elif manifest.is_sink:
            result = self._handle_sink(call, manifest, declassification_token)
        else:
            result = self._handle_transform(call, manifest)

        # Execute if allowed
        if result.allowed and tool_executor:
            try:
                result.output = tool_executor(call.tool_name, call.arguments)
            except Exception as e:
                result.output = f"EXECUTION_ERROR: {e}"

        # ElasticCap automatic recovery: when a DISPLAY sink succeeds, the
        # dependency chain that produced the displayed data has reached its
        # final consumer; mark its component complete so its budget stops
        # polluting subsequent unrelated operations (design_doc §4.3 Step 3).
        if (self._auto_recovery_enabled
                and result.allowed and manifest.is_sink
                and result.sink_event
                and result.sink_event.data_node_ids
                and any(p.operation == SinkType.DISPLAY
                        for p in manifest.exec_privileges)):
            # Mark the component of the dependency chain feeding this display
            last_dep = result.sink_event.data_node_ids[-1]
            self.dag.mark_component_complete(last_dep)

        result.latency_ms = (time.time() - start_time) * 1000
        self.stats.total_latency_ms += result.latency_ms
        self.stats.dag_nodes = len(self.dag.nodes)

        return result

    def _handle_source(self, call: ToolCall,
                       manifest: ToolManifest) -> ToolResult:
        self.stats.source_reads += 1

        resource_key = self._extract_resource_key(call)
        if resource_key and resource_key in self.source_budget_overrides:
            budget = self.source_budget_overrides[resource_key]
        elif manifest.default_source_budget:
            budget = manifest.default_source_budget
        else:
            budget = Budget.from_sinks(SinkPrivilege(SinkType.DISPLAY))

        label = f"{call.tool_name}({resource_key or '?'})"
        node_id = self.dag.add_source(label, budget, call.tool_name)

        self.stats.allowed_calls += 1
        return ToolResult(
            tool_call=call,
            allowed=True,
            output_node_id=node_id,
        )

    def _handle_transform(self, call: ToolCall,
                          manifest: ToolManifest) -> ToolResult:
        self.stats.transforms += 1

        input_ids = call.data_dependencies
        if not input_ids:
            # ElasticCap scope isolation: explicit-dep-less transforms
            # depend on active (non-completed-chain) nodes only.
            input_ids = self._filter_active_nodes(list(self.dag.nodes.keys()))

        if not input_ids:
            node_id = self.dag.add_source(
                f"{call.tool_name}(no_deps)",
                Budget.top(),
                call.tool_name,
            )
            self.stats.allowed_calls += 1
            return ToolResult(
                tool_call=call, allowed=True, output_node_id=node_id
            )

        label = f"{call.tool_name}({json.dumps(call.arguments, default=str)[:50]})"
        node_id = self.dag.add_transform(
            label, input_ids, call.tool_name, manifest.pass_through
        )

        self.stats.allowed_calls += 1
        return ToolResult(
            tool_call=call, allowed=True, output_node_id=node_id
        )

    def _handle_sink(self, call: ToolCall, manifest: ToolManifest,
                     declassification_token: Optional[str] = None) -> ToolResult:
        """Handle a sink tool call with auditor routing.

        ELASTICCAP CHANGE: When blocked, route to auditor instead of
        immediately rejecting. Implements suspend→audit→re-inject.
        """
        self.stats.sink_attempts += 1

        requested = self._infer_sink_privilege(call, manifest)
        if not requested:
            self.stats.blocked_calls += 1
            return ToolResult(
                tool_call=call,
                allowed=False,
                block_reason="Cannot determine requested sink privilege",
            )

        data_deps = call.data_dependencies
        if not data_deps:
            # Conservative fallback: depend on all known nodes, EXCEPT nodes
            # whose dependency chain was already released (marked complete).
            # This is the ElasticCap scope-isolation behavior: a previously
            # displayed secret no longer poisons the context budget of a
            # later, unrelated operation whose data lineage we cannot pin
            # down. Explicit data deps (below) still use the actual budget,
            # preserving the non-amplification theorem for explicit flows.
            data_deps = self._filter_active_nodes(list(self.dag.nodes.keys()))

        event = self.dag.check_sink(
            call.tool_name, requested, data_deps, declassification_token
        )

        if event.authorized:
            if declassification_token:
                self.stats.declassified_calls += 1
            self.stats.allowed_calls += 1
            return ToolResult(
                tool_call=call,
                allowed=True,
                sink_event=event,
            )

        # ═══════════════════════════════════════════════
        # ELASTICCAP: Route to auditor instead of blocking
        # ═══════════════════════════════════════════════

        if self._auditor:
            self.stats.audited_calls += 1

            # Resolve dependency chain for auditor
            dep_chain = self._resolve_dependency_chain(data_deps)

            audit_result = self._auditor.audit(
                sink_call=call,
                dependency_chain=dep_chain,
                user_intent=self._current_user_intent,
            )

            if audit_result.verdict == "GREEN":
                # Auto-declassify: re-inject with token
                self.stats.auditor_green += 1
                self.stats.declassified_calls += 1
                self.stats.allowed_calls += 1
                return ToolResult(
                    tool_call=call,
                    allowed=True,
                    sink_event=event,
                    audit_result=audit_result,
                )
            elif audit_result.verdict == "RED":
                self.stats.auditor_red += 1
                self.stats.blocked_calls += 1
                return ToolResult(
                    tool_call=call,
                    allowed=False,
                    sink_event=event,
                    block_reason=f"[Auditor-RED] {audit_result.reason}",
                    audit_result=audit_result,
                )
            else:  # YELLOW
                self.stats.auditor_yellow += 1
                self.stats.blocked_calls += 1
                return ToolResult(
                    tool_call=call,
                    allowed=False,
                    sink_event=event,
                    block_reason=f"[Auditor-YELLOW] {audit_result.reason}",
                    audit_result=audit_result,
                )

        # No auditor: behave like ChainCaps (direct block)
        self.stats.blocked_calls += 1
        return ToolResult(
            tool_call=call,
            allowed=event.authorized,
            sink_event=event,
            block_reason=event.block_reason,
        )

    def _resolve_dependency_chain(self, node_ids: List[str]) -> List:
        """Resolve the full dependency chain for a set of node IDs."""
        from .dag import DataNode
        visited = set()
        chain = []

        def _dfs(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            if nid in self.dag.nodes:
                node = self.dag.nodes[nid]
                chain.append(node)
                for parent_id in node.source_ids:
                    _dfs(parent_id)

        for nid in node_ids:
            _dfs(nid)

        return chain

    def _extract_resource_key(self, call: ToolCall) -> Optional[str]:
        args = call.arguments
        for key in ("path", "file", "filename", "url", "query", "table"):
            if key in args:
                return str(args[key])
        for v in args.values():
            if isinstance(v, str):
                return v
        return None

    def _infer_sink_privilege(self, call: ToolCall,
                              manifest: ToolManifest
                              ) -> Optional[SinkPrivilege]:
        if not manifest.exec_privileges:
            return None

        base = manifest.exec_privileges[0]
        args = call.arguments

        if base.operation == SinkType.SEND_HTTP:
            url = args.get("url", args.get("endpoint", base.scope))
            return SinkPrivilege(SinkType.SEND_HTTP, str(url))
        elif base.operation == SinkType.SEND_EMAIL:
            to = args.get("to", args.get("recipient", "*"))
            return SinkPrivilege(SinkType.SEND_EMAIL, str(to))
        elif base.operation == SinkType.WRITE_FILE:
            path = args.get("path", args.get("file", "*"))
            return SinkPrivilege(SinkType.WRITE_FILE, str(path))
        elif base.operation == SinkType.EXECUTE:
            cmd = args.get("command", args.get("cmd", "*"))
            return SinkPrivilege(SinkType.EXECUTE, str(cmd))

        return base

    def get_report(self) -> Dict:
        return {
            "stats": {
                "total_calls": self.stats.total_calls,
                "allowed": self.stats.allowed_calls,
                "blocked": self.stats.blocked_calls,
                "declassified": self.stats.declassified_calls,
                "audited": self.stats.audited_calls,
                "auditor_green": self.stats.auditor_green,
                "auditor_red": self.stats.auditor_red,
                "auditor_yellow": self.stats.auditor_yellow,
                "block_rate": self.stats.block_rate,
                "avg_latency_ms": self.stats.avg_latency_ms,
            },
            "dag": self.dag.stats,
            "sink_events": [
                {
                    "tool": e.tool_name,
                    "privilege": str(e.requested_privilege),
                    "authorized": e.authorized,
                    "reason": e.block_reason,
                }
                for e in self.dag.sink_events
            ],
        }

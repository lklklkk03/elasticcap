"""
ElasticCap Core: Elastic Dataflow DAG with Scope Isolation

Key improvement over ChainCaps' DataflowDAG:
- Instead of a single global _context_budget, we maintain per-component budgets
  based on dataflow dependency chain connectivity (union-find).
- When a dependency chain completes, its budget constraint is released,
  preventing cross-chain pollution (the "context guilt-by-association" problem).

Theory mapping:
- Dependency chain = Chinese Wall conflict class
- Component completion = Time decay (but topology-triggered, not time-triggered)
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .budget import Budget, SinkPrivilege, SinkType


@dataclass
class DataNode:
    """A node in the dataflow DAG representing a data item."""
    node_id: str
    label: str
    budget: Budget
    source_ids: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    is_source: bool = False

    @property
    def source_ancestors(self) -> Set[str]:
        return set()


@dataclass
class SinkEvent:
    """A record of an attempted or executed sink operation."""
    event_id: str
    tool_name: str
    requested_privilege: SinkPrivilege
    data_node_ids: List[str]
    propagated_budget: Budget
    authorized: bool
    declassification_token: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    block_reason: Optional[str] = None


class ElasticDAG:
    """Online dataflow DAG with scope-isolated budget propagation.

    Key differences from ChainCaps' DataflowDAG:
    1. _components: Dict[int, Budget] — per-component budgets instead of one global
    2. _node_to_component: Dict[str, int] — maps each node to its connected component
    3. _completed_components: Set[int] — completed chains with released budgets
    4. get_context_budget() only intersects budgets within the same component
    """

    def __init__(self, signing_key: Optional[bytes] = None):
        self.nodes: Dict[str, DataNode] = {}
        self.sink_events: List[SinkEvent] = []

        # === ElasticCap: component-based scope isolation ===
        # Instead of a single global _context_budget, we track budgets per
        # connected component (dependency chain).
        self._components: Dict[int, Budget] = {}
        self._node_to_component: Dict[str, int] = {}
        self._component_counter: int = 0
        self._completed_components: Set[int] = set()

        # Spent declassification tokens (replay prevention)
        self._spent_tokens: Set[str] = set()
        import os as _os
        self._signing_key = signing_key or _os.urandom(32)

    # ═══════════════════════════════════════════════════════════════
    # Component management (ElasticCap core logic)
    # ═══════════════════════════════════════════════════════════════

    def _find_or_create_component(self, input_node_ids: List[str]) -> int:
        """Find the connected component for a set of input nodes.

        If all inputs are new (no historical deps) → creates a new component.
        If inputs belong to existing components → merges them.

        Returns the component ID.
        """
        comps: Set[int] = set()
        for nid in input_node_ids:
            if nid in self._node_to_component:
                comp_id = self._node_to_component[nid]
                if comp_id not in self._completed_components:
                    comps.add(comp_id)

        if not comps:
            # New independent dependency chain
            self._component_counter += 1
            return self._component_counter

        # Merge all related components (take the minimum ID as merged target)
        merged = min(comps)
        for c in comps:
            if c != merged:
                # Merge budget: take the meet
                if c in self._components:
                    if merged in self._components:
                        self._components[merged] = \
                            self._components[merged].meet(self._components[c])
                    else:
                        self._components[merged] = self._components[c]
                    del self._components[c]
                # Reassign all nodes from old component to merged
                for nid, cid in list(self._node_to_component.items()):
                    if cid == c:
                        self._node_to_component[nid] = merged

        return merged

    def get_context_budget(self, input_node_ids: List[str]) -> Budget:
        """Get the effective context budget for a call.

        KEY ELASTICCAP CHANGE: Only intersects budgets from the same
        dependency chain (component), NOT globally — UNLESS there are no
        explicit input nodes, in which case we conservatively fall back
        to the ChainCaps-style global meet of ALL active components.
        This preserves scope isolation for explicit deps while maintaining
        ChainCaps-level security for implicit-dependency attacks.

        ChainCaps equivalent: self._context_budget (always global)
        """
        if not input_node_ids:
            # No explicit dependencies: conservative fallback to ChainCaps
            # global context budget (meet of ALL active component budgets).
            # This ensures we don't accidentally allow implicit-dependency
            # attacks that ChainCaps would have blocked.
            global_budget = Budget.top()
            for cid, comp_budget in self._components.items():
                if cid not in self._completed_components:
                    global_budget = global_budget.meet(comp_budget)
            return global_budget

        # Find which component(s) the inputs belong to
        comps: Set[int] = set()
        for nid in input_node_ids:
            if nid in self._node_to_component:
                cid = self._node_to_component[nid]
                if cid not in self._completed_components:
                    comps.add(cid)

        if not comps:
            return Budget.top()

        # Meet budgets only within the relevant components
        result = Budget.top()
        for cid in comps:
            if cid in self._components:
                result = result.meet(self._components[cid])

        return result

    def mark_component_complete(self, node_id: str):
        """Mark a node's component as complete, releasing its budget constraint.

        This is the ElasticCap equivalent of Chinese Wall "time decay" —
        once a dependency chain finishes, its constraints no longer
        pollute subsequent unrelated operations.
        """
        if node_id in self._node_to_component:
            comp_id = self._node_to_component[node_id]
            self._completed_components.add(comp_id)

    def _update_component_budget(self, comp_id: int, new_budget: Budget):
        """Update the budget for a component (always via meet — monotonic)."""
        if comp_id in self._components:
            self._components[comp_id] = self._components[comp_id].meet(new_budget)
        else:
            self._components[comp_id] = new_budget

    # ═══════════════════════════════════════════════════════════════
    # DAG construction (API compatible with ChainCaps DataflowDAG)
    # ═══════════════════════════════════════════════════════════════

    def add_source(self, label: str, init_budget: Budget,
                   tool_name: Optional[str] = None) -> str:
        """Add a source data node.

        ELASTICCAP CHANGE: Each new source starts its own component
        rather than globally meeting the context budget.
        """
        node_id = f"src_{uuid.uuid4().hex[:8]}"
        node = DataNode(
            node_id=node_id,
            label=label,
            budget=init_budget,
            tool_name=tool_name,
            is_source=True,
        )
        self.nodes[node_id] = node

        # New source → new component (independent dependency chain)
        self._component_counter += 1
        comp_id = self._component_counter
        self._node_to_component[node_id] = comp_id
        self._update_component_budget(comp_id, init_budget)

        return node_id

    def add_transform(self, label: str, input_node_ids: List[str],
                      tool_name: str,
                      pass_through: Optional[Budget] = None) -> str:
        """Add a transform node. Budget = meet(parents) ∩ Pass(t).

        ELASTICCAP CHANGE: Assigns the node to the merged component
        of its inputs, rather than updating a global budget.
        """
        if not input_node_ids:
            raise ValueError("Transform must have at least one input")

        # Compute meet of all input budgets
        input_budgets = []
        for nid in input_node_ids:
            if nid not in self.nodes:
                raise KeyError(f"Unknown input node: {nid}")
            input_budgets.append(self.nodes[nid].budget)

        meet_budget = input_budgets[0]
        for b in input_budgets[1:]:
            meet_budget = meet_budget.meet(b)

        if pass_through is not None:
            meet_budget = meet_budget.meet(pass_through)

        node_id = f"xfm_{uuid.uuid4().hex[:8]}"
        node = DataNode(
            node_id=node_id,
            label=label,
            budget=meet_budget,
            source_ids=list(input_node_ids),
            tool_name=tool_name,
        )
        self.nodes[node_id] = node

        # Assign to component and update component budget
        comp_id = self._find_or_create_component(input_node_ids)
        self._node_to_component[node_id] = comp_id
        self._update_component_budget(comp_id, meet_budget)

        return node_id

    def check_sink(self, tool_name: str,
                   requested_privilege: SinkPrivilege,
                   data_node_ids: List[str],
                   declassification_token: Optional[str] = None) -> SinkEvent:
        """Check if a sink operation is authorized.

        ELASTICCAP CHANGE: Uses get_context_budget() which is scope-isolated,
        rather than ChainCaps' global _context_budget.

        Note: explicit data-flow dependency nodes use their actual budgets
        (preserving the monotonic meet rule — the non-amplification theorem
        from ChainCaps stays intact). Budget *release* of completed
        dependency chains is handled in the engine layer when it materialises
        the fallback dependency set (see ``ElasticCapEngine._filter_deps``).
        """
        if data_node_ids:
            budgets = []
            for nid in data_node_ids:
                if nid not in self.nodes:
                    raise KeyError(f"Unknown data node: {nid}")
                budgets.append(self.nodes[nid].budget)

            propagated = budgets[0]
            for b in budgets[1:]:
                propagated = propagated.meet(b)
            # Scope-isolated context budget (KEY DIFFERENCE)
            propagated = propagated.meet(self.get_context_budget(data_node_ids))
        else:
            propagated = self.get_context_budget([])

        authorized = propagated.authorizes(requested_privilege)
        block_reason = None

        if not authorized and declassification_token:
            authorized = self._validate_declassification(
                declassification_token, requested_privilege, data_node_ids
            )
            if authorized:
                block_reason = None
            else:
                block_reason = (
                    f"Sink {requested_privilege} not authorized; "
                    f"declassification token invalid or scope mismatch"
                )
        elif not authorized:
            block_reason = (
                f"Sink {requested_privilege} not authorized by "
                f"propagated budget {propagated}"
            )

        event = SinkEvent(
            event_id=f"sink_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            requested_privilege=requested_privilege,
            data_node_ids=list(data_node_ids),
            propagated_budget=propagated,
            authorized=authorized,
            declassification_token=declassification_token,
            block_reason=block_reason,
        )
        self.sink_events.append(event)
        return event

    def _validate_declassification(self, token: str,
                                    privilege: SinkPrivilege,
                                    data_node_ids: List[str]) -> bool:
        """Validate a cryptographically signed declassification token."""
        import json as _json
        import hashlib
        import hmac as _hmac

        try:
            tok = _json.loads(token)
        except (TypeError, _json.JSONDecodeError):
            return False

        if not isinstance(tok, dict):
            return False

        payload_str = tok.get("payload", "")
        provided_hmac = tok.get("hmac", "")
        if not payload_str or not provided_hmac:
            return False

        expected_hmac = _hmac.new(
            self._signing_key, payload_str.encode(), hashlib.sha256
        ).hexdigest()
        if not _hmac.compare_digest(provided_hmac, expected_hmac):
            return False

        try:
            payload = _json.loads(payload_str)
        except (TypeError, _json.JSONDecodeError):
            return False

        if not payload.get("user_approval"):
            return False
        if payload.get("scope") != "one-shot":
            return False
        if payload.get("sink", "") != str(privilege):
            return False

        tok_lineage = set(payload.get("lineage", []))
        if data_node_ids and not set(data_node_ids).issubset(tok_lineage):
            return False

        token_id = payload.get("token_id", "")
        if not token_id or token_id in self._spent_tokens:
            return False

        self._spent_tokens.add(token_id)
        return True

    def get_source_ancestors(self, node_id: str) -> Set[str]:
        visited = set()
        sources = set()

        def _dfs(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes[nid]
            if node.is_source:
                sources.add(nid)
            for parent_id in node.source_ids:
                _dfs(parent_id)

        _dfs(node_id)
        return sources

    def verify_budget_preservation(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        source_ids = self.get_source_ancestors(node_id)
        if not source_ids:
            return True
        source_budgets = [self.nodes[sid].budget for sid in source_ids]
        ancestor_meet = source_budgets[0]
        for b in source_budgets[1:]:
            ancestor_meet = ancestor_meet.meet(b)
        return node.budget.is_subset_of(ancestor_meet)

    @property
    def stats(self) -> Dict:
        total_sinks = len(self.sink_events)
        blocked = sum(1 for e in self.sink_events if not e.authorized)
        declassified = sum(
            1 for e in self.sink_events
            if e.authorized and e.declassification_token
        )
        return {
            "total_nodes": len(self.nodes),
            "source_nodes": sum(1 for n in self.nodes.values() if n.is_source),
            "transform_nodes": sum(
                1 for n in self.nodes.values() if not n.is_source
            ),
            "total_sink_events": total_sinks,
            "authorized": total_sinks - blocked,
            "blocked": blocked,
            "declassified": declassified,
            "components": len(self._components),
            "completed_components": len(self._completed_components),
        }

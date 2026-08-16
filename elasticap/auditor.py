"""
ElasticCap: Declassification Auditor (Module 3)

Two-layer audit architecture:
  Layer 1 (Deterministic Rules): Regex-based sensitive pattern detection
    + aggregation check. No LLM overhead, 100% deterministic.
  Layer 2 (LLM Semantic Audit): For boundary cases where Layer 1 cannot
    make a definitive decision. Only triggered when needed.

Three-tier verdict:
  RED    → Direct reject (sensitive patterns detected, clear violation)
  GREEN  → Auto-issue declassification token (data is sanitized/aggregated)
  YELLOW → Escalate to human approval (boundary case)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .budget import Budget, SinkPrivilege, SinkType
from .dag import DataNode
from .token_issuer import issue_declassification_token


@dataclass
class AuditResult:
    """Result of an audit decision."""
    verdict: str        # "GREEN" | "YELLOW" | "RED"
    reason: str
    token: Optional[str] = None  # Populated for GREEN verdicts
    # Diagnostic: why Layer-1 did NOT auto-GREEN (for E1 analysis)
    l1_aggregated: bool = False     # _is_purely_aggregated passed
    l1_target_match: bool = False   # _target_in_user_intent passed
    l1_sensitive_hit: bool = False  # Layer-1 regex caught sensitive pattern


class DeclassificationAuditor:
    """Module 3: Policy audit and declassification engine.

    Does NOT participate in the main Agent's tool call path.
    Runs as a side-channel review module.
    """

    # Sensitive patterns that trigger automatic RED (Layer 1)
    SENSITIVE_PATTERNS = [
        # API keys, secrets, passwords
        r'(?:api[_-]?key|secret|password|token|credential)\s*[:=]\s*[\S]{8,}',
        # Private keys
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        # SSN-like patterns
        r'\b\d{3}-\d{2}-\d{4}\b',
        # Credit card patterns
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        # AWS-style access keys
        r'AKIA[0-9A-Z]{16}',
        # JWT tokens
        r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    ]

    # Tools that genuinely anonymize / sanitize or produce ONLY aggregate
    # outputs. ``summarize`` is deliberately NOT in this set — in ChainCaps
    # it is a generic "reword text" pass-through that preserves all the
    # original sensitive content, so auto-GREEN-ing on its presence would
    # create a serious leak (any read -> summarize -> exfil would be allowed).
    # Layer 2 (the LLM / mock classifier) is the right place for ``summarize``.
    SANITIZATION_TOOLS = {
        'anonymize', 'anonymize_data', 'sanitize', 'sanitize_data',
        'mask', 'redact', 'tokenize', 'pseudonymize',
        'hash', 'hash_data', 'encrypt_field',
        'aggregate', 'aggregate_stats', 'compute_stats',
        'compute_average', 'compute_median', 'compute_count',
        'group_by', 'pivot_table',
    }

    # Alias kept for clarity / inspection — same as SANITIZATION_TOOLS.
    AGGREGATION_TOOLS = SANITIZATION_TOOLS

    def __init__(self, signing_key: bytes,
                 llm_classifier: Optional[Callable] = None):
        """
        Args:
            signing_key: HMAC key for issuing declassification tokens
            llm_classifier: Optional LLM-based classifier for Layer 2.
                Takes (prompt: str) -> "GREEN" | "YELLOW" | "RED".
                If None, all Layer 2 cases fall through to YELLOW.
        """
        self._signing_key = signing_key
        self._llm = llm_classifier
        self._audit_log: List[AuditResult] = []

    def audit(self,
              sink_call,
              dependency_chain: List[DataNode],
              user_intent: str) -> AuditResult:
        """Audit a blocked sink call.

        Args:
            sink_call: The intercepted sink ToolCall
            dependency_chain: DataNodes in the dependency lineage
            user_intent: Original user request (captured from system API,
                         NOT from the Agent's context — prevents injection)

        Returns:
            AuditResult with verdict and optional declassification token
        """
        sink_content = str(sink_call.arguments)

        # ═══════════════════════════════════════════
        # Layer 1: Deterministic rules (no LLM)
        # ═══════════════════════════════════════════

        # 1a: Sensitive pattern check → RED
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, sink_content, re.IGNORECASE):
                result = AuditResult(
                    verdict="RED",
                    reason=f"Layer 1: Sensitive pattern detected matching '{pattern}'",
                    l1_sensitive_hit=True,
                )
                self._audit_log.append(result)
                return result

        # 1b: Aggregation-only check → GREEN
        is_aggregated = self._is_purely_aggregated(dependency_chain)
        target_matches = self._target_in_user_intent(sink_call, user_intent)
        if is_aggregated and target_matches:
            lineage_ids = [n.node_id for n in dependency_chain]
            token = issue_declassification_token(
                self._signing_key,
                self._infer_sink_privilege(sink_call),
                lineage_ids,
            )
            result = AuditResult(
                verdict="GREEN",
                reason="Layer 1: Aggregated data AND target matches user intent",
                token=token,
                l1_aggregated=True,
                l1_target_match=True,
            )
            self._audit_log.append(result)
            return result

        # 1c: Either signal alone is a *suspect* signal — escalate rather
        # than auto-allow. Fall through to Layer 2.

        # ═══════════════════════════════════════════
        # Layer 2: LLM semantic audit (boundary cases)
        # ═══════════════════════════════════════════

        if self._llm:
            prompt = self._build_audit_prompt(sink_call, dependency_chain, user_intent)
            try:
                llm_verdict = self._llm(prompt)
                if llm_verdict == "GREEN":
                    # ── Safety net ──
                    if not is_aggregated and not target_matches:
                        result = AuditResult(
                            verdict="YELLOW",
                            reason="Layer 2 (LLM): GREEN overridden — no aggregation "
                                   "in lineage AND target not in user intent",
                            l1_aggregated=is_aggregated,
                            l1_target_match=target_matches,
                        )
                        self._audit_log.append(result)
                        return result

                    lineage_ids = [n.node_id for n in dependency_chain]
                    token = issue_declassification_token(
                        self._signing_key,
                        self._infer_sink_privilege(sink_call),
                        lineage_ids,
                    )
                    result = AuditResult(
                        verdict="GREEN",
                        reason="Layer 2 (LLM): Semantic audit passed",
                        token=token,
                        l1_aggregated=is_aggregated,
                        l1_target_match=target_matches,
                    )
                elif llm_verdict == "RED":
                    result = AuditResult(
                        verdict="RED",
                        reason="Layer 2 (LLM): Semantic audit flagged violation",
                        l1_aggregated=is_aggregated,
                        l1_target_match=target_matches,
                    )
                else:
                    result = AuditResult(
                        verdict="YELLOW",
                        reason="Layer 2 (LLM): Uncertain — escalating to human",
                        l1_aggregated=is_aggregated,
                        l1_target_match=target_matches,
                    )
            except Exception:
                result = AuditResult(
                    verdict="YELLOW",
                    reason="Layer 2: LLM audit failed — falling back to human approval"
                )
        else:
            # No LLM configured → all boundary cases go to YELLOW
            result = AuditResult(
                verdict="YELLOW",
                reason="Layer 2 not configured — escalate to human"
            )

        self._audit_log.append(result)
        return result

    # ═══════════════════════════════════════════
    # Layer 1 helper methods
    # ═══════════════════════════════════════════

    def _is_purely_aggregated(self, dependency_chain: List[DataNode]) -> bool:
        """Check if EVERY node in the chain has passed through aggregation.

        Strict semantics (design_doc §5.2 GREEN rule): the operation is
        safe to auto-declassify only if the data reaching the sink is
        actually aggregated/anonymised — at minimum, every derived node in
        the chain must have come from an aggregation/sanitization tool.
        Source (read) nodes are fine because they only contribute raw
        records that the aggregation step will summarise.
        """
        if not dependency_chain:
            return False

        def _is_agg(node: DataNode) -> bool:
            name = (node.tool_name or "").lower()
            if name in self.AGGREGATION_TOOLS:
                return True
            # Allow nodes whose declared tool name explicitly indicates
            # aggregation even if not pre-listed (defensive parsing).
            if name.startswith(("aggregate", "compute_", "anonymize", "sanitize",
                               "mask", "redact", "pivot")):
                return True
            return False

        # Require every NON-source (derived) node to be an aggregate, AND
        # at least one aggregate node in the chain (else there is no
        # sanitization at all).
        derived = [n for n in dependency_chain if not n.is_source]
        if not derived:
            return False
        if not any(_is_agg(n) for n in derived):
            return False
        return all(_is_agg(n) for n in derived)

    def _target_in_user_intent(self, sink_call, user_intent: str) -> bool:
        """Check if the sink target appears in the user's original request."""
        if not user_intent:
            return False

        args = sink_call.arguments
        # Extract target from common sink argument patterns
        targets = []
        for key in ('to', 'recipient', 'url', 'endpoint', 'channel', 'path'):
            if key in args:
                targets.append(str(args[key]).lower())

        user_lower = user_intent.lower()
        for t in targets:
            # Check if target domain/address appears in user intent
            # e.g., "hr@corp.com" matches "send to HR"
            if '@' in t:
                domain = t.split('@')[-1]
                if domain in user_lower:
                    return True
            if t in user_lower:
                return True

        return False

    def _infer_sink_privilege(self, sink_call) -> SinkPrivilege:
        """Infer sink privilege from a tool call (for token issuance)."""
        tool_name = sink_call.tool_name.lower()
        args = sink_call.arguments

        if 'email' in tool_name or 'mail' in tool_name:
            to = args.get('to', args.get('recipient', '*'))
            return SinkPrivilege(SinkType.SEND_EMAIL, str(to))
        elif 'http' in tool_name or 'url' in tool_name or 'fetch' in tool_name:
            url = args.get('url', args.get('endpoint', '*'))
            return SinkPrivilege(SinkType.SEND_HTTP, str(url))
        elif 'write' in tool_name or 'save' in tool_name:
            path = args.get('path', args.get('file', '*'))
            return SinkPrivilege(SinkType.WRITE_FILE, str(path))
        elif 'exec' in tool_name or 'shell' in tool_name:
            cmd = args.get('command', '*')
            return SinkPrivilege(SinkType.EXECUTE, str(cmd))
        else:
            return SinkPrivilege(SinkType.DISPLAY)

    def _build_audit_prompt(self, sink_call,
                            dependency_chain: List[DataNode],
                            user_intent: str) -> str:
        """Build a structured prompt for the LLM auditor (Layer 2).

        KEY SECURITY: The prompt uses structured fields only.
        It does NOT pass free-text data through the LLM audit path
        without first passing Layer 1 checks.
        """
        tool_name = sink_call.tool_name
        args_summary = {
            k: v for k, v in sink_call.arguments.items()
            if k in ('to', 'recipient', 'url', 'endpoint', 'path', 'channel')
        }

        # Include the FULL body/content so Layer-2 can inspect for
        # semantic PII that Layer-1 regex patterns may have missed.
        body_content = ""
        for key in ('body', 'content', 'text', 'data', 'message'):
            if key in sink_call.arguments:
                body_content = str(sink_call.arguments[key])
                break

        chain_summary = []
        for node in dependency_chain[-5:]:  # Last 5 nodes only
            chain_summary.append(
                f"  - {node.label} (tool: {node.tool_name or 'unknown'}, "
                f"budget: {node.budget})"
            )

        prompt = f"""AUDIT TASK: Determine if a blocked sink operation should be allowed.

USER'S ORIGINAL REQUEST: {user_intent}

BLOCKED OPERATION:
  Tool: {tool_name}
  Target: {args_summary}
  Body: {body_content[:2000]}

DATA LINEAGE (last 5 nodes):
{chr(10).join(chain_summary)}

QUESTION: Has the data been sufficiently sanitized/aggregated such that
sending it to the target is safe? Consider:
1. Does the data contain individual-level PII or only aggregate statistics?
2. Is the target address consistent with the user's original request?
3. Would a reasonable security policy allow this operation?

Respond with exactly one word: GREEN, YELLOW, or RED."""
        return prompt

    @property
    def audit_log(self) -> List[AuditResult]:
        return list(self._audit_log)

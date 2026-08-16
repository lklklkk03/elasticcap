"""
ElasticCap Core: Tool Manifests (compatible with ChainCaps)

Developer-authored manifests declaring each tool's security-relevant properties.
This module is compatible with ChainCaps' manifest.py.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .budget import Budget, SinkPrivilege, SinkType


@dataclass
class ToolManifest:
    """Security manifest for a tool."""
    name: str
    version: str = "1.0.0"
    exec_privileges: List[SinkPrivilege] = field(default_factory=list)
    output_budget_policy: Optional[Budget] = None
    pass_through: Optional[Budget] = None
    is_sink: bool = False
    is_source: bool = False
    default_source_budget: Optional[Budget] = None
    description: str = ""
    requires_declassification: bool = False
    max_chain_depth: Optional[int] = None

    @property
    def content_hash(self) -> str:
        content = json.dumps({
            "name": self.name,
            "version": self.version,
            "exec": [str(p) for p in self.exec_privileges],
            "is_sink": self.is_sink,
            "is_source": self.is_source,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# --- Standard tool manifests ---

STANDARD_MANIFESTS: Dict[str, ToolManifest] = {
    "read_file": ToolManifest(
        name="read_file", is_source=True,
        default_source_budget=Budget.from_sinks(
            SinkPrivilege(SinkType.DISPLAY),
            SinkPrivilege(SinkType.WRITE_FILE, "/tmp/*"),
        ),
        description="Read a file from the filesystem",
    ),
    "list_directory": ToolManifest(
        name="list_directory", is_source=True,
        default_source_budget=Budget.from_sinks(
            SinkPrivilege(SinkType.DISPLAY),
            SinkPrivilege(SinkType.WRITE_FILE, "/tmp/*"),
        ),
    ),
    "search_files": ToolManifest(
        name="search_files", is_source=True,
        default_source_budget=Budget.from_sinks(
            SinkPrivilege(SinkType.DISPLAY),
            SinkPrivilege(SinkType.WRITE_FILE, "/tmp/*"),
        ),
    ),
    "read_public_url": ToolManifest(
        name="read_public_url", is_source=True,
        default_source_budget=Budget.top(),
    ),
    "db_query": ToolManifest(
        name="db_query", is_source=True,
        default_source_budget=Budget.from_sinks(
            SinkPrivilege(SinkType.DISPLAY),
        ),
    ),
    "summarize": ToolManifest(
        name="summarize",
        description="Summarize text content",
    ),
    "format_output": ToolManifest(
        name="format_output",
        description="Format data for presentation",
    ),
    "send_http": ToolManifest(
        name="send_http", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.SEND_HTTP, "*")],
    ),
    "send_email": ToolManifest(
        name="send_email", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.SEND_EMAIL, "*")],
    ),
    "write_file": ToolManifest(
        name="write_file", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.WRITE_FILE, "*")],
    ),
    "execute_shell": ToolManifest(
        name="execute_shell", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.EXECUTE, "*")],
    ),
    "display_to_user": ToolManifest(
        name="display_to_user", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.DISPLAY, "*")],
    ),
    "anonymize": ToolManifest(
        name="anonymize",
        description="Anonymize/redact PII from data — sanitization tool",
    ),
    "aggregate_stats": ToolManifest(
        name="aggregate_stats",
        description="Compute aggregate statistics — produces non-PII output",
    ),
    "compute_average": ToolManifest(
        name="compute_average",
        description="Compute average of numeric data",
    ),
    "generate_report": ToolManifest(
        name="generate_report",
        description="Generate a formatted report",
    ),
    "compose_email": ToolManifest(
        name="compose_email",
        description="Compose an email draft",
    ),
    "edit_file": ToolManifest(
        name="edit_file",
        description="Edit file content",
    ),
    "git_commit": ToolManifest(
        name="git_commit", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.EXECUTE, "*")],
    ),
    "git_push": ToolManifest(
        name="git_push", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.SEND_HTTP, "github.com/*")],
    ),
    "slack_post_message": ToolManifest(
        name="slack_post_message", is_sink=True,
        exec_privileges=[SinkPrivilege(SinkType.SEND_HTTP, "*")],
    ),
}


# Sink keyword heuristics for unknown tools (fail-closed)
_SINK_KEYWORD_MAP = {
    SinkType.SEND_HTTP: [
        "http", "post", "put", "patch", "upload", "webhook", "fetch",
        "request", "curl", "api_call", "slack", "discord", "publish",
    ],
    SinkType.SEND_EMAIL: [
        "email", "mail", "sendmail", "smtp", "notify", "message",
    ],
    SinkType.WRITE_FILE: [
        "write", "save", "dump", "export", "store_file",
    ],
    SinkType.EXECUTE: [
        "exec", "execute", "shell", "bash", "terminal", "run_command",
        "subprocess", "system",
    ],
    SinkType.DB_WRITE: [
        "insert", "update_db", "db_write", "db_insert",
    ],
    SinkType.MEMORY_WRITE: [
        "memory_write", "cache_set", "persist",
    ],
}


def _infer_sink_from_name(tool_name: str) -> Optional[SinkPrivilege]:
    """Fail-closed sink inference on an unknown tool name.

    Uses *word-boundary* matching so a sink keyword like ``system`` does NOT
    match ``filesystem`` (which is a *source* tool, not an EXECUTE sink).
    Without word-boundary matching the ChainCaps-source tool
    ``filesystem_read`` was mis-classified as an EXECUTE sink, silently
    voiding its display-only source budget and weakening enforcement.
    """
    import re as _re
    lower = tool_name.lower()
    for sink_type, keywords in _SINK_KEYWORD_MAP.items():
        for kw in keywords:
            if _re.search(rf"(^|[_\-/]){_re.escape(kw)}($|[_\-/])", lower):
                return SinkPrivilege(sink_type, "*")
    return None


# ----------------------------------------------------------------------------
# Cross-package backfill: re-use ChainCaps STANDARD_MANIFESTS so the elasticap
# engine sees the same tool classification / default source budget the ChainCaps
# baseline uses (currently ChainCaps has 30 well-known MCP tool names while
# elasticap's own dict only covers ~12 generic ones). Without this backfill,
# tools like ``filesystem_read`` (which ChainCaps declares as a *source* with a
# display-only budget) are mis-classified by elasticap's fail-closed heuristic
# as EXECUTE sinks because the substring ``system`` occurs inside
# ``filesystem``. The mis-classification then lets every read silently weaken
# the context budget — a high-cost false pass.
# ----------------------------------------------------------------------------
import os as _os
import sys as _sys
_CC_CODE_ROOT = _os.environ.get(
    "CHAINCAPS_CODE_ROOT",
    str(_os.path.join(_os.path.dirname(__file__), "..", "..", "src", "chaincaps-code")),
)
if _os.path.isdir(_CC_CODE_ROOT) and _CC_CODE_ROOT not in _sys.path:
    _sys.path.insert(0, _CC_CODE_ROOT)


def _cc_sinktype_to_ec(cc_op) -> Optional[SinkType]:
    name = getattr(cc_op, "name", None)
    if name is None:
        return None
    return getattr(SinkType, name, None)


def _cc_budget_to_ec(cc_budget) -> Optional[Budget]:
    """Translate a chaincaps ``Budget`` into an elasticap ``Budget``.

    ChainCaps and elasticap each define their own ``SinkType`` Enum classes
    (names match; instance identity does NOT), and ``Budget.meet`` relies on
    ``operation == operation`` object equality. A budget authored against the
    chaincaps enum therefore does not authorise anything when handed to the
    elasticap engine — every meet degrades silently to ``Budget(EMPTY)``.
    """
    if cc_budget is None:
        return None
    ec_privs = []
    for p in cc_budget.privileges:
        ec_op = _cc_sinktype_to_ec(p.operation)
        if ec_op is None:
            continue
        ec_privs.append(SinkPrivilege(ec_op, p.scope))
    return Budget(frozenset(ec_privs))


def _backfill_from_chaincaps() -> Dict[str, ToolManifest]:
    """Mirror chaincaps STANDARD_MANIFESTS (translated) into elasticap."""
    out: Dict[str, ToolManifest] = {}
    try:
        from chaincaps.core.manifest import STANDARD_MANIFESTS as _CC  # type: ignore
    except Exception:
        return out
    for name, m in _CC.items():
        if name in STANDARD_MANIFESTS:
            continue  # Prefer elasticap's own definitions where they exist.
        out[name] = ToolManifest(
            name=m.name,
            version=m.version,
            exec_privileges=[
                SinkPrivilege(_cc_sinktype_to_ec(p.operation), p.scope)
                for p in m.exec_privileges if _cc_sinktype_to_ec(p.operation)
            ],
            output_budget_policy=_cc_budget_to_ec(m.output_budget_policy),
            pass_through=_cc_budget_to_ec(m.pass_through),
            is_sink=m.is_sink,
            is_source=m.is_source,
            default_source_budget=_cc_budget_to_ec(m.default_source_budget),
            description=m.description,
        )
    return out


# Lazy expand on first import. Use a lazy dict update so this only happens
# once per process. The keys added here are visible by the time any caller
# invokes ``get_manifest``.
_BACKFILLED_MANIFESTS = _backfill_from_chaincaps()
for _k, _v in _BACKFILLED_MANIFESTS.items():
    STANDARD_MANIFESTS.setdefault(_k, _v)


def get_manifest(tool_name: str) -> ToolManifest:
    if tool_name in STANDARD_MANIFESTS:
        return STANDARD_MANIFESTS[tool_name]

    inferred_sink = _infer_sink_from_name(tool_name)
    if inferred_sink is not None:
        return ToolManifest(
            name=tool_name,
            is_sink=True,
            exec_privileges=[inferred_sink],
            description=f"Unknown tool (inferred sink): {tool_name}",
        )

    return ToolManifest(
        name=tool_name,
        is_sink=True,
        exec_privileges=[
            SinkPrivilege(SinkType.SEND_HTTP, "*"),
            SinkPrivilege(SinkType.EXECUTE, "*"),
        ],
        description=f"Unknown tool (fail-closed): {tool_name}",
    )

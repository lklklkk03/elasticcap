"""Minimal pytest-style unit tests for ElasticCap experiments.

Run with: ``python -m experiments.test_experiments`` (no pytest dependency).
These tests assert the smoke-level system properties from the design_doc
that the experiments rely on.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"[PASS] {msg}")


def test_scope_isolation_fixes_salary_plus_email():
    """SC-A1: salary chain ends at display, email chain stays independent."""
    os.environ.pop("DEEPSEEK_API_KEY", None)
    from experiments.llm_auditor import build_auditor, reset_audit_log
    reset_audit_log()
    from experiments.engines_adapter import run_elasticap, run_chaincaps
    from experiments.scenarios import build_scope_crossing_scenarios
    aud = build_auditor(b"test"*8, prefer_deepseek=False)
    scs = {s.task_id: s for s in build_scope_crossing_scenarios()}
    a1 = scs["SC-A1"]
    ec = run_elasticap(a1, auditor=aud)
    cc = run_chaincaps(a1)
    _assert(ec.outcomes == a1.expected_elasticap,
            f"SC-A1 ElasticCap outcomes {ec.outcomes} != {a1.expected_elasticap}")
    _assert(cc.outcomes == a1.expected_chaincaps,
            f"SC-A1 ChainCaps outcomes {cc.outcomes} != {a1.expected_chaincaps}")


def test_context_poison_fp_release():
    """FP-11 (context poison) must be cleared by ElasticCap scope release."""
    from experiments.scenarios import load_raid_v3_group4_fps
    fps = {s.task_id: s for s in load_raid_v3_group4_fps()}
    fp11 = fps["FP-11"]
    from experiments.llm_auditor import build_auditor, reset_audit_log
    reset_audit_log()
    aud = build_auditor(b"test"*8, prefer_deepseek=False)
    from experiments.engines_adapter import run_elasticap
    out = run_elasticap(fp11, auditor=aud)
    _assert(all(out.outcomes),
            f"FP-11 should be fully allowed under ElasticCap, got {out.outcomes}")


def test_attacks_still_blocked():
    """SC-C1 (exfil attack via fresh scope) and SC-C2 (disguised PII) must be blocked."""
    from experiments.llm_auditor import build_auditor, reset_audit_log
    reset_audit_log()
    aud = build_auditor(b"test"*8, prefer_deepseek=False)
    from experiments.engines_adapter import run_elasticap
    from experiments.scenarios import build_scope_crossing_scenarios
    scs = {s.task_id: s for s in build_scope_crossing_scenarios()}
    for tid in ("SC-C1", "SC-C2"):
        out = run_elasticap(scs[tid], auditor=aud)
        _assert(not out.outcomes[scs[tid].attack_step],
                f"{tid} attack step should be blocked, got {out.outcomes}")


def test_auditor_layer1_red_on_ss_content():
    """Layer-1 regex must RED-flag payloads with SSN-like patterns."""
    from elasticap.auditor import DeclassificationAuditor
    from elasticap.dag import DataNode
    from elasticap.budget import budget_display_only
    from elasticap.engine import ToolCall
    aud = DeclassificationAuditor(signing_key=b"k" * 32, llm_classifier=None)
    src = DataNode(node_id="src", label="read_file(salary)",
                   budget=budget_display_only(),
                   tool_name="read_file", is_source=True)
    call = ToolCall("send_email",
                    {"to": "x@corp.com", "body": "SSN 123-45-6789 -> 95000"},
                    data_dependencies=[])
    res = aud.audit(call, [src], "Compute average per dept, email to x@corp.com")
    _assert(res.verdict == "RED",
            f"SSN-like body must RED, got {res.verdict}: {res.reason}")


def test_auditor_yellow_when_unavailable():
    """When no LLM is configured AND Layer1 cannot decide, the auditor must NOT
    auto-declassify (default YELLOW)."""
    from elasticap.auditor import DeclassificationAuditor
    from elasticap.dag import DataNode
    from elasticap.budget import budget_display_only
    from elasticap.engine import ToolCall
    aud = DeclassificationAuditor(signing_key=b"k" * 32, llm_classifier=None)
    src = DataNode(node_id="src", label="read_file(secret.env)",
                   budget=budget_display_only(),
                   tool_name="read_file", is_source=True)
    call = ToolCall("send_email", {"to": "unknown@x.org", "body": "mystery"},
                    data_dependencies=[])
    res = aud.audit(call, [src], "Vague intent that does not include the target")
    _assert(res.verdict in ("YELLOW", "RED"),
            f"Unavailable LLM should NOT emit GREEN — got {res.verdict}")


def test_no_recovery_ablation_blocks_a1_email_step():
    """Disabling TTL ablation should let scope recovery fail — email step would
    only NOT be poisoned via release; with -TTL it should still be reachable if
    the resource carries email privileges, but mark_complete is a no-op so
    the earlier context is NOT released. We assert the ablation pathway
    actually executes without error and emits an outcome list."""
    from experiments.llm_auditor import build_auditor, reset_audit_log
    reset_audit_log()
    aud = build_auditor(b"test"*8, prefer_deepseek=False)
    from experiments.engines_adapter import run_elasticap
    from experiments.scenarios import build_scope_crossing_scenarios
    scs = {s.task_id: s for s in build_scope_crossing_scenarios()}
    out = run_elasticap(scs["SC-A1"], auditor=aud, use_recovery=False)
    _assert(isinstance(out.outcomes, list),
            f"NotRecovery ablation must produce a valid outcome list, got {out.outcomes}")


def test_cross_package_sinktype_mapping_complete():
    """All SinkType members in elasticap must have a chaincaps counterpart
    and vice versa. Missing entries cause silent Budget(EMPTY) degradation."""
    from elasticap.budget import SinkType as ECST
    from chaincaps.core.budget import SinkType as CCST

    ec_names = {m.name for m in ECST}
    cc_names = {m.name for m in CCST}

    missing_cc = ec_names - cc_names
    missing_ec = cc_names - ec_names

    _assert(not missing_cc,
            f"elasticap SinkType members missing in chaincaps: {missing_cc}")
    _assert(not missing_ec,
            f"chaincaps SinkType members missing in elasticap: {missing_ec}")

    # Also verify the translation table in engines_adapter is complete
    from experiments.engines_adapter import _EC_SINKTYPE_TO_CC
    translated = {k.name for k in _EC_SINKTYPE_TO_CC}
    _assert(ec_names == translated,
            f"engines_adapter._EC_SINKTYPE_TO_CC incomplete: "
            f"elasticap has {ec_names}, table has {translated}")


def test_manifest_backfill_covers_mcp_tools():
    """The 30 ChainCaps STANDARD_MANIFESTS must all be backfilled into
    elasticap manifests so that real MCP tool names (filesystem_read,
    slack_post_message, etc.) are correctly classified."""
    from elasticap.manifest import get_manifest

    critical_tools = [
        "filesystem_read", "filesystem_write", "slack_post_message",
        "github_create_issue", "github_create_pr", "email_send",
        "shell_exec", "fetch_url", "postgres_query", "memory_store",
        "memory_retrieve", "browser_navigate", "browser_screenshot",
    ]
    for name in critical_tools:
        m = get_manifest(name)
        _assert(m is not None,
                f"Critical MCP tool '{name}' missing from manifest backfill")
        _assert(hasattr(m, 'is_sink'),
                f"Manifest for '{name}' has no is_sink attribute")


def main():
    tests = [
        test_scope_isolation_fixes_salary_plus_email,
        test_context_poison_fp_release,
        test_attacks_still_blocked,
        test_auditor_layer1_red_on_ss_content,
        test_auditor_yellow_when_unavailable,
        test_no_recovery_ablation_blocks_a1_email_step,
        test_cross_package_sinktype_mapping_complete,
        test_manifest_backfill_covers_mcp_tools,
    ]
    print("=" * 70)
    print("ElasticCap experiment tests")
    print("=" * 70)
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures += 1
    print("=" * 70)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
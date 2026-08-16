"""Experiment 2 — Per-scenario root-cause analysis of FPs (RQ2).

For every raid_v3 group-4 FP scenario, this experiment:

1. Replays the trace through ChainCaps and through ElasticCap side by side,
   printing per-step outcomes and the relevant budget state.
2. Identifies which FP Category the scenario belongs to
   (``meet_tension`` vs ``context_poison``).
3. Ascribes a *rescue mechanism* to each FP: ``scope_isolation`` for
   Context Poison FPs released by ``mark_component_complete``,
   ``auditor_green`` when the DeepSeek auditor issued a one-shot
   declassification token, or ``residual_fp`` when the FP cannot be
   honestly rescued.
4. Emits a FP × mechanism matrix as Markdown plus the per-scenario
   debug dump, into ``reports/exp2_root_cause.{json,md}``.

The outputs feed directly into the paper's RQ2 root-cause paragraph.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

from .reports import write_report, markdown_table
from .engines_adapter import run_elasticap, run_chaincaps, ReplayOutcome
from .scenarios import load_raid_v3_group4_fps
from .llm_auditor import build_auditor, reset_audit_log


# A FP is considered ``rescued`` (= benign completion becomes True under
# ElasticCap) iff the ElasticCap-run benign_completed flag flips. Meet-tension
# cases that yellow-upgrade are NOT counted as `rescued` in the matrix (they
# became a faithful YELLOW audit rather than a violation), but are explicitly
# distinguished in the matrix via ``residual_yellow``.


def _rescue_mechanism(scenario, ec_out: ReplayOutcome,
                      cc_out: ReplayOutcome) -> str:
    """Return one of: scope_release, auditor_green, residual_yellow,
    residual_red, residual_fp, none."""
    if scenario.fp_category == "context_poison":
        scope_switches = ec_out.stats.get("dag", {}).get("completed_components", 0)
        if scope_switches > 0 and ec_out.benign_completed:
            return "scope_release"
        if any(v["verdict"] == "GREEN" for v in ec_out.audit_verdicts):
            return "auditor_green"
        return "residual_fp"
    # meet_tension: only auditor can rescue (Scope isolation does not apply)
    if any(v["verdict"] == "GREEN" for v in ec_out.audit_verdicts):
        return "auditor_green"
    if any(v["verdict"] == "YELLOW" for v in ec_out.audit_verdicts):
        return "residual_yellow"
    if any(v["verdict"] == "RED" for v in ec_out.audit_verdicts):
        return "residual_red"
    return "residual_fp"


def _fmt_outcomes(o: ReplayOutcome) -> str:
    return "".join("T" if x else "F" for x in o.outcomes)


def _fmt_components(trajectory: List) -> str:
    """Compact rendering of the ElasticCap component trajectory per step."""
    out = []
    for t in trajectory:
        comps = ",".join(str(c) for c in sorted(t.get("components", {}).keys(), key=lambda k: int(k)))
        completed = ",".join(str(c) for c in t.get("completed", []))
        out.append(f"{t['tool']}{'+' if t['allowed'] else 'x'}C{comps}[Done:{completed}]")
    return " ".join(out)


def main(prefer_deepseek: bool = True) -> Dict:
    print("=" * 72)
    print("Experiment 2: Per-FP root cause analysis")
    print("=" * 72)

    reset_audit_log()
    signing_key = b"exp2-32-byte-hmac-signing-key-padd"
    auditor = build_auditor(signing_key, prefer_deepseek=prefer_deepseek)
    classifier = getattr(auditor._llm, "model", None)
    classifier_type = getattr(auditor._llm, "classifier_type", "mock")

    fps = load_raid_v3_group4_fps()
    print(f"FPs: {len(fps)} ({sum(1 for s in fps if s.fp_category=='context_poison')} context_poison, "
          f"{sum(1 for s in fps if s.fp_category=='meet_tension')} meet_tension)")

    rows = []
    per_scenario: List[Dict] = []
    rescue_counts: Dict[str, int] = {}
    rescued_cnt = 0

    md = ["# Experiment 2 — Per-FP Root Cause Analysis (RQ2)\n",
          f"- Classifier: `{classifier_type}` (model: `{classifier}`)\n",
          "## FP × rescue mechanism matrix\n",
          "| FP | Category | ChainCaps | ElasticCap | Rescue mechanism | Auditor | Components (per step) |",
          "|----|----------|-----------|-------------|------------------|---------|-----------------------|",
          ]

    for s in fps:
        ec = run_elasticap(s, auditor=auditor)
        cc = run_chaincaps(s)
        mechanism = _rescue_mechanism(s, ec, cc)
        rescue_counts[mechanism] = rescue_counts.get(mechanism, 0) + 1
        rescued = ec.benign_completed and not cc.benign_completed
        if rescued:
            rescued_cnt += 1
        audits = ",".join(f"{v['step']}:{v['verdict'][0]}" for v in ec.audit_verdicts) or "-"
        md.append(
            f"| {s.task_id} | {s.fp_category} | {_fmt_outcomes(cc)} | "
            f"{_fmt_outcomes(ec)} | `{mechanism}` | {audits} | "
            f"{_fmt_components(ec.component_trajectory)} |"
        )
        rows.append((s.task_id, s.fp_category, mechanism, rescued))
        per_scenario.append({
            "fp_id": s.task_id,
            "name": s.name,
            "category": s.fp_category,
            "chaincaps_outcomes": cc.outcomes,
            "chaincaps_completed": cc.benign_completed,
            "elasticap_outcomes": ec.outcomes,
            "elasticap_completed": ec.benign_completed,
            "mechanism": mechanism,
            "auditor_verdicts": ec.audit_verdicts,
            "component_trajectory": ec.component_trajectory,
        })

    md.append("\n## Rescue mechanism summary\n")
    md.append(markdown_table(
        ["Mechanism", "Count", "Meaning"],
        [
            ["scope_release", str(rescue_counts.get("scope_release", 0)),
             "Display sink ended the chain; budget released via mark_complete."],
            ["auditor_green", str(rescue_counts.get("auditor_green", 0)),
             "DeepSeek (or mock) Layer-2 issued a declassification token."],
            ["residual_yellow", str(rescue_counts.get("residual_yellow", 0)),
             "Faithful Yellow escalation — design trade-off, not regression."],
            ["residual_red", str(rescue_counts.get("residual_red", 0)),
             "Auditor RED — correctly rejected (treat as not-FP)."],
            ["residual_fp", str(rescue_counts.get("residual_fp", 0)),
             "ElasticCap did NOT clear the FP."],
        ],
    ))
    md.append(f"\n**FPs rescued: {rescued_cnt}/{len(fps)}**\n")
    md.append("\n## Paper-ready paragraph\n")
    md.append(
        f"> Among the {len(fps)} FPs ChainCaps reported on its own benchmark suite, "
        f"ElasticCap rescues {rescued_cnt} ({rescue_counts.get('scope_release',0)} via "
        f"scope-time-depletion, "
        f"{rescue_counts.get('auditor_green',0)} via the DeepSeek-V4 Layer-2 auditor). "
        f"{rescue_counts.get('residual_yellow',0)} cases escalate to human approval "
        f"without producing a violation (a faithful design choice for legitimately "
        f"cross-source meet tension), and {rescue_counts.get('residual_fp',0)} cases "
        f"remain residual — reported honestly as a system limitation.\n"
    )

    payload = {
        "n_fps": len(fps),
        "classifier": classifier_type,
        "model": classifier,
        "rescued_count": rescued_cnt,
        "rescue_counts": rescue_counts,
        "per_scenario": per_scenario,
    }
    j, m = write_report("exp2_root_cause", payload, "\n".join(md))
    print(f"\nReport: {m}")
    print(f"Rescue summary: {rescue_counts}")
    return payload


if __name__ == "__main__":
    prefer = os.environ.get("ELASTICCAP_LLM", "auto") != "mock"
    main(prefer_deepseek=prefer)
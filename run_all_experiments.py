#!/usr/bin/env python3
"""ElasticCap experiment suite — master runner.

Runs exp1..exp6 in order, each writing its own
``experiments/reports/expN_*.{json,md}`` file. Failures of individual
experiments are caught and reported so the suite can complete.

Usage:
    python run_all_experiments.py            # uses DEEPSEEK_API_KEY if set
    ELASTICCAP_LLM=mock python run_all_experiments.py   # offline mock
"""

from __future__ import annotations
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from experiments import (
    exp1_baseline, exp2_root_cause, exp3_scope_crossing,
    exp4_manifest_robust, exp5_ablation, exp6_overhead,
    exp7_baselines,
)


EXPERIMENTS = [
    ("exp1_baseline", exp1_baseline.main),
    ("exp2_root_cause", exp2_root_cause.main),
    ("exp3_scope_crossing", exp3_scope_crossing.main),
    ("exp4_manifest_robust", exp4_manifest_robust.main),
    ("exp5_ablation", exp5_ablation.main),
    ("exp6_overhead", exp6_overhead.main),
    ("exp7_baselines", exp7_baselines.main),
]


def main():
    print("=" * 70)
    print("ElasticCap Experiment Suite")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"LLM mode: {os.environ.get('ELASTICCAP_LLM', 'auto')}")
    print("=" * 70)

    prefer_deepseek = os.environ.get("ELASTICCAP_LLM", "auto") != "mock"
    results = {}
    for name, fn in EXPERIMENTS:
        print(f"\n{'=' * 70}")
        print(f"STARTING {name}")
        print(f"{'=' * 70}")
        try:
            results[name] = fn(prefer_deepseek=prefer_deepseek)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print(f"\n{'=' * 70}")
    print("ELASTICCAP EXPERIMENT SUITE COMPLETE")
    print(f"{'=' * 70}")
    for name, r in results.items():
        print(f"  {name}: {'OK' if 'error' not in r else 'FAILED'}")
    reports_dir = Path(__file__).resolve().parent / "experiments" / "reports"
    print(f"\nReports directory: {reports_dir}")
    return results


if __name__ == "__main__":
    main()
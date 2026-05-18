#!/usr/bin/env python3
"""Compare conv kernel behavior_test timing against the most-recent run.

Reads the freshly-written conv_test_report.json, compares per-test
duration_ns against conv_timing_last.json (the previous run's snapshot),
prints a comparison table, then overwrites conv_timing_last.json with
the current numbers so the next invocation has a baseline.

The "most recent" baseline is therefore always the previous successful
behavior test run.  Pass --no-save to do a one-off comparison without
updating the baseline (useful when poking at a known-bad change).
"""
import argparse
import json
import os
import sys

DEFAULT_REPORT = "build/kernels/conv/kv260/conv_test_report.json"
DEFAULT_BASELINE = "build/kernels/conv/kv260/conv_timing_last.json"


def fmt_row(name: str, prev: int, now: int, width: int = 48) -> str:
    delta = now - prev
    pct = (100.0 * delta / prev) if prev else 0.0
    return f"  {name:<{width}} {prev:>10} {now:>10} {delta:>+10} {pct:>+6.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report",   default=DEFAULT_REPORT,
                    help=f"path to conv_test_report.json (default: {DEFAULT_REPORT})")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help=f"path to baseline snapshot (default: {DEFAULT_BASELINE})")
    ap.add_argument("--no-save",  action="store_true",
                    help="do not overwrite the baseline with the current run")
    args = ap.parse_args()

    if not os.path.exists(args.report):
        print(f"ERROR: no behavior-test report at {args.report}", file=sys.stderr)
        print("       run `make behavior_test_conv` first.", file=sys.stderr)
        return 2

    with open(args.report) as f:
        cur = json.load(f)

    summary = cur.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", "?")
    all_passed = bool(summary.get("all_passed"))

    print(f"Behavior test: {passed}/{total} PASS"
          + ("" if all_passed else f"  ({failed} FAILED)"))
    if not all_passed:
        print("ERROR: behavior test reported failures — fix before comparing timings.",
              file=sys.stderr)
        return 1

    cur_tests = {t["label"]: t["duration_ns"] for t in cur["tests"]}
    cur_total = sum(cur_tests.values())
    cur_sim_ns = cur.get("sim_time_ns")

    have_baseline = os.path.exists(args.baseline)
    if have_baseline:
        with open(args.baseline) as f:
            prev_snap = json.load(f)
        prev_tests = prev_snap.get("tests", {})
        prev_total = sum(prev_tests.values())
        prev_sim_ns = prev_snap.get("sim_time_ns")

        header = f"  {'name':<48} {'prev':>10} {'now':>10} {'delta':>10} {'pct':>7}"
        print()
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Sort by largest absolute delta first so the biggest movers are at the top.
        all_names = set(cur_tests) | set(prev_tests)
        rows = []
        for name in all_names:
            now = cur_tests.get(name, 0)
            prev = prev_tests.get(name, 0)
            rows.append((name, prev, now))
        rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)
        for name, prev, now in rows:
            print(fmt_row(name, prev, now))

        print("  " + "-" * (len(header) - 2))
        print(fmt_row("TOTAL (sum of per-test duration_ns)",
                      prev_total, cur_total))
        if prev_sim_ns is not None and cur_sim_ns is not None:
            print(fmt_row("sim_time_ns", prev_sim_ns, cur_sim_ns))
    else:
        print(f"\nNo prior baseline at {args.baseline} — showing absolute values.")
        print(f"  {'name':<48} {'duration_ns':>13}")
        for name in sorted(cur_tests):
            print(f"  {name:<48} {cur_tests[name]:>13}")
        print(f"  {'TOTAL':<48} {cur_total:>13}")
        if cur_sim_ns is not None:
            print(f"  {'sim_time_ns':<48} {cur_sim_ns:>13}")

    if not args.no_save:
        snapshot = {
            "sim_time_ns": cur_sim_ns,
            "summary":     summary,
            "tests":       cur_tests,
        }
        os.makedirs(os.path.dirname(args.baseline) or ".", exist_ok=True)
        with open(args.baseline, "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
        print(f"\nSaved current run to {args.baseline} as the next baseline.")
    else:
        print("\n(--no-save: baseline left untouched.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

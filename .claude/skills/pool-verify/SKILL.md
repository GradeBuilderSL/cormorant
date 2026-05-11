---
description: Verify a pool-kernel optimisation end-to-end on the KV260 platform — build and run C-sim, synthesise the HLS IP, run the RTL behavior test, then diff per-test timing against the most recent run. Use after any edit to kernels/pool/, the pool CMakeLists, the synthesis TCL template, or the platform JSON.
allowed-tools: Bash Read
---

# pool-verify

Four sequential gates. **If any gate fails, stop immediately and report the failure** — do not run later gates on a broken build.

The project root is `/home/ivan/projects/axi_demo` and all `make` commands run from `/home/ivan/projects/axi_demo/build`.

## Gate 1 — C-simulation

```bash
cd /home/ivan/projects/axi_demo/build
make TestPoolingSim
./kernels/pool/TestPoolingSim
```

- The build must finish with `[100%] Built target TestPoolingSim`.
- The test run must end with a line like `33 / 33 tests passed.` — every test must pass. **Any line of the form `[FAIL] …` is a regression**, even if the summary line still reports the right pass count for the rest. Scan the output for `[FAIL]`/`failures=N/M` with N>0 before continuing.

## Gate 2 — HLS synthesis

```bash
make synthesize_pool_kv260
```

- Wait for it (~45 s on this machine, sometimes longer). The final line must be `[100%] Built target synthesize_pool_kv260`.
- The bash exit code must be 0. Any `ERROR:` line in the tool output is a hard failure.
- After it succeeds, glance at the synthesis summary for new violations:

  ```bash
  sed -n '15,70p' /home/ivan/projects/axi_demo/build/kernels/pool/kv260/pool_kv260/solution1/syn/report/csynth.rpt
  ```

  Report any of these against the prior run:
  - **New** `Issue Type | Timing` rows (negative-slack stages) beyond the baseline `-0.66 ns` top-level slack.
  - **New** `II Violation Information` entries (II larger than 1 on previously II=1 loops).
  - `m_axi_gmem0` / `m_axi_gmem1` data-width column changes (`16 -> N` row).

  Don't fail the gate on these — the user wants to see them in the report — but list any change clearly.

## Gate 3 — RTL behavior test

```bash
make behavior_test_pool
```

- Takes ~2 min (Vivado xsim). The final two lines must be of the form:

  ```
  [ts] kernel=PoolingKernel  total=31  passed=31  failed=0  all_passed=True
  [ck] PoolingKernel: PASS  (31/31)  /home/ivan/projects/axi_demo/build/kernels/pool/kv260/pooling_test_report.json
  ```

- `failed=0` and `all_passed=True` are mandatory. **If anything else, stop here.**

## Gate 4 — Timing comparison vs most-recent run

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/compare_pool_timing.py
```

The script reads `build/kernels/pool/kv260/pooling_test_report.json` (just written by Gate 3), compares per-test `duration_ns` against `build/kernels/pool/kv260/pool_timing_last.json` (the previous run's snapshot), prints a delta table sorted by absolute movement, and overwrites `pool_timing_last.json` with the current run so the next invocation has a fresh baseline.

- First-ever run: there's no baseline yet, the script prints absolute values and saves a snapshot. Note this in your reply so the user knows the next run will produce a real diff.
- Add `--no-save` if you want a one-off comparison without overwriting the baseline (e.g. to keep a known-good reference while testing a speculative change). Use this when the user explicitly says "don't update the baseline".

## Reporting back to the user

After all four gates pass, summarise in this order:

1. **Pass/fail status** of each gate (one line each).
2. **Total wall-clock delta** vs the previous run — both ns and % (taken straight from the comparison script's TOTAL row).
3. **Top movers** — the 3-5 tests with the largest absolute deltas, listed as "Test name: prev → now (Δns, ±X%)".
4. **Synthesis-report changes** — any new timing/II violations, or data-width column changes, from Gate 2.

Keep the report tight: one short paragraph per section, no extra prose.

## What to skip

- Don't rebuild dependencies the user hasn't touched (other kernels, the inference scheduler, etc.).
- Don't re-run cmake unless a CMakeLists or `platforms/*.json` was edited — the tcl is auto-regenerated on those changes via `CMAKE_CONFIGURE_DEPENDS`, so `make` alone picks it up.
- Don't read the full `csynth.rpt` (it's >450 KB) — slice with `sed -n` or grep for what you need.

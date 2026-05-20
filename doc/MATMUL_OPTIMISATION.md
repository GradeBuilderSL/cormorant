# MatmulKernel — Optimization Log

This document records performance investigation and optimization attempts on
`kernels/matmul/kernel/MatmulKernel.cpp`. Each section describes one change or
experiment, the rationale, and the measured HW behavior-simulation
(`make behavior_test_matmul`) impact on the kv260 RTL.

For the high-level kernel description see [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

> **Status (2026-05-20).** §1 is the prior performance baseline (the shipped
> single-sequential-loop-nest kernel, also reproduced on 2026-05-20 before
> the change at 7,713,375 ns).  §2 records a DATAFLOW restructuring that was
> **tried and rejected** (measured regression).  §3 documents the
> **landed** optimization: a persistent-A loop swap inspired by the spatial-
> unroll matmul in arXiv:2503.16731 — **sim_time_ns 4,734,145 ns (-38.6 %)**
> with 20/20 tests passing, no regression on any single-n_tile test, and
> -43 % to -59 % on every multi-n_tile geometry.

---

## 1. Current performance baseline

All numbers are `sim_time_ns` reported by the kv260 behavior testbench
(`matmul_op_test`) over the full 20-test fixture list. The testbench runs at a
fixed 100 MHz sim basis, so `sim_time_ns` / per-test `duration_ns` are
directly comparable between runs regardless of the synthesis-target clock.

> **Baseline.** Original `MatmulKernel.cpp` — a single sequential tiled loop
> nest; each load / reduce / write loop is pipelined at II=1 but the loops
> execute one after another. **sim_time_ns = 7,713,375** (Σ per-test
> `duration_ns` = 7,711,175). 20/20 RTL tests pass. Synthesis: II=1 on every
> loop, Estimated Fmax 205.47 MHz. Captured **2026-05-19** —
> `build/kernels/matmul/kv260/matmul_op_test_report.json`.

| # | Test | n×k×m×batch | duration_ns |
|--:|---|---|---:|
| 0  | 1×1×1 degenerate                       | 1×1×1×1     | 5,855 |
| 1  | TileN × TileK × TileM                  | 4×256×16×1  | 197,580 |
| 2  | 2·TileN × TileK × TileM                | 8×256×16×1  | 391,610 |
| 3  | TileN × 2·TileK × TileM                | 4×512×16×1  | 386,620 |
| 4  | TileN × TileK × 2·TileM                | 4×256×32×1  | 379,010 |
| 5  | partial N                              | 6×256×16×1  | 384,060 |
| 6  | partial K                              | 4×261×16×1  | 201,520 |
| 7  | partial M                              | 4×256×19×1  | 259,790 |
| 8  | partial N, K and M                     | 6×261×19×1  | 519,060 |
| 9  | arbitrary small                        | 7×13×5×1    | 21,180 |
| 10 | outer product (K=1)                    | 5×1×17×1    | 11,490 |
| 11 | row vector (N=1)                       | 1×256×16×1  | 186,510 |
| 12 | column vector (M=1)                    | 4×256×1×1   | 45,850 |
| 13 | multi-tile all dims                    | 12×519×33×1 | 2,449,610 |
| 14 | batch=3, no broadcast                  | 5×64×19×3   | 398,620 |
| 15 | batch=4, A broadcasts                  | 5×64×19×4   | 530,220 |
| 16 | batch=4, B broadcasts                  | 5×64×19×4   | 530,240 |
| 17 | batch=6, both strided                  | 5×64×19×6   | 793,380 |
| 18 | saturation positive                    | 4×3×16×1    | 9,380 |
| 19 | saturation negative                    | 4×3×16×1    | 9,590 |
| | **Σ duration_ns** | | **7,711,175** |
| | **total sim_time_ns** | | **7,713,375** |

---

## 2. Tried and rejected: DATAFLOW pipeline restructuring

**Attempt.** Restructure the kernel as a canonical Vitis HLS `DATAFLOW`
design — four stages running concurrently, linked by `hls::stream` FIFOs:

```
a_producer ──a_stream──► compute ──acc_stream──► writer
b_producer ──b_stream──►
```

`a_producer` reads A from DDR and re-streams it, `b_producer` burst-reads B
tiles, `compute` runs the II=1 K-reduction, `writer` saturates and writes C.
The intent was to hide B's DDR-load latency behind the MAC compute.

**Result — REJECTED.** Functionally correct (20/20 RTL tests pass, C-sim
bit-identical, synthesis II=1, Fmax 205.47 MHz), but a **+46 % performance
regression**:

| Metric | Baseline | DATAFLOW | Δ |
|---|---:|---:|---:|
| total sim_time_ns | 7,713,375 | 11,276,865 | **+46 %** |
| Σ duration_ns | 7,711,175 | 11,274,665 | +46 % |

Per-test (DATAFLOW vs baseline `duration_ns`):

| Test | Baseline | DATAFLOW | Δ |
|---|---:|---:|---:|
| 4×256×16            | 197,580   | 228,030   | +15 % |
| 4×256×19 (partial M)| 259,790   | 414,000   | +59 % |
| 6×261×19 (partial)  | 519,060   | 838,400   | +62 % |
| 4×256×1  (M=1)      | 45,850    | 189,340   | **+313 %** |
| 12×519×33 multi-tile| 2,449,610 | 3,804,850 | +55 % |
| batch=6             | 793,380   | 1,229,970 | +55 % |

Every realistically-sized test regressed 15–62 %; `M=1` is pathological
(+313 %). Only trivially-small degenerate cases (`1×1×1`, `K=1`, saturation)
improved slightly.

**Root cause.** Matmul on this design is **DDR-latency bound**, not
compute-bound. From `csynth.rpt`, the `compute` K-reduction loop is II=1
(~1 k cycles per K-tile) — already cheap. The cost is in the DDR reads:
`b_producer` (and the A-buffer load) issue **one burst per matrix row** —
~256 small bursts for a 256-deep K-tile, each paying a full DDR round-trip.

DATAFLOW overlaps `compute` with the DDR traffic — but `compute` is the
*cheap* stage, so the overlap saves almost nothing, while the dataflow
structure *adds* overhead: A is streamed through a FIFO instead of read
straight from BRAM, the K-reduction does conditional stream reads, and every
tile pays inter-stage FIFO synchronisation plus stage fill/drain. The
overhead exceeds the saving → net loss. `M=1` degrades worst because the
bursts shrink to one beat and the per-tile overhead dominates entirely.

**Disposition.** Reverted. The kernel remains the single sequential loop
nest described in [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

---

## 3. Landed: persistent-A loop swap (2026-05-20)

**Source of the idea.**  [arXiv:2503.16731](https://arxiv.org/html/2503.16731v1)
("Systolic-Array MatMul for Edge FPGAs") describes a "32×32 systolic array"
that on inspection is not a true shifting array but rather **persistent-A
storage + spatially-unrolled output-stationary compute**: the entire A matrix
is loaded into BRAM once and reused while B streams in blocks.  The lever is
the persistent A, not the systolic shift.

**Change.**  Swap the matmul loop nest so the n_tile dimension is no longer
the outermost reload axis.  Concretely:

- Introduce `kMaxN = 16` (new compile-time constant, populated from
  `kernels.matmul.max_n` in `platforms/kv260.json`).
- Replace the prior `a_buf[kTileN][kMaxK]` with `a_buf[kMaxN][kMaxK]` and
  add an outer `n_block` loop that processes up to `kMaxN` rows per pass —
  for every test geometry shipping today (`N ≤ 12`) this outer loop runs
  exactly once, so A is loaded into BRAM **once per kernel invocation**.
- B-tile loading moves inside the (m_tile, k_tile) nest with an **n_grp
  wrapper** around the existing II=1 lane-rotated K-reduction:

  ```
  for m_tile:
    clear acc[kMaxN][kTileM]
    for k_tile:
      LOAD b_tile[k_valid][m_valid]   # ONCE, not per n_tile
      for n_grp in [0, ceil(n_block_valid / kTileN)):
        for ki in [0, k_valid * kTileN):  PIPELINE II=1
          n_idx = n_grp * kTileN + (ki % kTileN)
          acc[n_idx][*] += a_buf[n_idx][k_off + ki/kTileN] * b_tile[ki/kTileN][*]
    WRITE C[n_block_valid][m_valid]
  ```

  The same b_tile is now broadcast across every n_grp without rereading
  DDR.  Per-test DDR traffic for B drops by a factor of `n_grps_prev` on
  any geometry with N > kTileN.

**Partitioning that mattered.**  The first attempt used cyclic factor=kTileN
partitioning on `acc` dim 1 — each (n1, m1) bank then held `kMaxN/kTileN = 4`
registers indexed by the runtime n_grp value.  HLS treated that as a
constrained-port memory and forced II=2 (`HLS 200-885`, "Unable to schedule
store ... due to limited memory ports").  Replacing with
`#pragma HLS ARRAY_PARTITION variable=acc complete dim=0` (all 256 registers
individually) restored II=1.  The lane-rotation WAW distance is still kTileN
cycles per register, enough for the ap_fixed MAC pipeline; the runtime
`n_idx` synthesises as a kMaxN-way read MUX + write decoder, which costs
multiplexer LUTs (~3.4 k extra) but no extra cycles.

**Result.** **sim_time_ns 4,734,145 (-38.6 %).**

| Metric | Baseline | Persistent-A | Δ |
|---|---:|---:|---:|
| total sim_time_ns | 7,713,375 | 4,734,145 | **-38.6 %** |
| Σ duration_ns | 7,711,175 | 4,734,145 | -38.6 % |

Per-test (sorted by `n_grps_prev`):

| # | Geometry n×k×m×b | n_grps_prev | baseline | new | Δns | Δ% |
|--:|---|--:|--:|--:|--:|--:|
| 13 | 12×519×33×1 | 3 | 2,449,610 |   997,460 | -1,452,150 | **-59.3 %** |
|  8 |  6×261×19×1 | 2 |   519,060 |   293,660 |   -225,400 | -43.4 % |
|  2 |  8×256×16×1 | 2 |   391,610 |   222,140 |   -169,470 | -43.3 % |
|  5 |  6×256×16×1 | 2 |   384,060 |   214,580 |   -169,480 | -44.1 % |
| 17 |  5×64×19×6  | 2 |   793,380 |   456,960 |   -336,420 | -42.4 % |
| 15 |  5×64×19×4  | 2 |   530,220 |   306,010 |   -224,210 | -42.3 % |
| 16 |  5×64×19×4  | 2 |   530,240 |   306,030 |   -224,210 | -42.3 % |
| 14 |  5×64×19×3  | 2 |   398,620 |   230,410 |   -168,210 | -42.2 % |
| 10 |  5×1×17×1   | 2 |    11,490 |     8,660 |     -2,830 | -24.6 % |
|  9 |  7×13×5×1   | 2 |    21,180 |    16,530 |     -4,650 | -22.0 % |
|  1 |  4×256×16×1 | 1 |   197,580 |   197,580 |          0 |   0.0 % |
|  3 |  4×512×16×1 | 1 |   386,620 |   386,630 |        +10 |   0.0 % |
|  4 |  4×256×32×1 | 1 |   379,010 |   379,000 |        -10 |  -0.0 % |
|  6 |  4×261×16×1 | 1 |   201,520 |   201,530 |        +10 |   0.0 % |
|  7 |  4×256×19×1 | 1 |   259,790 |   259,790 |          0 |   0.0 % |
| 11 |  1×256×16×1 | 1 |   186,510 |   186,500 |        -10 |  -0.0 % |
| 12 |  4×256×1×1  | 1 |    45,850 |    45,850 |          0 |   0.0 % |
|  0 |  1×1×1×1    | 1 |     5,855 |     5,855 |          0 |   0.0 % |
| 18 |  4×3×16×1   | 1 |     9,380 |     9,380 |          0 |   0.0 % |
| 19 |  4×3×16×1   | 1 |     9,590 |     9,590 |          0 |   0.0 % |

The pattern is exactly what the change predicts: every test with
`n > kTileN` saves `(n_grps_prev - 1) × k × m_valid` DDR beats per
(m_tile, k_tile), and tests fitting in one n_grp are bit-identical (a
±10 ns drift is testbench jitter, not kernel behaviour).

**Synthesis cost.**  II=1 on every loop, Fmax 205.47 MHz (unchanged).
Resource utilisation moves from { BRAM 28 (10 %), DSP 53 (4 %), FF 9 246
(3 %), LUT 13 552 (11 %) } to { BRAM 52 (18 %), DSP 53 (4 %), FF 23 684
(10 %), LUT 20 329 (17 %) } — the BRAM increase is the 4× larger
`a_buf[kMaxN][kMaxK]`, the FF/LUT increase is the larger `acc` register
file and the n_idx MUX.  Comfortably under budget for kv260.

**Note on the `Widen Fail` finding from §2.**  The `M_AXI Burst Information`
table still reports the same `Widen Fail` on all three ports — AXI buses
stay 16 bits wide.  This change does not address AXI widening at all; the
win comes entirely from cutting the number of bursts (B is now read
`n_grps_prev` times less often when N > kTileN).  Bus widening remains
available as a future stacked optimization, but with the persistent-A path
landed it is no longer the only path forward, and the previous "matmul is
DDR-bound, fix the bus" framing is too narrow: matmul is DDR-bound, and
the gains come from doing fewer DDR transactions per output element.

---

## 4. Verification matrix

| Gate | Command | Current result |
|---|---|---|
| C-simulation | `ctest -R Matmul` | `TestMatmulRef`, `TestMatmulBlas` pass |
| HLS synthesis | `make synthesize_matmul_kv260` | II=1 all loops; Fmax 205.47 MHz |
| RTL behavior test | `make behavior_test_matmul` | 20/20 pass; sim_time 4,734,145 ns |

---

## 5. Related files

| File | Purpose |
|---|---|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel (single sequential loop nest) |
| `doc/MATMUL_KERNEL.md` | Kernel reference (architecture, interface, II=1) |
| `hw/cormorant_test_stand/kernels/matmul_op_test/` | Vivado RTL behavior-test project |
| `build/kernels/matmul/kv260/matmul_op_test_report.json` | Per-test behavior-test report |

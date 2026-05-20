# MatmulKernel — Optimization Log

This document records performance investigation and optimization attempts on
`kernels/matmul/kernel/MatmulKernel.cpp`. Each section describes one change or
experiment, the rationale, and the measured HW behavior-simulation
(`make behavior_test_matmul`) impact on the kv260 RTL.

For the high-level kernel description see [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

> **Status (2026-05-20).** §1 is the prior performance baseline (the shipped
> single-sequential-loop-nest kernel, reproduced on 2026-05-20 at
> 7,713,375 ns).  §2 records the first DATAFLOW restructuring attempt
> (4-stage, 2026-05-19) that was **tried and rejected**.  §3 documents the
> **landed** persistent-A loop swap inspired by arXiv:2503.16731 —
> -38.6 % vs §1.  §4 records a second DATAFLOW attempt (stream-based, on
> top of persistent-A) that was also **tried and rejected** (+46.5 % vs
> §3).  §5 is the current **landed** state: K-axis grouped processing
> with cache-style A reload — lifts the silent `K ≤ kChunkK` corruption
> bug, makes kBlockN / kChunkK on-chip cache sizes rather than runtime
> bounds, and drops a_buf BRAM from 18 % → 8 % at the cost of +4.1 % vs
> §3 on the chunk-path tests.  **Current sim_time: 4,926,105 ns
> (-36.1 % vs §1).**

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

- Introduce `kBlockN = 16` (new compile-time constant, populated from
  `kernels.matmul.block_n` in `platforms/kv260.json`).
- Replace the prior `a_buf[kTileN][kChunkK]` with `a_buf[kBlockN][kChunkK]` and
  add an outer `n_block` loop that processes up to `kBlockN` rows per pass —
  for every test geometry shipping today (`N ≤ 12`) this outer loop runs
  exactly once, so A is loaded into BRAM **once per kernel invocation**.
- B-tile loading moves inside the (m_tile, k_tile) nest with an **n_grp
  wrapper** around the existing II=1 lane-rotated K-reduction:

  ```
  for m_tile:
    clear acc[kBlockN][kTileM]
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
partitioning on `acc` dim 1 — each (n1, m1) bank then held `kBlockN/kTileN = 4`
registers indexed by the runtime n_grp value.  HLS treated that as a
constrained-port memory and forced II=2 (`HLS 200-885`, "Unable to schedule
store ... due to limited memory ports").  Replacing with
`#pragma HLS ARRAY_PARTITION variable=acc complete dim=0` (all 256 registers
individually) restored II=1.  The lane-rotation WAW distance is still kTileN
cycles per register, enough for the ap_fixed MAC pipeline; the runtime
`n_idx` synthesises as a kBlockN-way read MUX + write decoder, which costs
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
`a_buf[kBlockN][kChunkK]`, the FF/LUT increase is the larger `acc` register
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

## 4. Tried and rejected: DATAFLOW on top of persistent-A (2026-05-20)

After §3 landed, the compute-vs-memory ratio per (m_tile, k_tile) shifted in a
way that suggested overlap might finally be worthwhile.  On a full tile
(k_valid = kTileK, m_valid = kTileM):

|  | sequential per (m_tile, k_tile) |
|---|---:|
| B-load (DDR)         | k_valid · m_valid                       =  4096 |
| K-reduce, n_grps = 1 | n_grps · k_valid · kTileN               =  1024 |
| K-reduce, n_grps = 3 | n_grps · k_valid · kTileN               =  3072 |

DATAFLOW could in principle drive that to `max(producer, consumer)` per
k_tile, hiding the entire K-reduce behind the B-load.

**Attempt.**  Restructure the per-m_tile accumulate phase as a canonical
DATAFLOW region with two processes connected by a stream:

```
accumulate_m_tile_dataflow:
  #pragma HLS DATAFLOW
  hls::stream<BPack> b_stream                # one B row (kTileM elements) per pack
  load_b_tiles_all  (b_ptr,  b_stream, …)    # PRODUCER (DDR → stream)
  kreduce_all_ktiles(a_buf,  b_stream, acc, …)  # CONSUMER (stream → MAC)
```

`BPack` carries a full B row, so consumer-side stream traffic stays at one
pop per row — matching the producer's row-granular push rate.  The consumer
drains each k_tile's packs into a local `b_tile_local` (1 pack/cycle, II=1)
then runs the n_grp lane-rotated K-reduce.  Inside the DATAFLOW region the
producer's k_tile T+1 fill is meant to overlap the consumer's k_tile T
reduce.

A shared-array variant (b_tile_buf passed between processes, relying on
auto-ping-pong) was tried first and rejected purely on **C-simulation
correctness**: csim runs DATAFLOW stages sequentially, so the producer
overwrote the buffer with the last k_tile's data before the consumer ever
read k_tile 0; multi-k_tile geometries in `TestMatmulRef` returned wrong
results in 64 / 64 elements.  The stream-based version is correct in both
csim and HLS sim.

**Result — REJECTED.** Functionally correct (20/20 RTL tests pass, II=1 on
every loop, Fmax unchanged at 205.47 MHz), but a **+46.5 % regression vs
the §3 persistent-A baseline**:

| Metric | §3 persistent-A | §4 +DATAFLOW | Δ |
|---|---:|---:|---:|
| total sim_time_ns | 4,734,145 | 6,937,535 | **+46.5 %** |
| Σ duration_ns     | 4,734,145 | 6,937,535 | +46.5 % |

Every single test regressed — including ones where DATAFLOW could not
possibly have helped (k_tiles = 1 geometries with nothing to pipeline):

| Test | geom | §3 pers-A | §4 +DATAFLOW | Δ |
|--:|---|--:|--:|--:|
| 12 |  4×256×1×1   |    45,850 |   208,640 | **+355 %** |
|  7 |  4×256×19×1  |   259,790 |   447,100 |  +72 % |
|  8 |  6×261×19×1  |   293,660 |   476,230 |  +62 % |
| 17 |  5×64×19×6   |   456,960 |   735,730 |  +61 % |
| 15 |  5×64×19×4   |   306,010 |   491,860 |  +61 % |
| 16 |  5×64×19×4   |   306,030 |   491,850 |  +61 % |
| 14 |  5×64×19×3   |   230,410 |   369,760 |  +60 % |
| 13 | 12×519×33×1  |   997,460 | 1,424,900 |  +43 % |
| 11 |  1×256×16×1  |   186,500 |   237,640 |  +27 % |
|  4 |  4×256×32×1  |   379,000 |   481,270 |  +27 % |
|  1 |  4×256×16×1  |   197,580 |   248,680 |  +26 % |
|  3 |  4×512×16×1  |   386,630 |   478,590 |  +24 % |
|  6 |  4×261×16×1  |   201,530 |   249,270 |  +24 % |
|  5 |  6×256×16×1  |   214,580 |   265,750 |  +24 % |
|  2 |  8×256×16×1  |   222,140 |   273,300 |  +23 % |
|  9 |  7×13×5×1    |    16,530 |    22,060 |  +34 % |
|  0 |  1×1×1×1     |     5,855 |     5,955 |  +2 % |
| 10 |  5×1×17×1    |     8,660 |     8,930  |  +3 % |
| 18 |  4×3×16×1    |     9,380 |     9,880 |  +5 % |
| 19 |  4×3×16×1    |     9,590 |    10,140 |  +6 % |

**Root cause.** Three additive overheads, each individually small but
collectively > the overlap saving:

1. **HLS 200-1449 — a_buf cross-process read.** Synthesis emitted:
   `Process kreduce_all_ktiles has both a predecessor and reads an input
   from its caller (…).  This may lead to lower throughput.  Consider
   copying this input via a predecessor process.`  HLS could not fully
   pipeline the consumer because a_buf is read directly from the
   MatmulKernel-scope BRAM rather than streamed through a producer.
   Copying it into the dataflow region (which would be its own process)
   adds BRAM and per-block setup latency that is itself substantial for
   the small kv260 geometries.

2. **DATAFLOW process fill / drain per m_tile.**  The two-process
   pipeline pays its setup latency on every call to
   `accumulate_m_tile_dataflow`, i.e. once per m_tile.  For tests with
   exactly one k_tile per m_tile (the common case), there is no
   cross-iteration overlap to amortise it against and the setup latency
   is pure overhead.  This is the same failure mode as the 2026-05-19
   four-stage DATAFLOW (§2) — confirmed by the same `4×256×1×1`
   pathology (+355 %, identical sign to §2's +313 %).

3. **Consumer fill phase is not free.**  Even with the BPack-wide stream
   (one pop per row), the consumer still spends k_valid cycles draining
   into `b_tile_local` before it can K-reduce — so the consumer's wall
   time per k_tile is `k_valid + n_grps · k_valid · kTileN`, not the
   `n_grps · k_valid · kTileN` the back-of-envelope calculation assumed.
   On n_grps = 1 geometries that fill *equals* the K-reduce time, so the
   consumer total roughly doubles vs the sequential code's "K-reduce
   only" stage, and the producer alone is not slow enough to fully hide
   the doubled consumer.

**Synthesis cost** (also a net loss): BRAM 52 → 67 (+15), FF 23 684 →
48 948 (more than doubled), LUT 20 329 → 46 172 (more than doubled) —
DATAFLOW process control logic, FIFO depths, and the per-process
register replication for stable inputs.

**Disposition.**  Reverted at HEAD.  The kernel ships the §3 persistent-A
single-sequential-nest version; **§4 stays a tried-and-rejected note** so
future iterations don't repeat the same pattern.

Where a successful DATAFLOW could still come from, if it is pursued
later: it would need to (a) move a_buf into a producer process so the
consumer is "fed" by both A and B streams (eliminating the
HLS 200-1449 warning's serialisation), (b) collapse the consumer's
fill-then-reduce into a single fused pipeline that reads B directly out
of the stream during the K-reduce (rather than the current two-phase
fill + reduce), and (c) keep the K-reduce process small enough that the
fill / drain latency per (m_tile, k_tile) is dominated by the actual
overlap saving.  None of those changes are scheduled.

---

## 5. Landed: K-axis grouped processing with cache-style reload (2026-05-20)

**Motivation.**  §3 (persistent-A) introduced a hard runtime limit `K ≤ kChunkK`.
The kernel did *not* check this at the AXI-Lite registers; instead the
inner load loop wrote past `a_buf[*][kChunkK]` for any K > kChunkK, silently
corrupting on-chip memory.  The inference scheduler had no validator
for this either, so a model with a large MatMul would have produced
incorrect output with no diagnostic.  kBlockN had the same shape but was
already handled correctly via the outer `n_block` loop (each block
reloads A — duplicated reads in exchange for unbounded N).

**Change.**  Apply the same cache-style pattern to the K axis: enumerate
chunks of size kChunkK in an outer `k_chunk` loop, and reload `a_buf` on a
cache miss.  Both axes are now uniformly grouped-tile:

```
n_block loop   (cache miss boundary on N)
  m_tile loop
    clear acc
    k_chunk loop                       (cache miss boundary on K)
      if k_chunk != last_loaded_k_chunk:
        LOAD a_buf [n_block_valid × k_chunk_valid]    # cache miss
        last_loaded_k_chunk = k_chunk
      k_tile loop inside chunk
        LOAD b_tile
        n_grp K-reduce → acc           (a_buf indexed by chunk-local k offset)
    WRITE C                            (acc has accumulated across all chunks)
```

A `last_loaded_k_chunk` counter (reset at each `n_block` boundary)
makes the cache check sound:

- **K ≤ kChunkK ⇒ k_chunks = 1.** The first m_tile loads; every later
  m_tile sees `last_loaded_k_chunk == 0` and skips the reload.  A is
  loaded *exactly once per n_block*, identical traffic to §3.
- **K > kChunkK ⇒ k_chunks > 1.**  Each m_tile must reload every chunk
  in turn — `m_tiles × k_chunks` DDR loads of A.  This is the
  "duplicated readings on cache miss" cost accepted in exchange for
  arbitrary K.

`kBlockN` and `kChunkK` are now *cache sizes* rather than runtime workload
bounds.  The scheduler can pass any (N, K, M); the kernel handles them
correctly, paying extra DDR traffic when the working set exceeds the
on-chip cache.

**Platform-JSON change.**  `kernels.matmul.chunk_k` reduced from `2048`
to `256` in `platforms/kv260.json`.  Two reasons: (a) the previous
2048 was sized to fit every shipping test's K in a single chunk so
the chunk path had no on-hardware coverage; reducing it to 256 forces
tests with K ∈ {261, 512, 519} onto the multi-chunk path so the kv260
behaviour testbench exercises the new code; (b) the smaller a_buf
([16][256] vs [16][2048]) saves BRAM that is otherwise unused under
the current workloads.

**Result.** **sim_time_ns 4,926,105 (-36.1 % vs original baseline,
+4.1 % vs §3 persistent-A)** — 20/20 RTL tests pass, II=1 on every
loop, Fmax 205.47 MHz unchanged.

| Metric | original | §3 persistent-A | §5 +chunk | Δ §5 vs §3 |
|---|---:|---:|---:|---:|
| total sim_time_ns | 7,713,375 | 4,734,145 | 4,926,105 | +4.05 % |

Per-test (sorted by k_chunks × m_tiles, the chunk-path multiplier):

| # | geom | k_chunks×m_tiles | pers-A | +chunk | Δns | Δ% |
|--:|---|--:|--:|--:|--:|--:|
| 13 | 12×519×33×1 | 3 × 3 | 997,460 | 1,162,450 | +164,990 | **+16.5 %** |
|  8 |  6×261×19×1 | 2 × 2 | 293,660 |   315,920 |  +22,260 |   +7.6 % |
|  3 |  4×512×16×1 | 2 × 1 | 386,630 |   388,850 |   +2,220 |   +0.6 % |
|  6 |  4×261×16×1 | 2 × 1 | 201,530 |   202,680 |   +1,150 |   +0.6 % |
| — others (15 tests, k_chunks = 1) | | | | within ±0.2 % testbench jitter |

The cost on tests 3, 6, 8, 13 scales exactly as predicted by the cache
model — `(k_chunks × m_tiles - 1)` extra A reloads of (n_block_valid ×
kChunkK) elements each.  All k_chunks = 1 tests are unaffected.

**Synthesis cost** (net win on resources): the 8× smaller a_buf drops
BRAM from { 52 (18 %) } to { 24 (8 %) }; DSP unchanged at 53 (4 %);
FF and LUT roughly unchanged (24 k / 21 k → 30 k / 22 k — small bump
from the extra outer loop's control logic).  More headroom for future
upper-bound increases on either axis.

**Why this isn't an "optimisation" in the sim_time sense.**  §5 is a
*correctness + flexibility* change that gives back 4 % vs §3.  The win
is that the kernel now handles arbitrary K (previously a silent
corruption) and BRAM is freed up for other uses; the sim_time hit on
four tests is the cost of validating the chunk reload path on the kv260
behaviour testbench rather than only in C-sim.  Net vs original
baseline is still **-36.1 %**.

---

## 6. Verification matrix

| Gate | Command | Current result |
|---|---|---|
| C-simulation | `ctest -R Matmul` | `TestMatmulRef`, `TestMatmulBlas` pass |
| HLS synthesis | `make synthesize_matmul_kv260` | II=1 all loops; Fmax 205.47 MHz |
| RTL behavior test | `make behavior_test_matmul` | 20/20 pass; sim_time 4,926,105 ns |

---

## 7. Related files

| File | Purpose |
|---|---|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel (single sequential loop nest) |
| `doc/MATMUL_KERNEL.md` | Kernel reference (architecture, interface, II=1) |
| `hw/cormorant_test_stand/kernels/matmul_op_test/` | Vivado RTL behavior-test project |
| `build/kernels/matmul/kv260/matmul_op_test_report.json` | Per-test behavior-test report |

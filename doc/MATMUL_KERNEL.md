# Matmul Kernel — Detailed Implementation Description

## Overview

`MatmulKernel` is a Vitis HLS kernel implementing tiled matrix multiplication
`C = A × B` on the Xilinx KV260 FPGA. It is one of four hardware kernels in
the `axi_demo` project. The kernel computes a batch of independent 2-D
matrix products on row-major matrices, with optional per-operand batch
broadcasting (a stride of 0 reuses `A` or `B` across every batch step).

A five-level tiling strategy (N-block `kBlockN` × output rows `kTileN`
lane-rotation × output columns `kTileM` × K-chunk `kChunkK` × inner-K-tile
`kTileK`) keeps the working set on-chip and feeds an II=1 K-reduction
loop.  Both the N-block and K-chunk levels treat `a_buf` as a **cache**
for A: when the next outer iteration needs a different working set, A is
reloaded from DDR (duplicated reads in exchange for unbounded N and K).
Within an n_block, each loaded B-tile is **shared across all kTileN-grouped
rows** of A — eliminating the per-n_tile B-tile reloads the original code
paid.  Row-lane rotation (`n1 = ki % kTileN`) breaks the accumulator
read-after-write hazard so the inner loop sustains one K-step per clock
across `kTileM` parallel MAC lanes.  See
[MATMUL_OPTIMISATION.md §3 / §5](MATMUL_OPTIMISATION.md) for the landed
-38.6 % (persistent-A) and -36.1 % (after K-chunk reload) impact figures.

---

## 1. AXI Interface

**Memory ports (m_axi):**

| Bundle | Port | Direction | Description |
|--------|------|-----------|-------------|
| `gmem0` | `a` | Read | Matrix A `[n][k]`, row-major |
| `gmem1` | `b` | Read | Matrix B `[k][m]`, row-major |
| `gmem2` | `c` | Write | Matrix C `[n][m]`, row-major |

Keeping A, B, and C on separate AXI buses lets HLS issue their reads and
writes concurrently.

**AXI-Lite control registers (`s_axilite bundle=ctrl`):**

| Register | Type | Description |
|----------|------|-------------|
| `a`, `b`, `c` | `uint64_t` | Physical DDR base addresses |
| `n` | `unsigned` | Rows of A / rows of C |
| `k` | `unsigned` | Inner dimension (cols of A = rows of B); no upper bound (K > kChunkK paginates over outer k_chunk iterations) |
| `m` | `unsigned` | Cols of B / cols of C |
| `batch` | `unsigned` | Number of independent 2-D products |
| `a_batch_stride` | `unsigned` | Elements to advance `a` per batch step (0 = broadcast) |
| `b_batch_stride` | `unsigned` | Elements to advance `b` per batch step (0 = broadcast) |
| `c_batch_stride` | `unsigned` | Elements to advance `c` per batch step |
| `return` | — | `ap_ctrl_hs` (start / done / idle / ready) |

Memory layout is row-major: `A[row·k + col]`, `B[row·m + col]`,
`C[row·m + col]`.

---

## 2. Compile-Time Configuration (`Config.h.in`)

CMake substitutes the data types and tile constants into `Config.h`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `Data_t` | `ap_fixed<16,8>` | Element type (2-byte, range \[-128, 127.996\]) |
| `AccData_t` | `ap_fixed<32,16>` | Accumulator type (wider range, avoids overflow) |
| `kTileN` | 4 | Lane-rotation interleave depth (rows processed per inner ki sweep). Power of 2; must be ≥ MAC latency (≈3) for II=1 |
| `kTileM` | 16 | Output columns processed per cycle — one DSP accumulator lane each. Power of 2 |
| `kTileK` | 256 | On-chip B-buffer K-slice depth. Power of 2 (so `k_tile` indexing needs no divider) |
| `kChunkK` | 256 | On-chip A-cache depth (columns of A held per chunk). `K > kChunkK` is split into ceil(K/kChunkK) outer k_chunk iterations; each iteration reloads `a_buf` from DDR.  Not a workload bound — arbitrary K is supported. |
| `kBlockN` | 16 | On-chip A-cache height (rows held per outer pass). `N > kBlockN` is split into ceil(N/kBlockN) outer n_block iterations, each reloading A; tests with `N ≤ kBlockN` load A once.  Not a workload bound — arbitrary N is supported. |

If Vitis HLS headers are unavailable at configure time, the types fall back
to `float` / `double`.

**`AccData_t` overflow budget.** With `ap_fixed<16,8>` operands the
worst-case product is `127.996² ≈ 16383`; `ap_fixed<32,16>` saturates near
`32767`, so accumulation is exact for `K ≤ 2` at full-scale inputs and for
`K ≤ ~200` at the typical neural-net range `|v| ≤ 8`. Larger `K` budgets
need a wider accumulator (`-DMM_ACC_DATA_TYPE=ap_fixed<40,24>`).

**No scheduler-side runtime constraint on dimensions.**  Every dimension
(`n`, `k`, `m`, `batch`) is handled by an outer loop in the kernel: large
N paginates over n_blocks, large K paginates over k_chunks, large M
paginates over m_tiles, batch is just a pointer-offset loop.  `kBlockN`
and `kChunkK` are the *cache sizes* that determine how often DDR reloads
of A happen, not workload upper bounds.

---

## 3. On-Chip Memory

Three `static` on-chip buffers stage the tiles between DDR and the MAC array
(declared `static` so HLS infers BRAM/registers; in C-sim the storage
persists across calls, which is safe because every element is written before
it is read within a call):

| Buffer | Shape | Storage | Holds |
|--------|-------|---------|-------|
| `a_buf` | `[kBlockN][kChunkK]` | cyclic factor=`kTileN` on dim 1 → `kTileN` BRAMs each `(kBlockN/kTileN)·kChunkK` deep | Up to `kBlockN` full rows of A for the current `n_block`; loaded once, reused across all `m_tile`s, `k_tile`s, and `n_grp`s |
| `b_tile` | `[kTileK][kTileM]` | `kTileM` BRAMs (partition complete dim 2) | One `kTileK × kTileM` block of B; reloaded from DDR per `(m_tile, k_tile)` — shared across every `n_grp` in the block (no per-`n_tile` reload, unlike the prior version) |
| `acc` | `[kBlockN][kTileM]` | registers (partition complete dim 0) | `kBlockN × kTileM` partial dot products; cleared per `m_tile`; runtime `n_idx` index synthesises as a kBlockN-way MUX + decoder |

```cpp
static Data_t    a_buf [kBlockN ][kChunkK];
static Data_t    b_tile[kTileK][kTileM];
static AccData_t acc   [kBlockN ][kTileM];
#pragma HLS ARRAY_PARTITION variable=a_buf  cyclic factor=kTileN dim=1
#pragma HLS ARRAY_PARTITION variable=b_tile complete             dim=2
#pragma HLS ARRAY_PARTITION variable=acc    complete             dim=0
```

`a_buf` is cyclically partitioned on dim 1 with factor `kTileN`: the bank
index is `n_idx % kTileN` (the rotating `n1` from the K-loop — compile-time
within each pipeline iteration), and the within-bank address is
`(n_idx / kTileN, k_off + kk) = (n_grp, k_off + kk)` (constant per ki sweep,
varying per n_grp).  Each bank is therefore a single-port BRAM that issues
one read per cycle.

`b_tile` is partitioned on dim 2 so the `kTileM`-wide unrolled inner loop
reads one element per column bank per cycle.

`acc` is fully partitioned on dim 0 — every `(n, m)` is an individual
register.  An earlier cyclic-on-dim-1 layout (matching `a_buf`) forced
II=2 because HLS treated the depth-(kBlockN/kTileN) register-array banks as
constrained-port memories; full register partitioning sidesteps that.  The
lane-rotation WAW distance is still `kTileN` cycles per register, enough
for the ap_fixed MAC pipeline.

The kernel is **not** a `DATAFLOW` design — it is a sequential loop nest
with each load / reduce / write loop pipelined at II=1.  A DATAFLOW variant
was tried and rejected in 2026-05 (see [MATMUL_OPTIMISATION.md §2](MATMUL_OPTIMISATION.md));
the speedup that did land instead is the persistent-A loop swap in §3 of
that document, which keeps the same sequential structure but eliminates
per-n_tile B-tile DDR reloads.

---

## 4. Loop Structure and HLS Pragmas

```
for bi in [0, batch)                              // a/b/c advanced by *_batch_stride
  for n_block in [0, ceil(n / kBlockN))             // cache-miss boundary on N
    n_block_off   = n_block · kBlockN
    n_block_valid = min(kBlockN, n - n_block_off)
    n_grps        = ceil(n_block_valid / kTileN)
    last_loaded_k_chunk = -1                      // a_buf is empty at block start

    for m_tile in [0, ceil(m / kTileM))
      m_off, m_valid = m_tile·kTileM, min(kTileM, m - m_off)

      // CLEAR acc — kBlockN × kTileM registers, fully unrolled → 1 cycle
      for n1, m1 (UNROLL): acc[n1][m1] = 0

      for k_chunk in [0, ceil(k / kChunkK))         // cache-miss boundary on K
        k_chunk_off, k_chunk_valid = k_chunk·kChunkK, min(kChunkK, k - k_chunk_off)

        if k_chunk ≠ last_loaded_k_chunk:         // cache check
          // LOAD a_buf — n_block_valid rows × k_chunk_valid cols       PIPELINE II=1
          for n1 in [0, n_block_valid): for ki in [0, k_chunk_valid):
            a_buf[n1][ki] = a[(n_block_off + n1)·k + (k_chunk_off + ki)]
          last_loaded_k_chunk = k_chunk

        for k_tile in [0, ceil(k_chunk_valid / kTileK))
          k_tile_off   = k_tile · kTileK           // chunk-local
          k_tile_valid = min(kTileK, k_chunk_valid - k_tile_off)
          k_off_global = k_chunk_off + k_tile_off  // for B addressing

          // LOAD b_tile — k_tile_valid rows × m_valid cols, global K   PIPELINE II=1
          for k1 in [0, k_tile_valid): for m1 in [0, m_valid):
            b_tile[k1][m1] = b[(k_off_global + k1)·m + (m_off + m1)]

          // K-REDUCTION — n_grp wrapper, a_buf indexed chunk-locally   PIPELINE II=1
          for n_grp in [0, n_grps):
            for ki in [0, k_tile_valid·kTileN):
              n1     = ki % kTileN
              kk     = ki / kTileN                 // tile-local
              n_idx  = n_grp·kTileN + n1
              a_val  = a_buf[n_idx][k_tile_off + kk]   // chunk-local offset
              for m1 in [0, kTileM) UNROLL:
                acc[n_idx][m1] += AccData_t(a_val) · AccData_t(b_tile[kk][m1])

      // WRITE C — saturate_cast acc → C; acc has folded in every k_chunk
      for n1 in [0, n_block_valid): for m1 in [0, m_valid):
        c[(n_block_off + n1)·m + (m_off + m1)] = saturate_cast<Data_t>(acc[n1][m1])
```

**Cache hit / miss summary.**
- `K ≤ kChunkK` (k_chunks = 1): m_tile 0 misses and loads; every later m_tile
  hits — A is loaded *once per n_block*, identical to the pre-§5 traffic.
- `K > kChunkK` (k_chunks > 1): each m_tile reloads every chunk in turn —
  `m_tiles × k_chunks` DDR loads of A per n_block.  This is the cache-miss
  overhead, accepted in exchange for handling arbitrary K.
- Cache is invalidated at every `n_block` boundary (resident rows belong
  to the previous block).

`B` is reloaded per `(m_tile, k_tile)` and shared across every `n_grp` in
the n_block.  Partial last tiles load only the valid rows / columns —
unused `a_buf` / `b_tile` lanes hold stale data and the trailing `acc`
lanes (`n_idx ≥ n_block_valid`) are written by spurious MACs but never
emitted to C.

### HLS pragmas applied

| Pragma | Location | Effect |
|--------|----------|--------|
| `INTERFACE m_axi … bundle=gmem0/1/2` | top-level | AXI memory ports for A / B / C |
| `INTERFACE s_axilite … bundle=ctrl` | every scalar + `return` | AXI-Lite register file |
| `ARRAY_PARTITION variable=a_buf cyclic factor=kTileN dim=1` | `a_buf[kBlockN][kChunkK]` | `kTileN` parallel row banks; n_grp selects within-bank position |
| `ARRAY_PARTITION variable=b_tile complete dim=2` | `b_tile[kTileK][kTileM]` | `kTileM` parallel column banks |
| `ARRAY_PARTITION variable=acc complete dim=0` | `acc[kBlockN][kTileM]` | all `kBlockN·kTileM` accumulators in registers (kBlockN-way runtime MUX) |
| `PIPELINE II=1` | a_buf load / b_tile load / K-reduction / C write | One iteration per clock |
| `UNROLL` | inner `m1` loop + the `acc` clear | `kTileM` parallel MAC lanes |

---

## 5. II=1 Strategy — Accumulator Lane Rotation

The K-reduction loop iterates `k_valid × kTileN` times. Each group of
`kTileN` consecutive iterations processes **one** K element across all
`kTileN` row lanes:

```
n1 = ki % kTileN     // which row lane  — rotates 0, 1, …, kTileN-1
kk = ki / kTileN     // K index local to this K-tile
```

Because the lane index rotates, the same `acc[n1][m1]` register is written
only once every `kTileN` cycles. That distance (`kTileN ≥ MAC latency ≈ 3`)
covers the multiply-accumulate pipeline depth, so the read-after-write hazard
that would otherwise force II ≥ 3 is broken and HLS schedules the loop at
II=1. Since `kTileN` is a power of two, `ki % kTileN` is a bitwise AND and
`ki / kTileN` a right shift — no dividers appear in RTL.

The inner `m1` loop is fully unrolled, so `kTileM` MAC units fire every
cycle (one per output column). **Inner-loop throughput is `kTileM`
MACs/cycle**, sustained at II=1.

---

## 6. Batch Broadcasting

`batch` independent 2-D products are computed by the outer `bi` loop, which
advances each pointer by its `*_batch_stride`:

- `a_batch_stride = 0` → `A` stays fixed and broadcasts across every batch
  step (e.g. a shared weight matrix against a batch of activations).
- `b_batch_stride = 0` → `B` broadcasts.
- `c_batch_stride` is normally `n·m` (each product writes a fresh output).

This matches ONNX `MatMul` broadcasting on the leading batch dimension
without copying the broadcast operand in DDR.

---

## 7. Data Types and Saturation

`saturate_cast<Data_t>(v)` (defined in `MatmulKernel.h`) converts an
`AccData_t` accumulator back to `Data_t`. For `ap_fixed` it routes through
`ap_fixed<W,I,AP_TRN,AP_SAT>` — truncation toward zero, then saturation
clamping — matching ONNX fixed-point semantics; it is applied in the C-write
loop. The primary template is an identity pass-through for `float` / `double`
builds. The `ap_fixed` specialisation is guarded by `MATMUL_HAVE_APFIXED` so
the matmul subdirectory stays self-contained (it does not depend on the
VectorOPKernel headers).

MAC operands are widened to `AccData_t` before the multiply
(`AccData_t(a_val) · AccData_t(b_tile[kk][m1])`).

---

## 8. Test Coverage (`TestMatmulSim.cpp`)

C-simulation tests compiled with GCC. Each case runs `MatmulKernel` against
`ref_matmul_2d()` — a naive triple-nested-loop oracle that uses the same
`AccData_t` accumulation and `saturate_cast<Data_t>` output, so fixed-point
results are bitwise-identical (exact comparison); `float` builds allow a
1-ULP tolerance for reordered tile sums.

| Category | Cases |
|----------|-------|
| Degenerate | `1×1×1`; unit-N (`1×K×M`); unit-M (`N×K×1`) |
| Exact tiles | `kTileN × kTileK × kTileM` — one full tile in every dimension |
| Partial last tile | partial N (`kTileN+2`); partial M (`kTileM+3`); partial K (`kTileK+5`, spans 2 K-tiles) |
| Multi-tile | all dims span 2 tiles; `kTileN·… × kTileK·2+7 × kTileM·2+1` |
| Arbitrary | `7×13×5` (all dims below the tile sizes) |
| Batch | `batch=3` without broadcast |

A second test, **`TestMatmulBlas.cpp`**, validates the `float` build of the
kernel against `cblas_sgemm` when a BLAS library is found at configure time.
`make gen_matmul_test_data` re-runs the reference in `--dump-data` mode to
emit hex fixtures for the HDL testbench.

---

## 9. Inference Scheduler Integration

**`MatmulNode` (`nodes.py`, `kernel_name = "MatmulKernel"`)** maps the ONNX
`MatMul` operator to `XMatmulkernel` invocations:

- Validates the 2-D / batched shapes and the `n`, `k`, `m` dimensions.
- No K-dimension upper bound: arbitrary K is handled by the kernel's outer
  k_chunk loop with cache-style A reload (see `MATMUL_OPTIMISATION.md` §5).
- Supports batched matmul and stride-0 batch broadcasting; a row-strided
  decomposition handles alignment-gapped buffers.

`Gemm` is **not** a matmul node directly — `OnnxGraph._preprocess_model()`
decomposes `Gemm` into `MatMul` + optional `Add` at model-load time, so the
MatmulKernel only ever sees plain `MatMul`.

The code-generated `run_matmul()` writes the AXI-Lite registers and calls
`XMatmulkernel_Start()` non-blocking; `run_matmul_at()` (used inside the
4-D × 3-D outer loop, where iterations would otherwise race on the shared
Matmul registers) is the synchronous variant that polls internally. A
`kernel_wait(KERNEL_MATMUL)` drains the lane only when a dependent op needs
the result.

---

## 10. Build Targets

```bash
# C simulation (GCC, no Vitis)
make TestMatmulRef && ctest

# HLS synthesis + IP export for KV260
make synthesize_matmul_kv260
```

The synthesis target reads a `platforms/<name>.json` (part, optional board
and clock) and invokes Vitis HLS via `Synthesis.tcl.in`, which configures the
project, sets 64-bit AXI and the bus width, runs `csynth_design`, and exports
an IP-catalog archive.

---

## 11. Key Source Files

| File | Purpose |
|------|---------|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel — tiled loop nest |
| `kernels/matmul/include/MatmulKernel.h` | Kernel declaration, `saturate_cast<T>` |
| `kernels/matmul/include/Config.h.in` | CMake template → `Config.h` (`Data_t`, `AccData_t`, tile constants) |
| `kernels/matmul/test/TestMatmulSim.cpp` | C simulation tests (GCC) |
| `kernels/matmul/test/TestMatmulBlas.cpp` | `float` build validated against `cblas_sgemm` |
| `kernels/matmul/scripts/Synthesis.tcl.in` | Vitis HLS TCL template |
| `inference-scheduler/src/nodes.py` | `MatmulNode` class (ONNX → kernel params) |
| `inference-scheduler/src/codegen/_source.py` | `run_matmul()` / `run_matmul_at()` code generation |
| `inference-scheduler/src/graph.py` | `Gemm` → `MatMul` + `Add` decomposition |

---

## 12. Summary

| Aspect | Details |
|--------|---------|
| **Supported ONNX op** | `MatMul` (`Gemm` decomposed to `MatMul` + `Add` at load time) |
| **Operation** | `C = A × B`, batched, row-major |
| **Data type** | `ap_fixed<16,8>` (default) or `float` |
| **Accumulator type** | `ap_fixed<32,16>` (default) or `double` |
| **Tiling** | `kBlockN=16` n_block × `kTileN=4` lane-rotation × `kTileM=16` columns × `kChunkK=256` k_chunk × `kTileK=256` inner |
| **Inner-loop parallelism** | `kTileM=16` MACs/cycle (unrolled `m1` lanes) |
| **Initiation interval** | II=1 in every load / reduce / write loop |
| **II=1 mechanism** | Accumulator lane rotation `n1 = ki % kTileN` within each n_grp (RAW distance = `kTileN`) |
| **Architecture** | Sequential tiled loop nest (not `DATAFLOW`); persistent-A loop swap + grouped K with cache-style reload on miss |
| **On-chip buffers** | `a_buf` (BRAM, `kBlockN × kChunkK` cache), `b_tile` (BRAM), `acc` (registers, kBlockN×kTileM) |
| **A reuse** | `a_buf` loaded once per `(n_block, k_chunk)` combination; same chunk used by every m_tile in the n_block via `last_loaded_k_chunk` cache check (so `K ≤ kChunkK` → A loaded once per n_block, identical to pre-§5 traffic) |
| **B reuse** | `b_tile` loaded once per `(m_tile, k_tile)`; broadcast across every `n_grp` in the n_block |
| **Batch broadcasting** | `a_batch_stride` / `b_batch_stride` = 0 reuses A / B |
| **Workload bounds** | None — arbitrary `n`, `k`, `m`, `batch`. Large `N`/`K` paginate over `n_block` / `k_chunk` with DDR reloads on cache miss. `kBlockN`/`kChunkK` set the cache size that decides how often misses happen. |
| **AXI master ports** | 3 (gmem0 `a`, gmem1 `b`, gmem2 `c`) |
| **AXI-Lite registers** | 10 scalars/pointers + `return` |
| **Saturation** | `saturate_cast` with `AP_TRN` + `AP_SAT` at the C-write |
| **AXI-Lite base address** | `0xA001_0000` |
| **Driver prefix** | `xmatmulkernel` |
| **UIO device name** | `MatmulKernel_0` |

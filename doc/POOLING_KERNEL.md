# Pooling Kernel — Detailed Implementation Description

## Overview

`PoolingKernel` is a Vitis HLS kernel implementing ONNX-compliant 2-D spatial pooling on NCHW tensors. It is one of four hardware kernels in the `axi_demo` project, targeting the Xilinx KV260 FPGA. The kernel supports three pooling families (Max, Average, Lp) including their global variants, dilation, padding, and a configurable channel-tiling strategy for II=1 throughput.

---

## 1. AXI Interface

**Memory ports (m_axi):**

| Bundle | Port | Direction | Description |
|--------|------|-----------|-------------|
| `gmem0` | `x` | Read | Input feature map (NCHW) |
| `gmem1` | `y` | Write | Output feature map (NCHW) |

**AXI-Lite control registers (`s_axilite bundle=ctrl`) — 21 registers total:**

| Register | Type | Description |
|----------|------|-------------|
| `x`, `y` | `uint64_t` | Physical DDR base addresses |
| `batch`, `channels` | `unsigned` | Tensor outer dimensions |
| `in_h`, `in_w` | `unsigned` | Input spatial size |
| `out_h`, `out_w` | `unsigned` | Output spatial size |
| `pool_h`, `pool_w` | `unsigned` | Pool window size |
| `stride_h`, `stride_w` | `unsigned` | Stride |
| `pad_top`, `pad_left` | `unsigned` | Padding |
| `dil_h`, `dil_w` | `unsigned` | Dilation |
| `pool_type` | `unsigned` | 0=MaxPool, 1=AveragePool, 2=LpPool |
| `lp_order` | `unsigned` | 1 or 2 (only for LpPool) |
| `count_include_pad` | `unsigned` | 0 or 1 (only for AveragePool) |

64-bit AXI addressing is configured in the synthesis TCL (`config_interface -m_axi_addr64`).

---

## 2. Supported Operations

| `pool_type` | Name | Pad fill | Accumulation | Finalization |
|-------------|------|----------|--------------|--------------|
| 0 | MaxPool | `kAccMin` (identity) | `acc = max(acc, x[i])` | saturate_cast to `Data_t` |
| 1 | AveragePool | 0 | `acc += x[i]` | `acc × inv_denom`, then cast; denominator = `pool_h×pool_w` or `valid_count` |
| 2 | LpPool (p=1) | 0 | `acc += |x[i]|` | cast |
| 2 | LpPool (p=2) | 0 | `acc += x[i]²` | `sqrtf(acc)`, then cast |

Global variants (`GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool`) are handled by the caller passing `pool_h=in_h`, `pool_w=in_w`, `stride=1`, `pad=0` — the kernel sees no special case.

---

## 3. Compile-Time Configuration (`Config.h.in`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `Data_t` | `ap_fixed<16,8>` | Element type (2-byte, range \[-128, 127.996\]) |
| `AccData_t` | `ap_fixed<32,16>` | Accumulator type (wider range, avoids overflow) |
| `kTileC` | 8 | Channel tile width; must be a power of 2 |
| `kMaxPoolH` | 7 | Maximum compile-time pool window height |
| `kMaxPoolW` | 7 | Maximum compile-time pool window width |
| `kMaxLineBufRows` | 16 | Line-buffer row capacity; power of 2; bounds `(pool_h-1)*dil_h + 1` |
| `kMaxLineBufCols` | 64 | Line-buffer column capacity; W-tiling activates when `in_w` exceeds it |
| `kDataMin` | -128.0f | Sentinel for MaxPool identity (ap_fixed\<16,8\> minimum) |
| `kAccMin` | -32768.0f | Sentinel in accumulator range |

---

## 4. Loop Structure and HLS Pragmas

> **The kernel was significantly restructured after the initial implementation.**
> See [POOL_OPTIMIZATION.md](POOL_OPTIMIZATION.md) for the full optimization log
> with timing data; the section below describes the *current* architecture.

The kernel runs four concurrent stages inside a top-level `#pragma HLS DATAFLOW`
region:

```
                  ┌─────────────┐
   x (gmem0) ───► │ row_loader  │ ─row_data_pipe─┐
                  └─────────────┘                │
                                                 ▼
                                      ┌──────────────────┐
                                      │ window_emitter   │ ─window_pipe─┐ (vector kTileC lanes)
                                      │   (line_buf)     │              │
                                      └──────────────────┘ ─denom_pipe─┐│
                                                                       │▼
                                                       ┌────────────────────────┐
                                                       │ process_pool_kernel_tile│
                                                       │   (acc[kTileC])        │
                                                       └────────────────────────┘
                                                                       │
                                                                  acc_stream
                                                                       ▼
                                                       ┌────────────────────────┐
                                                       │ write_output_tile      │
                                                       └─────────┬──────────────┘
                                                                 ▼
                                                          y (gmem1)
```

Loop nest (all stages in lockstep): `(ni, ct, owt, oh, ow)` where `owt`
is a runtime W-tile dimension (collapsed to a single iteration when
`in_w ≤ kMaxLineBufCols`).

```
for ni in [0, batch):                          // batch dimension
  for ct in [0, ceil(channels/kTileC)):        // channel tile (outer of oh)
    for owt in [0, ow_tiles_w):                // W-tile (relaxes in_w cap)
      for oh in [0, out_h):                    // output row
        // row_loader   : load any new x[] rows for this (oh, ct, owt) into row_data_pipe
        // window_emitter: drain row_data_pipe → line_buf
        for ow in [ow_lo, ow_hi):              // output column within W-tile
          // window_emitter: emit denom (parallel adder tree),
          //                 emit pool_h*pool_w WindowLanes vectors
          // process_pool_kernel_tile: fused II=1 vectorised reduce on kTileC lanes
          // write_output_tile: drain c_valid AccData_t lanes to y[]
```

**Key invariants:**

- Each `x[]` cell read from DDR exactly once per `(ni, c)` when
  `in_w ≤ kMaxLineBufCols`. Wider inputs trigger W-tiling with bounded
  boundary-column re-reads (see [POOL_OPTIMIZATION.md §2.3](POOL_OPTIMIZATION.md)).
- `window_pipe` is a vector stream — one `WindowLanes` (kTileC packed
  `Data_t`) per `(khi, kwi)`, so the consumer's reduce loop runs in
  `pool_h × pool_w` cycles instead of `pool_h × pool_w × kTileC`.
- Phase 1 (DDR row loads) overlaps Phase 2 (window emit) via the
  `row_loader`/`window_emitter` split.

**HLS pragmas applied:**

| Pragma | Location | Effect |
|--------|----------|--------|
| `INTERFACE m_axi ... bundle=gmem0/1` | top-level | AXI memory ports |
| `INTERFACE s_axilite ... bundle=ctrl` | every scalar | AXI-Lite register file |
| `STABLE variable=...` | every read-only input | Suppresses synthetic DATAFLOW sync stages |
| `DATAFLOW` | top-level | Concurrent execution of the four stages |
| `STREAM variable=... depth=...` | each `hls::stream` | Sizes the inferred FIFO |
| `ARRAY_PARTITION variable=line_buf complete dim=1` | `window_emitter` | kTileC independent BRAM banks → vector emit |
| `ARRAY_PARTITION variable=acc complete dim=0` | `process_pool_kernel_tile` | kTileC parallel update lanes |
| `PIPELINE II=1` | row load, drain, vector emit, vector reduce, write | One element per clock |
| `UNROLL` | per-lane `c_l` / `c1` inner loops, valid_count adder tree | Spatial parallelism |

**II of the vector reduce loop:**

| Pool type | II achieved | Reason |
|-----------|---:|---|
| MaxPool | 1 | Single-cycle compare |
| AveragePool | 2–3 | ap_fixed<32,16> add latency; 1-cycle RAW distance |
| LpPool p=1 | 1–2 | Conditional negate + add |
| LpPool p=2 | 2–3 | mul + add (DSP MAC) |

---

## 5. Data Types and Saturation

`saturate_cast<Data_t>(v)` converts `AccData_t` back to `Data_t` at the finalization stage. For `ap_fixed`, the specialization uses `AP_TRN` (truncation) and `AP_SAT` (saturation clamping), matching ONNX's fixed-point semantics. A fallback template handles `float` builds (identity cast).

---

## 6. Test Coverage (`TestPoolingSim.cpp`)

31 test cases compiled and run with GCC (no Vitis required). Tolerance: `kTol = 0.02f`.

| Category | Cases |
|----------|-------|
| MaxPool | 2×2 s2, 3×3 s1 pad1, rect 6×10, batch=3, C=16, C=32, dilation=2, Global |
| AveragePool | no-pad, pad1 ±count_include_pad, C=12, rect, Global, batch=2 C=16 |
| LpPool | p=1 and p=2, 2×2, 3×3 pad1, GlobalLpPool (p=1 and p=2) |
| Edge cases | 1×1 output, all-padded corner (3×3 pad1 on 2×2 input) |
| Wide-W (in_w > kMaxLineBufCols) | MaxPool W=128 3×3 pad1, AvgPool W=96 3×3 pad1, MaxPool W=128 2×2 s2 — and batch=2 variants of each |

The dup-read predictor `expected_dup_reads_for(tc)` simulates the kernel's
exact line-buffer + W-tile load schedule, so `dup_reads=actual/predicted`
always shows cache-aware expectations and adapts when `kMaxLineBufCols`,
`kTileC`, or geometry change.

---

## 7. Inference Scheduler Integration

**`PoolNode` (`nodes.py`)** maps ONNX pool ops to kernel invocations:
- Validates NCHW 4-D shapes; parses `kernel_shape`, `strides`, `dilations`, `auto_pad` (NOTSET/VALID/SAME_UPPER/SAME_LOWER), `pads`, `p`, `count_include_pad`; rejects `ceil_mode=1`
- Global variants normalized to `pool=in_spatial, stride=1, pad=0`

**Code-generated `run_pool()` (`_source.py`)** sets all 19 AXI-Lite scalar registers and calls `XPoolingkernel_Start()` — non-blocking. The `inference_run()` body emits a `kernel_wait(KERNEL_POOL)` later, only when a downstream op needs the Pool output or another op wants to reuse the Pool lane, which lets work on other lanes (e.g. Conv, VectorOP) overlap with the Pool.

**Buffer layout (`_core.py`)** packs all pool tensor buffers into a single contiguous 64-byte-aligned DMA allocation. Slot colouring uses event-stream liveness intervals so two tensors share a slot only when one is fully drained before the other's producer starts.

**Reference simulation (`_simulate.py`)** implements float64 `_pool2d_ref()` matching kernel semantics for bit-accurate test comparison.

---

## 8. Build Targets

```bash
# C simulation (GCC, no Vitis)
make TestPoolingSim && ctest

# HLS synthesis + IP export for KV260
make synthesize_pool_kv260
```

The synthesis target reads `kernels/pool/platforms/kv260.json` (specifies part, optional board and clock) and invokes Vitis HLS via `Synthesis.tcl.in`, which configures the project, adds source files, applies directives, runs `csynth_design`, and exports an IP catalog archive.

---

## 9. Key Source Files

| File | Purpose |
|------|---------|
| `kernels/pool/kernel/PoolingKernel.cpp` | HLS kernel implementation |
| `kernels/pool/include/PoolingKernel.h` | Kernel declaration, `Op` enum, `saturate_cast<T>` |
| `kernels/pool/include/Config.h.in` | CMake template → `Config.h` (Data_t, AccData_t, tile constants) |
| `kernels/pool/test/TestPoolingSim.cpp` | C simulation tests (GCC) |
| `kernels/pool/scripts/Synthesis.tcl.in` | Vitis HLS TCL template |
| `kernels/pool/platforms/kv260.json` | KV260 platform config |
| `inference-scheduler/src/nodes.py` | `PoolNode` class (ONNX → kernel params) |
| `inference-scheduler/src/codegen/_source.py` | `run_pool()` code generation |
| `inference-scheduler/src/codegen/_core.py` | Pool node detection, buffer layout |
| `inference-scheduler/src/codegen/_simulate.py` | Float64 reference simulation |
| `inference-scheduler/test/test_pool.py` | Scheduler-level pool tests |

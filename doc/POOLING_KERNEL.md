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
| `kDataMin` | -128.0f | Sentinel for MaxPool identity (ap_fixed\<16,8\> minimum) |
| `kAccMin` | -32768.0f | Sentinel in accumulator range |

---

## 4. Loop Structure and HLS Pragmas

The kernel body is a five-level nested loop:

```
for ni in [0, batch)                    // batch dimension
  for oh in [0, out_h)                  // output row
    for ow in [0, out_w)                // output column
      // precompute: valid_count (non-padded pixels), inv_denom (1/denom)
      for ct in [0, ceil(channels/kTileC))  // channel tile
        // 1. LOAD: fill win_buf[kTileC][kMaxPoolH][kMaxPoolW] from x[]
        //    Pad pixels → identity fill; inner loop PIPELINE II=1
        // 2. INIT: reset acc[kTileC] to identity; UNROLL (registers)
        // 3. REDUCE: flat loop ri ∈ [0, pool_h*pool_w*kTileC); PIPELINE II=1
        //    lane  = ri & (kTileC-1)          // bitwise AND, no divide
        //    acc[lane] OP= win_buf[lane][khi][kwi]
        //    advance khi/kwi every kTileC iterations
        // 4. FINALIZE + WRITE: emit c_valid results; PIPELINE II=1
        //    apply post-reduction op; saturate_cast; write y[]
```

**HLS pragmas applied:**

| Pragma | Location | Effect |
|--------|----------|--------|
| `INTERFACE m_axi ... bundle=gmem0/1` | top-level | AXI memory ports |
| `INTERFACE s_axilite ... bundle=ctrl` | every scalar | AXI-Lite register file |
| `ARRAY_PARTITION variable=win_buf complete dim=1` | `win_buf[kTileC][…][…]` | kTileC independent BRAM banks → parallel access per lane |
| `ARRAY_PARTITION variable=acc complete dim=0` | `acc[kTileC]` | All accumulators in registers |
| `PIPELINE II=1` | load, reduce, write loops | One output per clock |

**II=1 achievability in the reduce loop:**

`acc[lane]` at step `ri` is next accessed at step `ri + kTileC`. Because `kTileC ≥` the operation latency of ap_fixed arithmetic (≈1–3 cycles), the dependency distance is always satisfied and II=1 is met without unrolling.

---

## 5. Data Types and Saturation

`saturate_cast<Data_t>(v)` converts `AccData_t` back to `Data_t` at the finalization stage. For `ap_fixed`, the specialization uses `AP_TRN` (truncation) and `AP_SAT` (saturation clamping), matching ONNX's fixed-point semantics. A fallback template handles `float` builds (identity cast).

---

## 6. Test Coverage (`TestPoolingSim.cpp`)

21 test cases compiled and run with GCC (no Vitis required). Tolerance: `kTol = 0.02f`.

| Category | Cases |
|----------|-------|
| MaxPool | 2×2 s2, 3×3 s1 pad1, rect 6×10, batch=3, C=16, dilation=2, Global |
| AveragePool | no-pad, pad1 ±count_include_pad, C=12, rect, Global, batch=2 C=16 |
| LpPool | p=1 and p=2, 2×2, 3×3 pad1, GlobalLpPool (p=1 and p=2) |
| Edge cases | 1×1 output, all-padded corner (3×3 pad1 on 2×2 input) |

---

## 7. Inference Scheduler Integration

**`PoolNode` (`nodes.py`)** maps ONNX pool ops to kernel invocations:
- Validates NCHW 4-D shapes; parses `kernel_shape`, `strides`, `dilations`, `auto_pad` (NOTSET/VALID/SAME_UPPER/SAME_LOWER), `pads`, `p`, `count_include_pad`; rejects `ceil_mode=1`
- Global variants normalized to `pool=in_spatial, stride=1, pad=0`

**Code-generated `run_pool()` (`_source.py`)** sets all 19 AXI-Lite scalar registers, calls `XPoolingkernel_Start()`, and spins on `XPoolingkernel_IsDone()`.

**Buffer layout (`_core.py`)** packs all pool tensor buffers into a single contiguous 64-byte-aligned DMA allocation.

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

# Convolutional Kernel — Detailed Implementation Description

## Overview

`ConvKernel` is a Vitis HLS kernel implementing ONNX-compliant 2-D convolution on NCHW tensors. It is one of four hardware kernels in the `axi_demo` project, targeting the Xilinx KV260 FPGA. The kernel supports standard convolution (group=1) and depthwise convolution (group=in_ch), optional per-channel bias, padding, stride, and dilation. A two-level channel-tiling strategy (output-channel tile kTileM × input-channel tile kTileIC) enables II=1 throughput via a flat-counter lane-rotation scheme.

---

## 1. AXI Interface

**Memory ports (m_axi):**

| Bundle | Port | Direction | Description |
|--------|------|-----------|-------------|
| `gmem0` | `x` | Read | Input feature map `[batch][in_ch][in_h][in_w]` |
| `gmem1` | `weight` | Read | Filter weights (layout depends on mode) |
| `gmem2` | `bias` | Read | Per-channel bias `[out_ch]` (not accessed when `has_bias=0`) |
| `gmem3` | `y` | Write | Output feature map `[batch][out_ch][out_h][out_w]` |

**AXI-Lite control registers (`s_axilite bundle=ctrl`) — 21 registers total:**

| Register | Type | Description |
|----------|------|-------------|
| `x`, `weight`, `bias`, `y` | `uint64_t` | Physical DDR base addresses |
| `batch` | `unsigned` | Batch size N |
| `in_ch`, `in_h`, `in_w` | `unsigned` | Input tensor dimensions |
| `out_ch`, `out_h`, `out_w` | `unsigned` | Output tensor dimensions |
| `kh`, `kw` | `unsigned` | Filter kernel size |
| `stride_h`, `stride_w` | `unsigned` | Convolution stride |
| `dilation_h`, `dilation_w` | `unsigned` | Dilation |
| `pad_top`, `pad_left` | `unsigned` | Padding (top row / left column) |
| `has_bias` | `unsigned` | 0 = skip bias; 1 = add per-channel bias |
| `is_depthwise` | `unsigned` | 0 = standard (group=1); 1 = depthwise (group=in_ch) |

64-bit AXI addressing is configured in the synthesis TCL (`config_interface -m_axi_addr64`).

---

## 2. Supported Modes

### Standard Convolution (`is_depthwise=0`)

- Weight layout: `[out_ch][in_ch][kh][kw]`
- Each output channel is the inner product of the full input-channel stack against the corresponding filter
- Supported: bias, padding, stride, dilation, multi-tile M and IC

### Depthwise Convolution (`is_depthwise=1`)

- Weight layout: `[out_ch][1][kh][kw]`
- Each output channel convolves with exactly one input channel (group=in_ch)
- No IC-tile loop; each tile lane operates on its own input channel slice
- Supported: bias, padding, stride, dilation

**Unsupported:** grouped convolution with 1 < group < in_ch is rejected by the scheduler.

---

## 3. Compile-Time Configuration (`Config.h.in`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `Data_t` | `ap_fixed<16,8>` | Element type (2-byte, range \[-128, 127.996\]) |
| `AccData_t` | `ap_fixed<32,16>` | Accumulator type (wider range, avoids overflow) |
| `kTileM` | 8 | Output-channel tile width; must be a power of 2 |
| `kTileIC` | 16 | Input-channel tile width; must be a power of 2 |
| `kMaxKH` | 7 | Maximum compile-time kernel height |
| `kMaxKW` | 7 | Maximum compile-time kernel width |

If Vitis HLS headers are unavailable at CMake configure time, both types fall back to `float`.

---

## 4. On-Chip Memory

```cpp
static Data_t    patch[kTileIC][kMaxKH][kMaxKW];
// Standard conv: holds input patch for current (oh, ow, ic_tile)
// Depthwise conv: holds spatial patches for kTileM lanes (one per channel lane)
// ARRAY_PARTITION complete dim=1 → kTileIC independent BRAM banks

static Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
// Holds weight tile for current (m_tile, ic_tile)
// Depthwise: only [m1][0][khi][kwi] slots used
// ARRAY_PARTITION complete dim=1 → kTileM independent register arrays

static AccData_t acc[kTileM];
// One accumulator per output-channel lane
// ARRAY_PARTITION complete dim=0 → all registers independent
```

---

## 5. Loop Structure and HLS Pragmas

### Standard Convolution

```
for ni in [0, batch)                          // batch
  for oh in [0, out_h)                        // output row
    for ow in [0, out_w)                      // output column
      for mt in [0, ceil(out_ch / kTileM))   // output-channel tile
        // INIT: acc[0..kTileM-1] = 0  (UNROLL, 1 cycle)
        // BIAS LOAD (if has_bias): acc[m1] = bias[m_off + m1]  (PIPELINE II=1)
        for ict in [0, ceil(in_ch / kTileIC))   // input-channel tile
          // LOAD PATCH: fill patch[ic_l][khi][kwi] from x[]
          //   out-of-bounds → 0 (implicit zero-padding); PIPELINE II=1
          // LOAD WEIGHTS: fill w_buf[m1][ic_l][khi][kwi] from weight[]
          //   per-m1 pointer stride over weight DDR; PIPELINE II=1
          // REDUCE: flat counter ri ∈ [0, ic_valid * kh * kw * kTileM)
          //   m1 = ri & (kTileM - 1)              // bitwise AND, no divide
          //   acc[m1] += patch[ic_cnt][khi_cnt][kwi_cnt]
          //            * w_buf[m1][ic_cnt][khi_cnt][kwi_cnt]
          //   advance ic_cnt / khi_cnt / kwi_cnt every kTileM iterations
          //   PIPELINE II=1
        // WRITE: emit m_valid outputs; saturate_cast; stride y[] by out_h*out_w
        //   PIPELINE II=1
```

### Depthwise Convolution

```
for ni in [0, batch)
  for oh in [0, out_h)
    for ow in [0, out_w)
      for mt in [0, ceil(out_ch / kTileM))
        // INIT: acc[0..kTileM-1] = 0  (UNROLL)
        // BIAS LOAD (if has_bias): PIPELINE II=1
        // LOAD PATCH: patch[m1][khi][kwi] from input channel (m_off + m1)
        //   PIPELINE II=1 (inner khi/kwi loop, per m1 lane)
        // LOAD WEIGHTS: w_buf[m1][0][khi][kwi] from weight[m_off + m1]
        //   PIPELINE II=1
        // REDUCE: flat counter ri ∈ [0, kh * kw * kTileM)
        //   m1 = ri & (kTileM - 1)
        //   acc[m1] += patch[m1][khi_cnt][kwi_cnt]
        //            * w_buf[m1][0][khi_cnt][kwi_cnt]
        //   PIPELINE II=1
        // WRITE: PIPELINE II=1
```

**HLS pragmas applied:**

| Pragma | Location | Effect |
|--------|----------|--------|
| `INTERFACE m_axi ... bundle=gmem0/1/2/3` | top-level | AXI memory ports |
| `INTERFACE s_axilite ... bundle=ctrl` | every scalar | AXI-Lite register file |
| `ARRAY_PARTITION variable=patch complete dim=1` | `patch[kTileIC][…][…]` | kTileIC independent BRAM banks |
| `ARRAY_PARTITION variable=w_buf complete dim=1` | `w_buf[kTileM][…][…][…]` | kTileM independent register arrays |
| `ARRAY_PARTITION variable=acc complete dim=0` | `acc[kTileM]` | All accumulators in registers |
| `PIPELINE II=1` | load, reduce, write loops | One operation per clock |

**II=1 achievability in the reduce loop:**

`acc[m1]` at step `ri` is next written at step `ri + kTileM`. Because `kTileM` (default 8) exceeds the MAC latency of `ap_fixed<16,8>` arithmetic (≈3 cycles), the RAW dependency distance is always satisfied and II=1 is met without unrolling.

---

## 6. Data Types and Saturation

`saturate_cast<Data_t>(v)` converts `AccData_t` accumulator back to `Data_t` at the write stage. For `ap_fixed` the specialization uses `AP_TRN` (truncation toward zero) and `AP_SAT` (saturation clamping), matching ONNX fixed-point semantics. A fallback template handles `float` builds (identity cast).

---

## 7. Test Coverage (`TestConvSim.cpp`)

20 test cases compiled with GCC (no Vitis required). Tolerance: exact match for `ap_fixed`, relative 1e-5 for `float`.

**Reference implementations:**
- `ref_conv()` — naive 7-nested-loop standard convolution
- `ref_depthwise_conv()` — naive 6-nested-loop depthwise convolution

| Category | Cases |
|----------|-------|
| Standard conv — basic | 1×1 kernel; 3×3 no-pad; 3×3 same-pad; 3×3 stride=2 |
| Standard conv — bias/batch | pad=1 + bias + 2 output channels; batch=2 |
| Standard conv — partial tiles | in_ch = kTileIC+5; out_ch = kTileM+3 |
| Standard conv — dilation/kernel | dilation=2; 5×5 kernel; 14×14 input multi-tile; non-square 6×8 input 3×5 kernel |
| Standard conv — asymmetric pad | mixed stride and asymmetric padding \[1,1,0,0\] |
| Depthwise conv | 3×3 no-bias; 3×3 pad=1+bias; partial TILE_M; dilation=2; stride=2 pad=1 |
| Saturation (ap_fixed only) | positive overflow → AP_MAX; negative overflow → AP_MIN |

---

## 8. Inference Scheduler Integration

**`ConvNode` (`nodes.py`)** maps ONNX `Conv` operators to `XConvkernel` invocations:
- Validates 4-D NCHW shapes for input, weight, bias, and output
- Parses `group`, `strides`, `dilations`, `pads`, `auto_pad` (NOTSET/VALID/SAME_UPPER/SAME_LOWER)
- Determines `is_depthwise`: group=1 → standard, group=in_ch → depthwise, otherwise rejected
- Enforces `kh ≤ kMaxKH`, `kw ≤ kMaxKW`

**Code-generated `run_conv()` (`_source.py`)** sets all 21 AXI-Lite registers, calls `XConvkernel_Start()`, and spins on `XConvkernel_IsDone()`. The `bias` argument may be `NULL` when `has_bias=0`; `gmem2` is not accessed by the kernel in that case.

**Layout constraint (`_core.py`):** `ConvKernel` writes a flat NCHW output. If the output tensor feeds a broadcast `VectorOP` node that requires an advancing-strided layout (`n_chunks > 1`), the scheduler raises a `SchedulerError`. Per-channel bias must be passed as the Conv operator's 3rd input, not as a separate downstream `Add` node.

**Reference simulation (`_simulate.py`):** `_conv2d_ref()` and `_depthwise_conv2d_ref()` implement float64 references matching kernel semantics (same padding, dilation, bias handling) for bit-accurate test comparison. Outputs are quantized via `dtype.truncate()` at node boundaries.

---

## 9. Build Targets

```bash
# C simulation (GCC, no Vitis)
make TestConvRef && ctest

# HLS synthesis + IP export for KV260
make synthesize_conv_kv260
```

The synthesis target reads `kernels/conv/platforms/kv260.json` (specifies part, optional board and clock) and invokes Vitis HLS via `Synthesis.tcl.in`, which configures the project, sets 64-bit AXI and bus width, runs `csynth_design`, and exports an IP catalog archive.

---

## 10. Key Source Files

| File | Purpose |
|------|---------|
| `kernels/conv/kernel/ConvKernel.cpp` | HLS kernel implementation |
| `kernels/conv/include/ConvKernel.h` | Kernel declaration, `saturate_cast<T>` |
| `kernels/conv/include/Config.h.in` | CMake template → `Config.h` (Data_t, AccData_t, tile constants) |
| `kernels/conv/test/TestConvSim.cpp` | C simulation tests (GCC) |
| `kernels/conv/scripts/Synthesis.tcl.in` | Vitis HLS TCL template |
| `kernels/conv/platforms/kv260.json` | KV260 platform config |
| `inference-scheduler/src/nodes.py` | `ConvNode` class (ONNX → kernel params) |
| `inference-scheduler/src/codegen/_source.py` | `run_conv()` code generation |
| `inference-scheduler/src/codegen/_core.py` | Conv node detection, layout validation |
| `inference-scheduler/src/codegen/_simulate.py` | Float64 reference simulation |

---

## 11. Summary

| Aspect | Details |
|--------|---------|
| **Supported ONNX op** | `Conv` (2-D, NCHW layout) |
| **Modes** | Standard (group=1), Depthwise (group=in_ch) |
| **Data type** | `ap_fixed<16,8>` (default) or `float` |
| **Accumulator type** | `ap_fixed<32,16>` (default) or `float` |
| **Tiling** | kTileM=8 output channels × kTileIC=16 input channels |
| **Initiation interval** | II=1 (all pipelined inner loops) |
| **Lane rotation** | Flat counter `ri`, bitmask `m1 = ri & (kTileM-1)` |
| **AXI master ports** | 4 (gmem0 input, gmem1 weight, gmem2 bias, gmem3 output) |
| **AXI-Lite registers** | 21 scalars |
| **Padding** | Implicit zero-pad (out-of-bounds reads return 0) |
| **Kernel size limit** | kMaxKH=7, kMaxKW=7 (compile-time) |
| **Bias** | Optional 3rd DDR input; guarded by `has_bias` flag |
| **Weight layout (standard)** | `[out_ch][in_ch][kh][kw]` |
| **Weight layout (depthwise)** | `[out_ch][1][kh][kw]` |
| **AXI-Lite base address** | `0xA002_0000` |
| **Driver prefix** | `xconvkernel` |
| **UIO device name** | `ConvKernel_0` |
| **Test coverage** | 20 test cases (standard, depthwise, saturation, dilation) |

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
| `kTileM` | 8 | Output-channel tile width; must be a power of 2; also the depthwise PM unroll factor |
| `kTileIC` | 16 | Input-channel tile width; must be a power of 2; also the standard PN unroll factor |
| `kMaxKH` | 7 | Maximum compile-time kernel height |
| `kMaxKW` | 7 | Maximum compile-time kernel width |
| `kMaxInCh` | 1024 | Sizes `bias_buf` (line_buf is IC-tiled, doesn't depend on this) |
| `kMaxOutCh` | 1024 | Sizes `bias_buf` in `bias_producer` |
| `kMaxInW` | 64 | Column dimension of `line_buf` |
| `kMaxLineBufRows` | 16 | Circular row capacity of `line_buf`; power of 2 (used as bitmask) |
| `kMaxAccPersistEntries` | 16384 | `partial_outputs[]` buffer size; one output row (`out_w·out_ch`) must fit |

If Vitis HLS headers are unavailable at CMake configure time, both types fall back to `float`.

**Runtime constraints validated by the inference scheduler:**

- `in_ch ≤ kMaxInCh`, `out_ch ≤ kMaxOutCh`, `in_w ≤ kMaxInW`
- `(kh-1)·dilation_h + 1 ≤ kMaxLineBufRows`
- `out_w · out_ch ≤ kMaxAccPersistEntries` *(was the stricter `out_h · out_w · out_ch ≤ …` before the oh-chunking restructure)*

When `out_h · out_w · out_ch > kMaxAccPersistEntries` the kernel transparently splits the output along the `oh` axis into chunks of `oh_per_chunk = floor(kMaxAccPersistEntries / (out_w·out_ch))` rows; see §5.3.

---

## 4. On-Chip Memory

Buffers are declared inside `process_conv_kernel_tile` (re-allocated per inner
iteration; HLS hoists them to BRAM/registers).

```cpp
// Per-(oh, ow, mt) scratch — partitioned for the parallel MAC inner loop.

Data_t    patch[kTileIC][kMaxKH][kMaxKW];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=0
// All three dimensions fully partitioned → every cell is a register.
// Standard: patch[ic_l][khi][kwi] for current (oh, ow, ic_tile).
// Depthwise: patch[m1][khi][kwi]  for current (oh, ow, m_tile);
// the [kTileIC] depth covers kTileM lanes (kTileM ≤ kTileIC).

// STANDARD-path weight buffer:
Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2
// dim=2 (ic_l, PN axis) partitioned complete → kTileIC parallel banks.
// accumulate_standard reads kTileIC weights per cycle along the ic_l axis;
// m1 is sequentially rotated and kh/kw are runtime-shared, so dims 1, 3, 4
// stay default (no partition).

// DEPTHWISE-path weight buffer (different shape — no in_ch dimension):
Data_t    w_buf[kTileM][kMaxKH][kMaxKW];
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
// dim=1 (m1, PM axis) partitioned complete → kTileM parallel banks.
// accumulate_depthwise reads kTileM weights per cycle along the m1 axis.

AccData_t acc[kTileM];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
// All kTileM accumulators in registers (independent).

// Per-(ni, chunk) persistent state — chunk-scoped, lives in the consumer:
AccData_t partial_outputs[kMaxAccPersistEntries];
// Holds chunk_oh_count·out_w·out_ch accumulators that survive across
// ic-tiles (standard) or mt-tiles (depthwise) within a chunk.  Indexed
// by (oh_local·out_w + ow)·out_ch + m_off + m1 where oh_local = oh - oh_start.

// Per-(ni, chunk, ict) line buffer — lives in the standard patch producer:
Data_t    line_buf[kTileIC][kMaxLineBufRows][kMaxInW];
// Within a (chunk, ict) each input pixel is fetched from DDR exactly once;
// the (kh-1)-row overlap is re-fetched at chunk boundaries.

// Per-(ni, chunk, mt) line buffer — depthwise variant, same shape but
// indexed by m1 instead of ic_l:
Data_t    line_buf[kTileM][kMaxLineBufRows][kMaxInW];
```

---

## 5. Loop Structure and HLS Pragmas

The kernel is a top-level `#pragma HLS DATAFLOW` region with six concurrent
sub-functions (see §3 of [CONV_OPTIMISATION.md](CONV_OPTIMISATION.md) for the
dataflow diagram).  Each function owns its own m_axi port (or stream) and
implements one of the six pipeline stages: input patch assembly, IC×M
broadcast, weight streaming, bias streaming, the conv compute consumer, and
the saturating output writer.

### 5.1 Standard Convolution — consumer loop nest

```
for ni in [0, batch)
  for chunk in [0, num_chunks)                          // §5.3 oh-chunking
    oh_start  = chunk · oh_per_chunk
    oh_end    = min(out_h, oh_start + oh_per_chunk)
    chunk_oh  = oh_end - oh_start

    // PHASE 1: init partial_outputs from bias_stream — PIPELINE II=1
    for oh_local, ow, mt, m1:
      partial_outputs[(oh_local·out_w + ow)·out_ch + m_off + m1]
         = bias_stream.read()

    // PHASE 2a: accumulate (ict OUTER of oh_in_chunk)
    for ict in [0, ceil(in_ch / kTileIC))
      for oh_local in [0, chunk_oh)
        for ow in [0, out_w)
          for mt in [0, ceil(out_ch / kTileM))
            // Drain kTileIC × kh × kw patch values from patch_stream — PIPELINE II=1
            // Drain m_valid × ic_valid × kh × kw weights from weight_stream — PIPELINE II=1
            // acc[0..kTileM-1] := partial_outputs[idx_base + …]    (II=1)
            // accumulate_standard():
            //   for ri in [0, kh · kw · kTileM):                   PIPELINE II=1
            //     m1 = ri & (kTileM - 1)                           // lane rotation
            //     lane_sum = Σ_{ic_l = 0..kTileIC-1, UNROLL}
            //                  patch[ic_l][khi][kwi] · w_buf[m1][ic_l][khi][kwi]
            //                  // w_buf masked to 0 for ic_l ≥ ic_valid (X-prop guard)
            //     acc[m1] += lane_sum
            // partial_outputs[idx_base + …] := acc[m1]              (II=1)

    // PHASE 3: drain partial_outputs to acc_stream — PIPELINE II=1
    for oh_local, ow, mt, m1:
      acc_stream.write(partial_outputs[…])
```

**Inner-MAC throughput is `kTileIC` MACs/cycle** (PN-wide adder tree fed by
the unrolled `ic_l` loop).  Loop bound shrinks from
`ic_valid · kh · kw · kTileM` to `kh · kw · kTileM`; lane rotation on `m1`
preserves the kTileM-cycle RAW distance on `acc[m1]`.

### 5.2 Depthwise Convolution — consumer loop nest

```
for ni in [0, batch)
  for chunk in [0, num_chunks)
    // PHASE 1: identical to standard

    // PHASE 2b: accumulate (mt OUTER of oh_in_chunk)
    for mt in [0, ceil(out_ch / kTileM))
      // Load w_buf[kTileM][kh][kw] ONCE per (chunk, mt) — PIPELINE II=1
      for oh_local in [0, chunk_oh)
        for ow in [0, out_w)
          // Drain kTileM × kh × kw patch values from patch_stream — II=1
          // acc[0..kTileM-1] := partial_outputs[idx_base + …]   (II=1)
          // accumulate_depthwise():
          //   for ri in [0, kh · kw):                            PIPELINE II=1
          //     for m1 in [0, kTileM), UNROLL:
          //       acc[m1] += patch[m1][khi][kwi] · w_buf[m1][khi][kwi]
          // partial_outputs[idx_base + …] := acc[m1]              (II=1)

    // PHASE 3: drain — identical to standard
```

**Inner-MAC throughput is `kTileM` MACs/cycle** (PM-wide channel-parallel
lanes; depthwise has no input-channel reduction).  Loop bound shrinks from
`kh · kw · kTileM` to `kh · kw`.

### 5.3 oh-chunking

When `out_h · out_w · out_ch > kMaxAccPersistEntries` the persistent
`partial_outputs[]` buffer cannot hold the full output.  Rather than reject
such layers or fall back to a no-persistent-acc mode, the kernel splits the
output along `oh` into chunks that fit:

```
oh_per_chunk = max(1, kMaxAccPersistEntries / (out_w · out_ch))
num_chunks   = ceil(out_h / oh_per_chunk)
```

Each chunk runs the full three-phase pipeline above for its `oh` sub-range.
The chunk loop is placed INNER to `ni` (and outer to everything else) in
all producers + the consumer so the linear stream order seen by
`bias_producer`, `broadcast_patches`, and `write_output_tile` is the same
`(ni, oh, ow, mt, m1)` as before — those three need no chunk-awareness.

**Duplicate-read overhead** at chunk boundaries:

- Input: the patch producer's `line_buf` is invalidated when (chunk, ict) advances; the next chunk re-loads its first kh-row window (`(kh-1)·stride_h` rows re-fetched per (ni, ict) chunk transition).
- Weights (standard path): unchanged — weights were already replayed per `(oh, ow, mt)`; chunking just slices the replay axis.
- Weights (depthwise path): the small per-mt slice (`kTileM·kh·kw` values) is reloaded `num_chunks` times per `(ni, mt)`. Negligible.

For the common case (`out_h · out_w · out_ch ≤ kMaxAccPersistEntries`)
`num_chunks = 1` and the chunk loop adds only a few cycles of wrapper
overhead.  See `compute_oh_chunking()` in `ConvKernel.cpp`.

### 5.4 HLS pragmas

| Pragma | Location | Effect |
|--------|----------|--------|
| `DATAFLOW` | top-level | Six concurrent producers/consumers |
| `INTERFACE m_axi ... bundle=gmem0/1/2/3` | top-level | AXI memory ports |
| `INTERFACE s_axilite ... bundle=ctrl` | every scalar | AXI-Lite register file |
| `STABLE variable={x,weight,bias}` | top-level | Tells HLS the base pointers don't change across DATAFLOW processes |
| `ARRAY_PARTITION variable=patch complete dim=0` | `patch[kTileIC][kMaxKH][kMaxKW]` | All cells become registers |
| `ARRAY_PARTITION variable=w_buf complete dim=2` | standard `w_buf[kTileM][kTileIC][kMaxKH][kMaxKW]` | kTileIC banks for the PN unroll |
| `ARRAY_PARTITION variable=w_buf complete dim=1` | depthwise `w_buf[kTileM][kMaxKH][kMaxKW]` | kTileM banks for the PM unroll |
| `ARRAY_PARTITION variable=acc complete dim=0` | `acc[kTileM]` | All accumulators in registers |
| `PIPELINE II=1` | every load / reduce / drain loop | One iteration per clock |
| `UNROLL` | inner `ic_l` loop (standard) / inner `m1` loop (depthwise) | Replicates MACs across the parallel axis |
| `STREAM depth=…` | every `hls::stream` between dataflow stages | FIFO sizing (e.g. `weight_stream` = `kTileM·kTileIC·kMaxKH·kMaxKW`) |

### 5.5 II=1 achievability in the reduce loops

**Standard (`accumulate_standard`).**  Each PIPELINE iteration fires a
`kTileIC`-wide PN adder tree feeding one `acc[m1] += lane_sum` per cycle.
The lane rotation `m1 = ri & (kTileM-1)` cycles through `kTileM` lanes, so
the RAW distance on any individual `acc[m1]` is `kTileM` cycles — enough to
cover the multiplier latency (1 DSP cycle) + adder-tree depth
`log2(kTileIC) = 4` + final accumulator add.  HLS schedules II=1 without
needing a `DEPENDENCE` escape.

**Depthwise (`accumulate_depthwise`).**  Each PIPELINE iteration writes
*all* `kTileM` accumulators (PM-wide unroll).  The per-lane RAW distance on
`acc[m1]` is 1 cycle.  Because `ap_fixed<32,16>` add is a single-cycle
32-bit integer adder at 300 MHz, the single-cycle recurrence closes
cleanly and HLS schedules II=1.

**X-propagation guard (standard only).**  For partial IC tiles
(`ic_valid < kTileIC`), `w_buf[m1][ic_l ≥ ic_valid][…]` is left
uninitialised — `X` in RTL.  C-sim sees zero (because `patch` is producer-
zero-padded) but RTL `0 · X = X` would propagate to the AXI output.  The
`accumulate_standard` inner loop guards the weight read with
`ic_l < ic_valid ? w_buf[…] : 0`, MUXing the bank output to 0 on invalid
lanes so the product is `0 · 0 = 0`.  Cost: one LUT per PN lane on the
weight input; no DSP impact.

---

## 6. Data Types and Saturation

`saturate_cast<Data_t>(v)` converts `AccData_t` accumulator back to `Data_t` at the write stage. For `ap_fixed` the specialization uses `AP_TRN` (truncation toward zero) and `AP_SAT` (saturation clamping), matching ONNX fixed-point semantics. A fallback template handles `float` builds (identity cast).

---

## 7. Test Coverage (`TestConvSim.cpp`)

32 test cases compiled with GCC (no Vitis required). Tolerance: exact match for `ap_fixed`, relative 1e-5 for `float`. The RTL behavior testbench currently uses pre-baked fixtures for 30 of these (the saturation and chunking tests are C-sim-only until `make gen_conv_test_data` is re-run).

**Reference implementations:**
- `ref_conv()` — naive 7-nested-loop standard convolution
- `ref_depthwise_conv()` — naive 6-nested-loop depthwise convolution

| Category | Cases |
|----------|-------|
| Standard conv — basic | 1×1 kernel; 3×3 no-pad; 3×3 same-pad; 3×3 stride=2 |
| Standard conv — bias/batch | pad=1 + bias + 2 output channels; batch=2 |
| Standard conv — partial tiles | in_ch = kTileIC+5; out_ch = kTileM+3; 1×1 exact-tile multiples |
| Standard conv — dilation/kernel | dilation=2; 5×5 kernel; 14×14 input multi-tile; non-square 6×8 input 3×5 kernel; 1×5 horizontal |
| Standard conv — asymmetric pad/stride/dilation | 7×7 stride=2 asymmetric pad; 3×3 stride h=2 w=1; 3×3 dilation h=1 w=2 |
| Standard conv — batch + tiling | batch=3 C=TILE_IC M=TILE_M stride=2 (ResNet-style) |
| **Standard conv — oh-chunking** | **out=32×32×32 (2 chunks; exercises §5.3)** |
| Depthwise conv | 3×3 no-bias; 3×3 pad=1+bias; partial TILE_M+3; dilation=2; stride=2 pad=1; batch=2; exact TILE_M×2 + bias; 5×5 kernel; asymmetric stride |
| **Depthwise conv — oh-chunking** | **32 ch / 32×32 out (2 chunks)** |
| Saturation (ap_fixed only) | std positive overflow → AP_MAX; std negative overflow → AP_MIN; DW positive overflow → AP_MAX |

---

## 8. Inference Scheduler Integration

**`ConvNode` (`nodes.py`)** maps ONNX `Conv` operators to `XConvkernel` invocations:
- Validates 4-D NCHW shapes for input, weight, bias, and output
- Parses `group`, `strides`, `dilations`, `pads`, `auto_pad` (NOTSET/VALID/SAME_UPPER/SAME_LOWER)
- Determines `is_depthwise`: group=1 → standard, group=in_ch → depthwise, otherwise rejected
- Enforces `kh ≤ kMaxKH`, `kw ≤ kMaxKW`

**Code-generated `run_conv()` (`_source.py`)** sets all 21 AXI-Lite registers and calls `XConvkernel_Start()` — non-blocking. The `inference_run()` body emits a `kernel_wait(KERNEL_CONV)` later, only when a downstream op needs the Conv output or another op wants to reuse the Conv lane, which lets work on other lanes (e.g. Pool, VectorOP) overlap with the Conv. `bias` may be `NULL` when `has_bias=0`; `gmem2` is not accessed by the kernel in that case.

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
| **Inner-MAC parallelism (standard)** | PN-wide adder tree: kTileIC=16 MACs/cycle, lane-rotated on m1 |
| **Inner-MAC parallelism (depthwise)** | PM-wide channel-parallel: kTileM=8 MACs/cycle |
| **Initiation interval** | II=1 (all pipelined inner loops; see §5.5) |
| **Dataflow stages** | 6 (input_patch_producer, broadcast_patches, bias_producer, stream_load_weights, process_conv_kernel_tile, write_output_tile) |
| **oh-chunking** | Auto-splits output along oh when `out_h·out_w·out_ch > kMaxAccPersistEntries`; duplicate input rows re-fetched at chunk boundaries |
| **AXI master ports** | 4 (gmem0 input, gmem1 weight, gmem2 bias, gmem3 output) |
| **AXI-Lite registers** | 21 scalars |
| **Padding** | Implicit zero-pad (out-of-bounds reads return 0) |
| **Kernel size limit** | kMaxKH=7, kMaxKW=7 (compile-time) |
| **Persistent acc constraint** | `out_w·out_ch ≤ kMaxAccPersistEntries` *(was `out_h·out_w·out_ch ≤ …` pre-chunking)* |
| **Bias** | Optional 3rd DDR input; guarded by `has_bias` flag |
| **Weight layout (standard)** | `[out_ch][in_ch][kh][kw]` |
| **Weight layout (depthwise)** | `[out_ch][1][kh][kw]` |
| **AXI-Lite base address** | `0xA002_0000` |
| **Driver prefix** | `xconvkernel` |
| **UIO device name** | `ConvKernel_0` |
| **Test coverage** | 32 C-sim cases (30 covered by RTL fixtures pending fixture regen for the two oh-chunking tests) |

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AXI-stream vector operation IP core for Xilinx FPGAs, implemented in Vitis HLS. The kernel reads up to two equal-length element arrays, applies a runtime-selected element-wise operation, and writes the results to a third array. Six operations are supported (Add, Sub, Mul, Div, Relu, Relu6); the selector is a runtime AXI-Lite register. The vector length and element data type are also configurable at runtime and CMake configure time respectively.

The repository also contains **`inference-scheduler/`**, a Python code-generator that parses an ONNX model and emits a self-contained C file that drives `VectorOPKernel` sequentially for every supported layer using the generated XVectoropkernel driver API and the Xil bare-metal library.

## Build System

All four kernels live under `kernels/` and are built from a single top-level CMake project. `hlslib` is not bundled — it is read from the sibling `gemm_hls` project by default.

```bash
mkdir build && cd build

# Configure all four kernels at once
cmake ../

# C simulation tests (no hardware needed)
make TestSimulation    # VectorOPKernel
make TestConvRef       # ConvKernel
make TestMatmulRef     # MatmulKernel
make TestPoolingSim    # PoolingKernel
ctest                  # run all

# HLS synthesis + Vivado IP export for the KV260
make synthesize_vectorop_kv260
make synthesize_conv_kv260
make synthesize_matmul_kv260
make synthesize_pool_kv260
```

Each kernel can also be built standalone:

```bash
cd kernels/vectorop && mkdir build && cd build
cmake ../ -DVA_DATA_TYPE=ap_fixed\<16,8\>
make TestSimulation
make synthesize_vectorop_kv260
```

### Key CMake Parameters (VectorOPKernel)

| Parameter | Default | Description |
|---|---|---|
| `VA_DATA_TYPE` | `ap_fixed<16,8>` | Element type: `float`, `double`, `half`, `uint8_t`, `ap_fixed<W,I>` |
| `VA_HLSLIB_DIR` | `../../../gemm_hls/hlslib` | Path to hlslib (must contain `cmake/` and `include/`) |
| `VA_PLATFORM` | `xilinx_u250_…` | Vitis platform for the `hw` / `hw_emu` xclbin flow |
| `VA_TARGET_CLOCK` | *(empty)* | Target MHz for Vitis flow; empty = platform default |
| `VA_VECTOR_SIZE` | 1024 | Informational default written to `Config.h` |

### Per-platform HLS synthesis

Each `kernels/<kernel>/platforms/<name>.json` file defines a `synthesize_<kernel>_<name>` CMake target that runs Vitis HLS and exports an IP catalog archive.

**Required JSON fields:** `part`
**Optional JSON fields:** `board`, `clock` (default 300 MHz)

Adding a new platform requires only a JSON file and a re-run of cmake.

## Architecture

### Kernel Interface

```
DDR/PL ──► m_axi_gmem0 (a, read)  ─┐
DDR/PL ──► m_axi_gmem1 (b, read)  ─┤  VectorOPKernel  ├──► m_axi_gmem2 (c, write) ──► DDR/PL
           s_axi_ctrl: a_addr, b_addr, c_addr, size, op, ap_ctrl_hs
```

HLS infers sequential burst reads on `gmem0`/`gmem1` and a burst write on `gmem2`. The loop body is pipelined at II=1. For unary ops (Relu, Relu6) no AXI transactions are issued on `gmem1`.

### Supported Operations

| Code | Name | Expression | Arity |
|------|------|------------|-------|
| 0 | `OP_ADD` | `saturate_cast(a[i] + b[i])` | binary |
| 1 | `OP_SUB` | `saturate_cast(a[i] - b[i])` | binary |
| 2 | `OP_MUL` | `saturate_cast(a[i] * b[i])` | binary |
| 3 | `OP_DIV` | `saturate_cast(a[i] / b[i])` | binary |
| 4 | `OP_RELU` | `max(a[i], 0)` | unary |
| 5 | `OP_RELU6` | `min(max(a[i], 0), 6)` | unary |

### Key Files

- **`kernels/vectorop/kernel/VectorOP.cpp`** — HLS kernel. Single pipelined loop over `a[]`, `b[]`, `c[]` with a switch on `op`. Three `m_axi` ports (`gmem0/1/2`) with `offset=slave`; all registers (addresses, `size`, `op`, return) are in `s_axi_ctrl`.
- **`kernels/vectorop/include/VectorOP.h`** — Kernel declaration, `Op` enum (OP_ADD … OP_RELU6), and `saturate_cast<T>` template.
- **`kernels/vectorop/include/Config.h.in`** — CMake template that produces `Config.h` with `Data_t`, `kDataWidthBits`, and `kSeed`.
- **`kernels/vectorop/test/TestSimulation.cpp`** — Tests all 6 operations across multiple sizes plus saturation boundary cases. Tolerance: relative 1e-5 for FP, exact for integers.
- **`kernels/vectorop/scripts/Synthesis.tcl.in`** — Vitis HLS TCL template. CMake substitutes paths, flags, and part strings; generates one `.tcl` per platform under `build/<name>/`.
- **`kernels/vectorop/platforms/kv260.json`** — KV260 Starter Kit platform config.

### Inference Scheduler

**`inference-scheduler/`** — Python code-generator. Parses an ONNX model and
emits a complete C project that drives up to four hardware kernels:

| Kernel | ONNX ops |
|--------|----------|
| VectorOPKernel | `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(0,6)` |
| MatmulKernel | `MatMul` |
| ConvKernel | `Conv` |
| PoolingKernel | `MaxPool`, `AveragePool`, `LpPool`, `GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool` |
| (zero-cost) | `Reshape` (buffer alias), `Gemm` (decomposed → MatMul + Add) |

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate test models (all generators required before running tests)
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py

# Run the scheduler on a model
.venv/bin/python inference_scheduler.py test/models/mixed_ops.onnx --out-dir /tmp/out

# Run all tests (897 tests)
.venv/bin/python -m pytest test/ -v
```

Key source files:
- **`inference-scheduler/inference_scheduler.py`** — CLI entry point
- **`inference-scheduler/src/graph.py`** — ONNX loading, shape inference, Gemm preprocessing, tensor registry
- **`inference-scheduler/src/nodes.py`** — `ScheduledNode`, `MatmulNode`, `ConvNode`, `PoolNode`, `ReshapeNode`
- **`inference-scheduler/src/tensor.py`** — Weight encoding (float → ap_fixed<16,8>), buffer declarations
- **`inference-scheduler/src/codegen/`** — Multi-mixin code generator (header, source, buf_impl, test, cmake)

See `doc/INFERENCE_SCHEDULER.md` for the full technical reference.

### HLS Pragmas Used

`#pragma HLS INTERFACE m_axi offset=slave` (three data ports, one bundle each), `#pragma HLS INTERFACE s_axilite` (pointer addresses + size + return, all in `bundle=ctrl`), `#pragma HLS PIPELINE II=1` (loop body).

## Dependencies

- **`hlslib/`** (from sibling `gemm_hls`) — Provides `FindVitis.cmake`, `add_vitis_kernel`, `add_vitis_program`, and `hls_stream.h` simulation support.
- **Xilinx Vitis 2025.2** at `/mnt/data/xilinx/2025.2`. Source `settings64.sh` before building. From 2024.x, `vitis-run --tcl` replaces the older `vitis_hls -f` invocation; `FindVitis.cmake` handles this automatically via `${Vitis_HLS_TCL_FLAG}`.

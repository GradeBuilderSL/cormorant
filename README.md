# vadd_hls — AXI-Stream Vector Operation IP Core

A Vitis HLS implementation of a configurable element-wise vector accelerator for Xilinx FPGAs. The kernel reads up to two input arrays, applies a runtime-selected operation, and writes the results to a third array.

## Features

- **Six runtime-selectable operations** — Add, Sub, Mul, Div, Relu, Relu6; chosen via AXI-Lite register
- **Three AXI4 master ports** — two reads (`gmem0`/`gmem1`), one write (`gmem2`); independently banked
- **Runtime vector length** — passed via AXI-Lite `size` register; no recompilation needed
- **II=1 pipeline** — one element processed per clock cycle after burst ramp-up
- **Saturating arithmetic** — results clipped to the representable range for `ap_fixed` types
- **Configurable data type** — `float` (default), `double`, `half`, `uint8_t`, `ap_fixed<W,I>`
- **IP catalog export** — produces a `ip_catalog.zip` ready to drop into a Vivado block design
- **Inference scheduler** — Python tool that compiles ONNX models to C driver code targeting this IP

## Hardware Interface

```
              ┌──────────────────────────────────────────┐
              │  m_axi_gmem0  [AXI4, read]   ◄── a[]     │
              │  m_axi_gmem1  [AXI4, read]   ◄── b[]     │
              │  m_axi_gmem2  [AXI4, write]  ──► c[]     │
              │                                          │
              │  s_axi_ctrl   [AXI4-Lite]                │
              │    a_addr  — base address of a[]         │
              │    b_addr  — base address of b[]         │
              │    c_addr  — base address of c[]         │
              │    size    — number of elements          │
              │    op      — operation selector (0–5)    │
              │    ap_ctrl_hs — start / done / idle      │
              └──────────────────────────────────────────┘
```

HLS infers AXI4 burst reads on `gmem0` and `gmem1` and a burst write on `gmem2`. Each bundle is an independent AXI4 master port, so all three can be connected to different memory banks or slaves. For unary operations (Relu, Relu6) no AXI transactions are issued on `gmem1`.

### Operation Codes

| `op` | Name | Expression | Arity |
|------|------|------------|-------|
| 0 | `OP_ADD` | `saturate_cast(a[i] + b[i])` | binary |
| 1 | `OP_SUB` | `saturate_cast(a[i] - b[i])` | binary |
| 2 | `OP_MUL` | `saturate_cast(a[i] * b[i])` | binary |
| 3 | `OP_DIV` | `saturate_cast(a[i] / b[i])` | binary |
| 4 | `OP_RELU` | `max(a[i], 0)` | unary |
| 5 | `OP_RELU6` | `min(max(a[i], 0), 6)` | unary |

## Project Structure

```
axi_demo/
├── CMakeLists.txt              # Build system
├── include/
│   ├── Config.h.in             # CMake-generated configuration header
│   └── VectorOP.h              # Op enum, saturate_cast, kernel declaration
├── kernel/
│   └── VectorOP.cpp            # HLS kernel source (6 operations)
├── test/
│   └── TestSimulation.cpp      # C simulation test (6 ops × sizes + saturation)
├── platforms/
│   └── kv260.json              # Xilinx KV260 Starter Kit
├── scripts/
│   └── Synthesis.tcl.in        # HLS synthesis + IP export TCL template
├── doc/
│   ├── SIMULATION_ISSUES.md    # PS VIP simulation quirks and workarounds
│   └── INFERENCE_SCHEDULER.md  # Inference scheduler technical reference
└── inference-scheduler/        # ONNX → XVectoropkernel code generator
    ├── inference_scheduler.py  # CLI entry point
    ├── requirements.txt
    └── src/
        ├── graph.py            # ONNX parsing and graph resolution
        ├── nodes.py            # Op mapping and call emission
        ├── tensor.py           # Weight encoding, buffer declarations
        └── codegen.py          # C file assembly
```

## Prerequisites

- **Xilinx Vitis 2024.x or 2025.x** — for HLS synthesis and IP export
- **[hlslib](https://github.com/definelicht/hlslib)** — provides `FindVitis.cmake` and HLS simulation support. By default the build reads it from the sibling `gemm_hls` project; override with `-DVA_HLSLIB_DIR=…`
- **CMake ≥ 3.19**

## Quick Start

```bash
# 1. Source Vitis environment
source /mnt/data/xilinx/2025.2/Vitis/settings64.sh

# 2. Configure
mkdir build && cd build
cmake ../

# 3. Run the C simulation test (no hardware needed)
make TestSimulation
./TestSimulation
```

Expected output:
```
[1/8] size=1 ... PASS
[2/8] size=8 ... PASS
...
[8/8] size=4096 ... PASS

All 8 tests passed.
```

## Generating the Vivado IP

```bash
# Inside the build directory:
make synthesize_kv260
```

This runs Vitis HLS synthesis for the `xck26-sfvc784-2LV-c` part at 300 MHz and exports a Vivado IP catalog archive:

```
build/kv260/ip_catalog.zip
```

To add the IP to a Vivado project:

1. **IP Catalog → Add Repository** — point Vivado at the extracted `ip_catalog/` directory (or use the zip directly via **IP Catalog → Add Zip**)
2. Instantiate **VectorOPKernel** in a block design
3. Connect `m_axi_gmem0`, `m_axi_gmem1`, `m_axi_gmem2` to AXI4 memory interconnect (each can target a different bank)
4. Connect `s_axi_ctrl` to an AXI-Lite master (e.g. Zynq PS M\_AXI\_HPM or MicroBlaze)

## CMake Parameters

| Parameter | Default | Description |
|---|---|---|
| `VA_DATA_TYPE` | `float` | Element type: `float`, `double`, `half`, `uint8_t`, `ap_fixed<W,I>` |
| `VA_HLSLIB_DIR` | `../gemm_hls/hlslib` | Path to hlslib |
| `VA_PLATFORM` | `xilinx_u250_…` | Vitis platform for the xclbin flow |
| `VA_TARGET_CLOCK` | *(empty)* | Target clock in MHz; empty = platform default |

### Changing the data type

```bash
cmake ../ -DVA_DATA_TYPE=double
cmake ../ -DVA_DATA_TYPE="ap_fixed<16,8>"
```

The stream TDATA width adjusts automatically (64 bits for `double`, 16 bits for `ap_fixed<16,8>`).

## Adding a New Platform

Create `platforms/<name>.json`:

```json
{
  "description": "My board",
  "part": "xczu9eg-ffvb1156-2-e",
  "board": "xilinx.com:zcu102:part0:3.4",
  "clock": 250
}
```

Re-run cmake, then:

```bash
make synthesize_<name>
```

`board` and `clock` are optional (clock defaults to 300 MHz).

## Inference Scheduler

`inference-scheduler/` is a Python tool that takes an ONNX model and generates a C source file implementing the full inference loop on the KV260, using only `VectorOPKernel` invocations.

### Supported ONNX operators

`Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(min=0, max=6)` — all element-wise ops that map directly to the six kernel operation codes. Unsupported ops cause the tool to exit with an error.

### Quick start

```bash
cd inference-scheduler

# Create environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate test ONNX models (written to test/models/)
.venv/bin/python test/gen_test_models.py

# Compile an ONNX model to a C driver file
.venv/bin/python inference_scheduler.py test/models/mixed_ops.onnx -o inference.c

# Run the test suite (28 tests)
.venv/bin/python -m pytest test/test_scheduler.py -v
```

### Generated file structure

The emitted C file (`inference.c`) contains:

| Section | Description |
|---------|-------------|
| Includes | `xvectoropkernel.h`, `xil_cache.h`, `xil_types.h` |
| `typedef uint16_t Data_t` | ap_fixed<16,8> element type |
| `VECTOROP_*` defines | Op code constants matching `VectorOP.h` |
| Weight arrays | `static const uint16_t name[N]` — float weights encoded as ap_fixed<16,8> |
| Intermediate buffers | `static Data_t name[N]` — one buffer per intermediate tensor |
| `run_op()` | Configures registers, flushes/invalidates cache, starts kernel, polls done |
| `inference_init()` | Calls `XVectoropkernel_Initialize()` |
| `inference_run()` | Sequential `run_op()` calls, one per ONNX node |

See `doc/INFERENCE_SCHEDULER.md` for the full technical reference.

## Synthesis Results (KV260, float)

| Metric | Value |
|---|---|
| Target device | xck26-sfvc784-2LV-c |
| Clock target | 300 MHz |
| Estimated Fmax | 409 MHz |
| Loop II | 1 (achieved) |
| BRAM | 0 |
| DSP | 3 (floating-point adder) |

## License

This project is released under the BSD 3-Clause License.

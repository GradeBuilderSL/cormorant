# axi_demo — FPGA Neural-Network Inference Accelerator

A set of Vitis HLS kernels for Xilinx FPGAs together with a Python
code-generator that compiles ONNX models to self-contained C projects that
drive the kernels on the KV260.

---

## Hardware Kernels

| Kernel | ONNX ops | AXI masters | Key feature |
|--------|----------|-------------|-------------|
| **VectorOPKernel** | `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(0,6)` | gmem0 (a, r), gmem1 (b, r), gmem2 (c, w) | II=1, runtime op selector |
| **MatmulKernel** | `MatMul` | gmem0 (a, r), gmem1 (b, r), gmem2 (c, w) | Tiled GEMM, batched |
| **ConvKernel** | `Conv` | gmem0 (x, r), gmem1 (w, r), gmem2 (bias, r), gmem3 (y, w) | NCHW, stride/dilation/padding/bias |
| **PoolingKernel** | `MaxPool`, `AveragePool`, `LpPool` and Global variants | gmem0 (x, r), gmem1 (y, w) | NCHW, dilation, count_include_pad |

**Zero-cost transformations (no hardware call):**
- `Reshape` — output pointer aliased to source buffer; no data copy.
- `Gemm` — decomposed to `MatMul` + optional `Add` at model-load time.

All kernels use `ap_fixed<16,8>` as the default data type (configurable via
CMake) and share the same `saturate_cast` / AP_TRN+AP_SAT saturation pattern.

---

## Repository Layout

```
axi_demo/
├── kernel/          VectorOPKernel — element-wise vector ops
│   └── VectorOP.cpp
├── matmul/          MatmulKernel — tiled matrix multiply
│   └── kernel/MatmulKernel.cpp
├── conv/            ConvKernel — 2-D NCHW convolution
│   └── kernel/ConvKernel.cpp
├── pool/            PoolingKernel — 2-D NCHW pooling
│   └── kernel/PoolingKernel.cpp
│
├── include/         VectorOPKernel headers
├── platforms/       Per-board JSON configs (kv260.json, …)
├── scripts/         Synthesis.tcl.in — HLS + IP-export template
├── doc/             Technical reference documents
│
└── inference-scheduler/   ONNX → C code-generator
    ├── inference_scheduler.py
    ├── src/
    │   ├── graph.py        ONNX loading, shape inference, Gemm decomposition
    │   ├── nodes.py        ScheduledNode / MatmulNode / ConvNode / PoolNode / ReshapeNode
    │   ├── tensor.py       Weight encoding, buffer declarations
    │   ├── kernels.py      KernelDesc registry (driver prefixes, file lists)
    │   ├── dtype.py        DataType abstraction (ap_fixed<W,I>, float32)
    │   ├── layout.py       TensorLayout (alloc sizes, broadcast strides)
    │   └── codegen/        Multi-mixin code generator
    │       ├── _core.py    Pool layout, tensor sizing
    │       ├── _header.py  include/inference.h
    │       ├── _source.py  src/inference.c
    │       ├── _buf_impl.py src/inference_buf.c (XRT / bare-metal DMA)
    │       ├── _test.py    test/test_inference.c
    │       ├── _cmake.py   CMakeLists.txt
    │       └── _simulate.py Fixed-point forward simulation
    └── run_remote_tests.py  SSH-based hardware test runner
```

---

## Prerequisites

- **Xilinx Vitis 2025.2** at `/mnt/data/xilinx/2025.2` — HLS synthesis and IP export.
  Source `settings64.sh` before building.
- **[hlslib](https://github.com/definelicht/hlslib)** — provides `FindVitis.cmake`
  and HLS simulation support.  Reads from sibling `gemm_hls/` by default;
  override with `-DVA_HLSLIB_DIR=…`.
- **CMake ≥ 3.19**
- **Python ≥ 3.10** (inference-scheduler only)

---

## Building a Kernel

Each kernel is a self-contained CMake project. The workflow is identical for
all four:

```bash
# VectorOPKernel example — same steps apply to matmul/, conv/, pool/
cd axi_demo          # (or cd matmul / conv / pool)

mkdir build && cd build
cmake ../

# C simulation test (no hardware needed)
make TestSimulation
./TestSimulation

# HLS synthesis + Vivado IP export
make synthesize_kv260
# → build/kv260/ip_catalog.zip
```

### Key CMake parameters (VectorOPKernel)

| Parameter | Default | Description |
|---|---|---|
| `VA_DATA_TYPE` | `ap_fixed<16,8>` | Element type |
| `VA_HLSLIB_DIR` | `../gemm_hls/hlslib` | Path to hlslib |
| `VA_TARGET_CLOCK` | *(empty)* | Target MHz; empty = platform default (300 MHz) |

### Adding a new platform

Create `platforms/<name>.json` (required: `part`; optional: `board`, `clock`):

```json
{ "part": "xczu9eg-ffvb1156-2-e", "board": "xilinx.com:zcu102:part0:3.4", "clock": 250 }
```

Re-run cmake, then `make synthesize_<name>`.

---

## Kernel Interfaces

### VectorOPKernel

```
gmem0 (a, read), gmem1 (b, read), gmem2 (c, write)
s_axi_ctrl: a_addr, b_addr, c_addr, size, op, ap_ctrl_hs
```

| `op` | Name | Expression |
|------|------|------------|
| 0 | ADD  | `saturate_cast(a[i] + b[i])` |
| 1 | SUB  | `saturate_cast(a[i] - b[i])` |
| 2 | MUL  | `saturate_cast(a[i] * b[i])` |
| 3 | DIV  | `saturate_cast(a[i] / b[i])` |
| 4 | RELU | `max(a[i], 0)` |
| 5 | RELU6 | `min(max(a[i], 0), 6)` |

### MatmulKernel

```
gmem0 (a, read), gmem1 (b, read), gmem2 (c, write)
s_axi_ctrl: a/b/c addresses, n, k, m, batch, a/b/c_batch_stride
```

Computes `c[batch][n][m] = a[batch][n][k] × b[batch][k][m]` with tiling.
Batch strides enable strided-view broadcasting without data copies.

### ConvKernel

```
gmem0 (x, read), gmem1 (weight, read), gmem2 (bias, read), gmem3 (y, write)
s_axi_ctrl: all addresses + batch, in_ch, in_h/w, out_ch, out_h/w,
            kh, kw, stride_h/w, dilation_h/w, pad_top, pad_left, has_bias
```

NCHW 2-D convolution. `groups=1` only. Bias is optional (`has_bias=0` skips
gmem2 reads).

### PoolingKernel

```
gmem0 (x, read), gmem1 (y, write)
s_axi_ctrl: all addresses + batch, channels, in_h/w, out_h/w, pool_h/w,
            stride_h/w, pad_top, pad_left, dil_h/w,
            pool_type, lp_order, count_include_pad
```

| `pool_type` | Operation |
|-------------|-----------|
| 0 | MaxPool |
| 1 | AveragePool |
| 2 | LpPool (p = `lp_order`) |

Global variants (GlobalMaxPool, GlobalAveragePool, GlobalLpPool) are handled
by the scheduler setting `pool_h=in_h`, `pool_w=in_w`, `stride=1`, `pad=0`.

---

## Inference Scheduler

`inference-scheduler/` reads an ONNX model and emits a complete C project:

```
<out_dir>/
├── CMakeLists.txt         INFERENCE_TARGET=BARE_METAL|LINUX
├── include/inference.h    Public API: Data_t, size macros, struct inference_buf,
│                          init / run / deinit declarations
├── src/
│   ├── inference.c        Weight ROMs, run_op/run_matmul/run_conv/run_pool helpers,
│   │                      single-pool init, inference_run
│   └── inference_buf.c    DMA buffer alloc/sync (Linux XRT or bare-metal Xil)
├── test/test_inference.c  On-device smoke test: ramp fill → run → compare GT
├── scripts/check_inference_setup.sh
└── driver/                Kernel driver sources (copied from HLS output)
```

### Quick start

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate all test ONNX models (required before running tests)
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py

# Generate a C project from an ONNX model
.venv/bin/python inference_scheduler.py path/to/model.onnx --out-dir /tmp/out \
    --driver-dir conv/build/kv260/conv_kv260/solution1/impl/misc/drivers/ConvKernel_v1_0/src

# Run the full test suite (930 tests)
.venv/bin/python -m pytest test/ -v
```

### Generated C API

```c
// include/inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES    16u
#define INFERENCE_BUF_POOL_SIZE_BYTES  N

struct inference_buf { … };          // defined here so inference.c can declare
typedef struct inference_buf inference_buf_t;  // static view instances

// One init parameter per active kernel (only present kernels appear):
int  inference_init(const char *vectoropkernel_instance
                    [, const char *matmulkernel_instance]
                    [, const char *convkernel_instance]
                    [, const char *poolkernel_instance]);

// All graph inputs then all graph outputs:
void inference_run(inference_buf_t *<input…>, inference_buf_t *<output…>);
void inference_deinit(void);

// inference_buf.c
inference_buf_t *inference_buf_alloc(unsigned n_elem);
void             inference_buf_free(inference_buf_t *buf);
Data_t          *inference_buf_ptr(inference_buf_t *buf);
uint64_t         inference_buf_phys(const inference_buf_t *buf);
void             inference_buf_sync_to_device(inference_buf_t *buf);
void             inference_buf_sync_from_device(inference_buf_t *buf);
```

### DMA buffer model

`inference_init()` makes **one** `inference_buf_alloc()` call that covers all
weights and intermediate buffers.  Each tensor gets a `inference_buf_init_view()`
slice at a 64-byte-aligned offset.  Weights are `memcpy`'d from ROM arrays and
flushed to DDR in one shot.  `inference_run()` flushes graph inputs at the top
and invalidates graph outputs at the bottom; intermediate buffers are never
synced (the PL kernels access DDR directly via their AXI master ports).

### Hardware test runner

```bash
# Copy and edit the appropriate template config
cp remote_config_pool.json remote_config.json   # or _conv / _matmul / _vectorop

# Run all models against the KV260 over SSH
.venv/bin/python run_remote_tests.py remote_config.json
```

The runner generates each project locally, uploads it, builds on the board
with `cmake + make`, executes `test_inference`, and reports PASS/FAIL per model.
Driver directory paths in `local.driver_dirs` are validated before upload —
a missing path fails fast with a clear error rather than a cryptic CMake error
on the board.

---

## Data Type

Default: `ap_fixed<16,8>` — 16-bit two's complement, 8 integer bits, 8
fractional bits.  Encoding: `1.0 = 0x0100`, `0.5 = 0x0080`, range `[−128, 127.996]`.

The `DataType` abstraction in `inference-scheduler/src/dtype.py` allows
`float32` and other types to be used without changing any other source file.

---

## Documentation

| Document | Description |
|----------|-------------|
| `doc/INFERENCE_SCHEDULER.md` | Full inference scheduler technical reference |
| `doc/ARCHITECTURE.md` | Codegen internals — node classes, layout engine, mixin assembly |
| `inference-scheduler/CLAUDE.md` | Developer quick-reference (kernels, node classes, test matrix) |
| `doc/SIMULATION_ISSUES.md` | PS VIP simulation quirks and workarounds |

---

## License

BSD 3-Clause License.

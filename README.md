# Cormorant — FPGA Neural-Network Inference Accelerator

<p align="center">
  <img src="doc/images/cormorant.png" alt="Cormorant" width="320"/>
</p>

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

## Quick Start

### 1. Clone and initialise

```bash
git clone <repo-url> cormorant && cd cormorant

# Initialise the Vivado block-design submodule (needed for hardware builds)
git submodule update --init
```

### 2. Run HLS simulation tests (no hardware needed)

```bash
mkdir build && cd build
cmake ../
make TestSimulation   # VectorOPKernel
make TestConvRef      # ConvKernel
make TestMatmulRef    # MatmulKernel
make TestPoolingSim   # PoolingKernel
ctest                 # run all four
```

### 3. Compile an ONNX model to a C inference project

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate all test models and run the Python test suite (1006 tests)
.venv/bin/python test/gen_all_models.py
.venv/bin/python -m pytest test/ -q

# Compile your own model
.venv/bin/python inference_scheduler.py mymodel.onnx --out-dir /tmp/mymodel_out
```

### 4. Synthesise HLS kernels for KV260

```bash
# Source Vitis (required for HLS synthesis)
source <Xilinx install dir>/settings64.sh

cd build

# Synthesise all four kernels + build the device-tree overlay
make synthesize_kv260
make dtbo_kv260_cormorant
# overlay output: build/dts/kv260/design_cormorant.dtbo
```

### 5. Build the Vivado block design (bitstream)

Requires the `hw/cormorant_hw_128` submodule and Vivado 2025.2.

```bash
# Synthesise, implement, and write bitstream (depends on synthesize_kv260)
make build_hw_kv260
# bitstream: hw/cormorant_hw_128/cormorant_hw_128.runs/impl_1/design_cormorant_wrapper.bit

# Run behavioral simulation against the testbench
make sim_hw_kv260
```

### 6. Deploy and run on the KV260

```bash
cd inference-scheduler

# Load the bitstream + device-tree overlay from the host over SSH
cp bitstream_config_kv260.json.example bitstream_config_kv260.json
# (edit bitstream_config_kv260.json first: set ssh.host and check paths)
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json

# Confirm UIO devices are visible (on the board)
cat /sys/class/uio/uio*/name
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool

# Run correctness tests over SSH (see inference-scheduler/doc/REMOTE_TESTING.md)
cp remote_config.json.example remote_config.json
$EDITOR remote_config.json   # set ssh.host and local.driver_dirs
.venv/bin/python run_remote_tests.py --config remote_config.json
```

### 7. Run an end-to-end demo

Three demos under `demo/` take an ONNX model all the way to a running KV260
inference project. Each follows the same **download → generate → deploy**
flow behind a one-shot `run_demo.py`:

| Demo | Model | Input | Output |
|------|-------|-------|--------|
| [`demo/mnist/`](demo/mnist/) | MNIST convnet + LeNet | 10 000 MNIST test images | top-1 accuracy + per-image latency |
| [`demo/image_classification/`](demo/image_classification/) | MobileNetV1 1.0/224 | static JPG/PNG files | top-5 ImageNet predictions |
| [`demo/camera/`](demo/camera/) | MobileNetV1 1.0/224 | live RealSense camera | real-time annotated frames streamed back over SSH |

```bash
cd demo/<name>
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

cp <name>_config.json.example <name>_config.json
$EDITOR <name>_config.json    # set ssh.host, key_file, uio_devices

.venv/bin/python run_demo.py
```

See **[demo/README.md](demo/README.md)** for the demo overview and each
demo's own `README.md` for full details.

---

## Dependencies

### Host (development machine)

| Dependency | Min version | Purpose | Install |
|------------|-------------|---------|---------|
| **GCC / G++** | 9 | C simulation tests (`ctest`) | `sudo apt install build-essential` |
| **CMake** | 3.19 | Build system | `sudo apt install cmake` |
| **Python** | 3.10 | Inference scheduler, tests | `sudo apt install python3 python3-venv` |
| **Xilinx Vitis** | 2025.2 | HLS synthesis, IP export | Xilinx installer; set `XILINX_VITIS` |
| **Vivado** | 2025.2 | Block design, bitstream | Included with Vitis |
| **dtc** | any | Device-tree overlay build | `sudo apt install device-tree-compiler` |

Python packages (installed into `.venv` via `pip install -r requirements.txt`):

| Package | Min version | Purpose |
|---------|-------------|---------|
| `onnx` | 1.14 | Model parsing and shape inference |
| `numpy` | 1.24 | Weight encoding, simulation |
| `paramiko` | 3.0 | SSH/SFTP for remote test runner |
| `onnxsim` | 0.4 | `simplify_onnx.py` — constant folding, shape pinning |
| `onnxoptimizer` | 0.3 | `simplify_onnx.py` — `fuse_bn_into_conv` safety-net pass |
| `onnxruntime` | 1.14 | `simplify_onnx.py --check` smoke test |

### Target board (KV260)

| Dependency | Min version | Purpose | Install |
|------------|-------------|---------|---------|
| **GCC** | 11 | On-board build | pre-installed on KV260 Ubuntu |
| **CMake** | 3.19 | On-board build | `sudo apt install cmake` |
| **XRT runtime** | 2.13 | DMA buffer management, UIO | `sudo apt install xrt` or from Xilinx |

---

## Building the Kernels

All four kernels share a single top-level CMake project.

```bash
cd cormorant
mkdir build && cd build
cmake ../

# C simulation tests (no hardware needed)
make TestSimulation    # VectorOPKernel
make TestConvRef       # ConvKernel
make TestMatmulRef     # MatmulKernel
make TestPoolingSim    # PoolingKernel
ctest

# HLS synthesis + Vivado IP export (requires Vitis)
make synthesize_vectorop_kv260
make synthesize_conv_kv260
make synthesize_matmul_kv260
make synthesize_pool_kv260

# All four at once
make synthesize_kv260

# Device tree overlay (requires dtc)
make dtbo_kv260_cormorant
# output: build/dts/kv260/design_cormorant.dtbo
```

Each kernel can also be built standalone:

```bash
cd cormorant/kernels/vectorop
mkdir build && cd build
cmake ../
make TestSimulation
make synthesize_vectorop_kv260
```

### Key CMake parameters

| Parameter | Default | Description |
|---|---|---|
| `AXI_BUS_WIDTH` | `32` | AXI master bus width in bits (32, 64, 128, 256, 512); must match the Vivado block design |
| `VA_DATA_TYPE` | `ap_fixed<16,8>` | Element type (VectorOPKernel) |
| `VA_TARGET_CLOCK` | *(empty)* | Target MHz; empty = platform default (300 MHz) |
| `VA_ENABLE_VITIS_FLOW` | `OFF` | Enable Vitis hw/hw_emu xclbin targets (requires installed platform) |

### Adding a new platform

Drop a `platforms/<platform>.json` file next to `kv260.json` and re-run
CMake. The JSON is the single source of truth for the FPGA part /
board / clock and the compile-time bounds each kernel synthesises
against (the `kernels.{conv,matmul,pool}` block); both the C++ build
and the Python scheduler validators read it. A missing field is a
`FATAL_ERROR` during CMake configure.

```bash
cmake .. -DAXI_PLATFORM=<platform>    # makes <platform> the C-sim default
make synthesize_<platform>             # synthesise all four kernels
```

The full schema — top-level fields, the three per-kernel constant
tables, constraint formulas, and the post-edit workflow (including
regenerating the inference-scheduler hardware-bound test fixtures
after a `max_*` change) — lives in
**[doc/PLATFORM_CONFIGURATION.md](doc/PLATFORM_CONFIGURATION.md)**.

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
├── driver/                Kernel driver sources (copied from HLS output)
└── report.md              Human-readable summary: model metadata, parameters,
                           applied transformations, per-layer table, and (for
                           fixed-point dtypes) per-tensor / per-layer
                           quantization error
```

### Quick start

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate all test ONNX models with a single command
.venv/bin/python test/gen_all_models.py

# Run the full Python test suite (1006 tests, no hardware needed)
.venv/bin/python -m pytest test/ -q

# Generate a C project from an ONNX model
.venv/bin/python inference_scheduler.py path/to/model.onnx --out-dir /tmp/out \
    --driver-dir build/kernels/vectorop/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src
```

### Report

Every run also produces `report.md` in the output directory — a
self-contained, human-readable summary covering the input model
(name, SHA-256, dtype, lanes used), parameter count and storage
breakdown, activation pool slot reuse, applied transformations
(Gemm decomposition, Reshape folding, buffer reuse, cross-lane
parallelism), and a full per-layer table. For fixed-point dtypes it
adds per-tensor weight-quantization and per-layer activation-truncation
metrics: `Max |abs|`, `NRMSE` (normalised RMS error), and
`SQNR (dB)` — the standard signal-to-quantisation-noise figure.

```bash
.venv/bin/python inference_scheduler.py model.onnx --out-dir /tmp/out
# → /tmp/out/report.md  (open in any markdown viewer)

# Suppress with --no-report when scripting throwaway builds.
```

See [`inference-scheduler/doc/INFERENCE_SCHEDULER.md §11`](inference-scheduler/doc/INFERENCE_SCHEDULER.md#11-generated-report-reportmd)
for the full layout and the quantization-metric definitions.

### Generated C API

```c
// include/inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES    16u
#define INFERENCE_BUF_POOL_SIZE_BYTES  N

struct inference_buf { … };
typedef struct inference_buf inference_buf_t;

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
weights and intermediate buffers.  Each tensor gets an `inference_buf_init_view()`
slice at a 64-byte-aligned offset.  Intermediate tensors with non-overlapping
execution lifetimes share the same slot, significantly reducing pool size
(see **[inference-scheduler/doc/BUFFER_REUSE.md](inference-scheduler/doc/BUFFER_REUSE.md)**).
Weights are `memcpy`'d from ROM arrays and flushed to DDR once during init.
`inference_run()` flushes graph inputs at the top and invalidates graph outputs
at the bottom; intermediate buffers are never synced.

---

## Supported Models

The scheduler has been validated end-to-end (compile → on-device run →
output check against Python ground truth) against the ONNX models below.
Each link is the pre-prepared file — already passed through
[`simplify_onnx.py`](inference-scheduler/simplify_onnx.py) so it loads
into `inference_scheduler.py` without further surgery. For preparing
your own models see
[`inference-scheduler/doc/MODEL_PREPARATION.md`](inference-scheduler/doc/MODEL_PREPARATION.md).

| Model | Input shape (NCHW) | Task |
|-------|--------------------|------|
| [ConvMNIST](https://drive.google.com/file/d/1a-A-t2JBC9r5IaEjpWp9915n0wBoIj0y) | `1×1×28×28` | MNIST digit classifier (small convnet) |
| [LeNet](https://drive.google.com/file/d/1tNQe_wDvVzuPnEMJvIzeULMZrUlYS7VF) | `1×1×28×28` | Classic LeNet-5 on MNIST |
| [MobileNet V1](https://drive.google.com/file/d/1PzFSPkXpkIpiKfyo8tORl2AkfXgjdQvw) | `1×3×224×224` | ImageNet 1000-class classifier (depthwise-separable) |
| [MobileNet V2](https://drive.google.com/file/d/1ti97y2P_Fc8TRUk0oVm_AG7yrmk5Fuw1) | `1×3×224×224` | ImageNet 1000-class classifier (inverted residuals) |
| [ResNet-18](https://drive.google.com/file/d/1DKyALYam5jAzMSK8ulgFQuSbQ62-EVvr) | `1×3×224×224` | ImageNet 1000-class classifier (residual blocks) |

---

## Testing

Five layers of testing, each independent — full details in
**[doc/TESTING.md](doc/TESTING.md)**:

| Layer | Needs | One-liner |
|-------|-------|-----------|
| Python unit tests | nothing | `.venv/bin/python -m pytest test/ -q` (1006 tests) |
| HLS C-sim | gcc, CMake | `make TestSimulation TestConvRef TestMatmulRef TestPoolingSim && ctest` |
| Vivado behavioural sim | Vitis, Vivado | `make sim_hw_kv260` (needs `synthesize_kv260` first) |
| On-device correctness | KV260 over SSH, bitstream loaded | `run_remote_tests.py --config remote_config.json` |
| On-device performance | KV260 over SSH, bitstream loaded | `run_remote_perf.py --config perf_config.json` |

The first three run on the host. The last two need the KV260 with the
Cormorant bitstream loaded (Quick Start step 6, or
[`inference-scheduler/doc/REMOTE_TESTING.md`](inference-scheduler/doc/REMOTE_TESTING.md)).

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
| [`doc/TESTING.md`](doc/TESTING.md) | Full testing reference — Python unit tests, HLS C-sim, Vivado sim, on-device correctness + perf |
| [`inference-scheduler/doc/INFERENCE_SCHEDULER.md`](inference-scheduler/doc/INFERENCE_SCHEDULER.md) | Full inference scheduler technical reference |
| [`inference-scheduler/doc/REMOTE_TESTING.md`](inference-scheduler/doc/REMOTE_TESTING.md) | SSH remote testing and performance benchmarking |
| [`inference-scheduler/doc/BUFFER_REUSE.md`](inference-scheduler/doc/BUFFER_REUSE.md) | Live-interval buffer reuse optimisation |
| [`demo/README.md`](demo/README.md) | End-to-end KV260 demos overview (mnist, image_classification, camera) |
| [`inference-scheduler/doc/ARCHITECTURE.md`](inference-scheduler/doc/ARCHITECTURE.md) | Codegen internals — node classes, layout engine, mixin assembly |
| [`doc/PLATFORM_CONFIGURATION.md`](doc/PLATFORM_CONFIGURATION.md) | Platform JSON schema, per-kernel constants, adding/editing a platform |
| [`doc/PROFILER.md`](doc/PROFILER.md) | Per-layer wall-clock + DDR-bandwidth profiling runtime (`inference_prof` + `inference_ddr`) |
| [`doc/CONV_KERNEL.md`](doc/CONV_KERNEL.md) | ConvKernel architecture and tiling details |
| [`doc/POOLING_KERNEL.md`](doc/POOLING_KERNEL.md) | PoolingKernel architecture |
| [`doc/MATMUL_KERNEL.md`](doc/MATMUL_KERNEL.md) | MatmulKernel architecture and tiling details |
| [`doc/VECTOROP_KERNEL.md`](doc/VECTOROP_KERNEL.md) | VectorOPKernel architecture (element-wise ops) |
| [`doc/SIMULATION_ISSUES.md`](doc/SIMULATION_ISSUES.md) | PS VIP simulation quirks and workarounds |

---

## Funding

[![dAIEDGE Project](https://img.shields.io/badge/dAIEDGE-Project-6A5ACD?style=for-the-badge)](https://daiedge.eu/)
[![EU Horizon Europe](https://img.shields.io/badge/Funded%20by-EU%20Horizon%20Europe-003399?style=for-the-badge&logo=europeanunion&logoColor=white)](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)

This work was supported by the **[dAIEDGE Open Call Programme](https://daiedge.eu/)**, funded by the **[European Union's Horizon Europe research and innovation programme](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)** under project number **#101120726**.

---

## License

Copyright 2026 GradeBuilder SL. Licensed under the
[Apache License, Version 2.0](LICENSE).

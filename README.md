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
# (edit bitstream_config_kv260.json first: set ssh.host and check paths)
.venv/bin/python upload_bitstream.py --config bitstream_config_kv260.json

# Confirm UIO devices are visible (on the board)
cat /sys/class/uio/uio*/name
# fabric_vecop
# fabric_matmul
# fabric_conv
# fabric_pool

# Run correctness tests over SSH (see inference-scheduler/doc/REMOTE_TESTING.md)
cp remote_config_all_models.json remote_config.json
$EDITOR remote_config.json   # set ssh.host and local.driver_dirs
.venv/bin/python run_remote_tests.py --config remote_config.json
```

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

Create `platforms/<platform>.json`:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `part` | yes | — | Xilinx device part string |
| `board` | no | *(none)* | Board identifier passed to `set_part -board` |
| `clock` | no | `300` | Target clock in MHz |
| `axi_bus_width` | no | `AXI_BUS_WIDTH` CMake variable | AXI master bus width in bits |

```json
{
  "part": "xczu9eg-ffvb1156-2-e",
  "board": "xilinx.com:zcu102:part0:3.4",
  "clock": 250,
  "axi_bus_width": 64
}
```

Re-run cmake. The new platform gets synthesis targets for all four kernels
(`synthesize_<kernel>_<platform>`, `synthesize_<platform>`) and a device tree
overlay target for any `.dts` file placed under `dts/<platform>/`.

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

# Generate all test ONNX models with a single command
.venv/bin/python test/gen_all_models.py

# Run the full Python test suite (1006 tests, no hardware needed)
.venv/bin/python -m pytest test/ -q

# Generate a C project from an ONNX model
.venv/bin/python inference_scheduler.py path/to/model.onnx --out-dir /tmp/out \
    --driver-dir build/kernels/vectorop/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src
```

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

## Testing

### Python unit tests (no hardware)

The full test suite runs entirely on the host — no FPGA needed:

```bash
cd inference-scheduler

# Generate all test models first (one-time step)
.venv/bin/python test/gen_all_models.py

# Run all 1006 tests
.venv/bin/python -m pytest test/ -q

# Run a specific module
.venv/bin/python -m pytest test/test_pool_alloc.py -v
```

### Hardware simulation (Vivado, no board)

Behavioral simulation of the full block design against the SystemVerilog testbench:

```bash
# From the build directory (requires synthesize_kv260 to have run first)
make sim_hw_kv260
```

Expected testbench output:
```
##  VectorOPKernel     12 /  12  (0 failed)
##  ConvKernel         18 /  18  (0 failed)
##  MatmulKernel        8 /   8  (0 failed)
##  PoolingKernel      10 /  10  (0 failed)
##  TOTAL: 48 / 48 passed  —  ALL TESTS PASSED
```

### On-device hardware tests (KV260)

End-to-end correctness testing over SSH. The runner generates a C project per
model, uploads it, builds on the board, executes the test binary, and compares
every output element against Python-simulated ground truth.

**Prerequisite:** the Cormorant bitstream must be loaded on the board before
running any hardware tests.  Use `upload_bitstream.py` (see Quick Start step 6
or `inference-scheduler/doc/REMOTE_TESTING.md` → *Bitstream Upload*).

Each example config in `inference-scheduler/` targets a specific set of kernels.
Copy the one that matches your loaded bitstream, then fill in your board details:

```bash
cd inference-scheduler

# Copy and adapt the example that matches your loaded bitstream
cp remote_config_all_models.json remote_config.json
$EDITOR remote_config.json
```

The two fields you must set are:

- **`ssh.host`** — IP address or hostname of the KV260 (`"192.168.100.8"` by default)
- **`local.driver_dirs`** — paths on your host machine to the Vitis HLS-generated
  driver sources for each kernel, e.g.:
  ```
  "VectorOPKernel": "<cormorant_base>/build/kernels/vectorop/kv260/<target>/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src"
  ```

The `remote.uio_devices` map must list the UIO sysfs name for every kernel in
the config. After loading the `design_cormorant.dtbo` overlay all four names are
`fabric_vecop`, `fabric_matmul`, `fabric_conv`, and `fabric_pool` — the example
configs already have these set correctly for the full-cormorant bitstream.

```bash
# Verify board prerequisites before running
.venv/bin/python run_remote_tests.py --config remote_config.json --check-only

# Run all models in the config
.venv/bin/python run_remote_tests.py --config remote_config.json

# Run a specific model
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/single_add.onnx
```

For full SSH setup, config reference, and debugging guide see
**[inference-scheduler/doc/REMOTE_TESTING.md](inference-scheduler/doc/REMOTE_TESTING.md)**.

### Performance benchmarking (KV260)

`run_remote_perf.py` measures raw kernel throughput and latency. Unlike the
correctness runner it does not check output values — it only times how fast
each kernel runs for a configurable set of parameter cases.

The script uploads a single self-contained C benchmark project, builds all
four kernel binaries in one pass, then runs each case and reports results.
Kernels whose driver files are absent are silently skipped, so you can
benchmark only what is currently deployed.

Copy the example config and fill in your board details before the first run:

```bash
cd inference-scheduler
cp perf_config.json.example perf_config.json
$EDITOR perf_config.json   # set ssh.host and local.driver_dirs
```

**Config file:** `perf_config.json` — extends the same SSH schema as the
correctness configs with an additional `benchmarks` section.

```json
"benchmarks": {
  "VectorOPKernel": { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "MatmulKernel":   { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "ConvKernel":     { "enabled": false, "warmup": 10, "cases": [ ... ] },
  "PoolingKernel":  { "enabled": true,  "warmup": 10, "cases": [ ... ] }
}
```

The default `perf_config.json` ships with **49 benchmark cases** across the
four kernels (20 VectorOPKernel, 12 MatmulKernel, 9 ConvKernel, 8 PoolingKernel).

```bash
cd inference-scheduler

# Edit local.driver_dirs paths, then:

# Full benchmark run (all enabled kernels)
.venv/bin/python run_remote_perf.py --config perf_config.json

# Specific kernels only
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --kernels VectorOPKernel MatmulKernel

# Override iteration and warmup counts for a quick spot-check
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --iters 20 --warmup 5

# Preflight check — verify board is ready without running benchmarks
.venv/bin/python run_remote_perf.py --config perf_config.json --check-only
```

**Sample output:**

```
  VectorOPKernel
  ───────────────────────────────────────────────────────────────────────────────────
  Label                    Parameters                        Lat(ms)      GB/s
  ───────────────────────────────────────────────────────────────────────────────────
  ADD-1K                   ADD    size=1024    outer=1        0.0207     0.297
  ADD-4K                   ADD    size=4096    outer=1        0.0516     0.476
  ADD-16K                  ADD    size=16384   outer=1        0.1754     0.561
  ADD-64K                  ADD    size=65536   outer=1        0.6709     0.586
  ADD-256K                 ADD    size=262144  outer=1        2.6528     0.593
  ...
  ADD-bcast-8x16K          ADD    size=16384   outer=8        1.3813     0.569
  RELU-bcast-8x16K         RELU   size=16384   outer=8        1.3484     0.389
  MUL-bcast-dw-12544x16    MUL    size=16      outer=12544    5.8274     0.207
  ───────────────────────────────────────────────────────────────────────────────────
                                                 peak GB/s                0.593
                                               min latency     0.0181
  20/20 OK

  MatmulKernel
  ─────────────────────────────────────────────────────────────────────────────
  Label              Parameters                        Lat(ms)    GOps/s
  ─────────────────────────────────────────────────────────────────────────────
  8x8x8              N=8    K=8    M=8    batch=1       0.0176     0.058
  32x32x32           N=32   K=32   M=32   batch=1       0.3039     0.216
  256x256x256        N=256  K=256  M=256  batch=1     126.2730     0.266
  dw-12544x16x1      N=12544 K=16   M=1    batch=1     13.3935     0.030
  ─────────────────────────────────────────────────────────────────────────────
                                         peak GOps/s                0.266
                                         min latency     0.0176
  12/12 OK

  ConvKernel
  ─────────────────────────────────────────────────────────────────────────────────
  Label                  Parameters                        Lat(ms)    GOps/s
  ─────────────────────────────────────────────────────────────────────────────────
  3x3-1ch-28x28-32out    1ch 28x28→32ch 3x3k               14.5811     0.031
  ─────────────────────────────────────────────────────────────────────────────────
                                             peak GOps/s                0.031
                                             min latency    14.5811
  1/1 OK

  PoolingKernel
  ─────────────────────────────────────────────────────────────────────────────────
  Label                  Parameters                        Lat(ms)      GB/s
  ─────────────────────────────────────────────────────────────────────────────────
  MaxPool-2x2-56x56      MaxPool 2x2 64ch 56x56            21.6922     0.023
  MaxPool-2x2-28x28      MaxPool 2x2 64ch 28x28             5.3711     0.023
  GlobalAvgPool-7x7      AvgPool 7x7 64ch 7x7               0.1908     0.034
  ─────────────────────────────────────────────────────────────────────────────────
                                               peak GB/s                0.036
                                             min latency     0.1908
  8/8 OK

  ── OVERALL: All 41 cases passed ──
```

| Metric | Meaning |
|--------|---------|
| `Lat(ms)` | Mean kernel wall-clock time per call (after warmup) |
| `GB/s` | Memory bandwidth — VectorOPKernel and PoolingKernel |
| `GOps/s` | Arithmetic throughput — MatmulKernel and ConvKernel |
| `peak` | Best metric across all passing cases in the group |

For the full config reference, per-kernel case field definitions, and
debugging guide see the **Performance Benchmarking** section of
**[inference-scheduler/doc/REMOTE_TESTING.md](inference-scheduler/doc/REMOTE_TESTING.md)**.

### UIO device names

After loading the `design_cormorant.dtbo` overlay the four kernels appear as:

| Kernel | UIO sysfs name |
|--------|----------------|
| `VectorOPKernel` | `fabric_vecop` |
| `MatmulKernel` | `fabric_matmul` |
| `ConvKernel` | `fabric_conv` |
| `PoolingKernel` | `fabric_pool` |

Verify on the board: `cat /sys/class/uio/uio*/name`

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
| `inference-scheduler/doc/INFERENCE_SCHEDULER.md` | Full inference scheduler technical reference |
| `inference-scheduler/doc/REMOTE_TESTING.md` | SSH remote testing and performance benchmarking |
| `inference-scheduler/doc/BUFFER_REUSE.md` | Live-interval buffer reuse optimisation |
| `doc/ARCHITECTURE.md` | Codegen internals — node classes, layout engine, mixin assembly |
| `doc/CONV_KERNEL.md` | ConvKernel architecture and tiling details |
| `doc/POOLING_KERNEL.md` | PoolingKernel architecture |
| `doc/SIMULATION_ISSUES.md` | PS VIP simulation quirks and workarounds |

---

## Funding

[![dAIEDGE Project](https://img.shields.io/badge/dAIEDGE-Project-6A5ACD?style=for-the-badge)](https://daiedge.eu/)
[![EU Horizon Europe](https://img.shields.io/badge/Funded%20by-EU%20Horizon%20Europe-003399?style=for-the-badge&logo=europeanunion&logoColor=white)](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)

This work was supported by the **[dAIEDGE Open Call Programme](https://daiedge.eu/)**, funded by the **[European Union's Horizon Europe research and innovation programme](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)** under project number **#101120726**.

---

## License

Copyright 2026 GradeBuilder SL. Licensed under the
[Apache License, Version 2.0](LICENSE).

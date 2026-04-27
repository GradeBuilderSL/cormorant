# Remote Hardware Testing

`run_remote_tests.py` automates end-to-end hardware validation of ONNX models on
a physical KV260 board. For each model it:

1. Generates a C inference project locally (`inference_scheduler.py`)
2. Uploads the project to the board via SSH/SFTP
3. Builds on the board (`cmake -DINFERENCE_TARGET=LINUX` + `make`)
4. Runs the generated `test_inference` binary as root
5. Collects pass/fail results and prints a summary

The test binary fills inputs with a deterministic ramp pattern, runs the FPGA
kernel, and compares every output element against gold-standard values
pre-computed by the Python fixed-point simulator. A mismatch in any element
fails the test.

---

## Prerequisites

### Local machine

- Python virtual environment with `paramiko`:
  ```bash
  cd inference-scheduler
  .venv/bin/pip install paramiko
  ```
- Driver sources from Vitis HLS synthesis (for `local.driver_dir` in the config):
  ```
  <axi_demo>/build/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src/
  ```

### Remote board (KV260)

| Requirement | How to verify |
|-------------|---------------|
| `gcc` ≥ 11, `cmake` ≥ 3.19, `make` | `gcc --version`, `cmake --version` |
| XRT runtime (`xrt.h` + `libxrt_core.so`) | `pkg-config --exists xrt && echo ok` |
| Kernel overlay(s) loaded | `cat /sys/class/uio/uio*/name` — must print the names in `remote.uio_devices` |
| Passwordless sudo, or SSH as root | `sudo -n true && echo ok` |

The `--check-only` flag verifies all of these remotely before running any tests
(see [Preflight Check](#preflight-check)).

---

## Setup

### 1. SSH key authentication (recommended)

```bash
# Generate a dedicated key if you don't have one
ssh-keygen -t ed25519 -f ~/.ssh/kv260-testkey -N ""

# Copy it to the board
ssh-copy-id -i ~/.ssh/kv260-testkey root@192.168.100.8
```

Password authentication is also supported; set `ssh.password` in the config and
leave `ssh.key_file` null.

### 2. Config file

Copy the appropriate example and fill in your board details:

```bash
# For run_remote_tests.py (correctness tests)
cp remote_config.json.example remote_config.json
$EDITOR remote_config.json

# For run_remote_perf.py (performance benchmarks)
cp perf_config.json.example perf_config.json
$EDITOR perf_config.json
```

Minimum required change in both: set `ssh.host` to your board's IP or hostname.

---

## Config File Reference

| Key | Default | Description |
|-----|---------|-------------|
| `ssh.host` | *(required)* | Board IP address or hostname |
| `ssh.user` | `"root"` | SSH login user |
| `ssh.port` | `22` | SSH port |
| `ssh.key_file` | `null` | Path to SSH private key; `null` to use password |
| `ssh.password` | `null` | SSH password; `null` when using key auth |
| `ssh.connect_timeout` | `15` | TCP connect timeout in seconds |
| `remote.work_dir` | `"/tmp/inference_hw_tests"` | Base directory on the board; created automatically, cleaned up after the run |
| `remote.uio_devices` | `{}` | Per-kernel UIO sysfs names — `{"KernelName": "sysfs_name"}`. The string in `/sys/class/uio/uio*/name`, **not** a `/dev/uioN` path. Empty dict uses defaults from the test binary headers. |
| `remote.uio_device` | `null` | **Deprecated.** Old single-string UIO name; treated as `{"VectorOPKernel": value}`. Use `uio_devices` instead. |
| `remote.driver_dir` | `null` | Path **on the board** to copy all kernel driver sources from |
| `remote.driver_dirs` | `{}` | Per-kernel paths **on the board**: `{"VectorOPKernel": "/path", ...}` |
| `remote.cmake_args` | `[]` | Extra `-D` flags appended to the `cmake` invocation |
| `local.driver_dir` | `null` | Single local directory with all driver files to bundle into each project before upload |
| `local.driver_dirs` | `{}` | Per-kernel local paths; files are merged before upload: `{"VectorOPKernel": "/path1", "MatmulKernel": "/path2"}` |
| `build.jobs` | `4` | Parallel `make -j` jobs on the board |
| `build.timeout` | `180` | Combined cmake + make timeout in seconds |
| `run.timeout` | `120` | Per-model `test_inference` execution timeout in seconds |
| `run.use_sudo` | `true` | Prefix the test binary with `sudo -n`; requires passwordless sudo |
| `cleanup` | `true` | Remove `remote.work_dir` after all tests finish |
| `models` | `[]` | List of model paths relative to the `inference-scheduler` directory |

### UIO device names

Each hardware kernel has a UIO sysfs name set by the device tree overlay (DTSI).
For the `design_matmul_vecop` bitstream the names are:

| Kernel | UIO sysfs name |
|--------|---------------|
| `VectorOPKernel` | `VectorOPKernel_0` |
| `MatmulKernel` | `MatmulKernel_0` |

Verify after loading the DTBO:
```bash
cat /sys/class/uio/uio*/name
# VectorOPKernel_0
# MatmulKernel_0
```

These names are passed to cmake as `-DINFERENCE_VECTOROPKERNEL_INSTANCE` and
`-DINFERENCE_MATMULKERNEL_INSTANCE` compile definitions, which override the
matching `#ifndef` macros in `inference.h`.

### Driver source resolution

The runner places kernel driver sources (`.c`/`.h` files) in the project's
`driver/` directory before uploading. Four options, evaluated in priority order:

1. **`local.driver_dirs`** — per-kernel local paths; the runner merges them into
   one directory and passes it to `inference_scheduler.py --driver-dir`. Best for
   multi-kernel models where each kernel's HLS output lives separately.
2. **`local.driver_dir`** — single local directory with all driver files. Use for
   single-kernel models or when you have pre-merged the files.
3. **`remote.driver_dirs`** — per-kernel paths **on the board**; the runner copies
   each kernel's files via SSH after upload.
4. **`remote.driver_dir`** — single board-side directory for all driver files.

When `local.*` is set, the drivers are bundled before upload and `remote.*`
driver options are skipped. When none is set, `driver/` contains only a
`README.md` and the build will fail until you populate it manually.

---

## Running Tests

### Full test run

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json
```

### Preflight check (no tests)

Verify SSH connectivity and all board prerequisites without running any tests:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --check-only
```

Example output (VectorOPKernel-only model):

```
Connecting to root@192.168.100.8:22 …
  Connected

Remote prerequisites
    OK      cmake                        cmake version 3.22.1
    OK      make                         GNU Make 4.3
    OK      gcc                          gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    OK      xrt headers                  xrt via pkg-config
    OK      sudo / root access           passwordless sudo OK
    OK      uio (VectorOPKernel: VectorOPKernel_0)  /dev/uio4
```

For a mixed-kernel model with both kernels configured:

```
    OK      uio (VectorOPKernel: VectorOPKernel_0)  /dev/uio4
    OK      uio (MatmulKernel: MatmulKernel_0)      /dev/uio5
```

### Subset of models

Override the model list on the command line (paths relative to `inference-scheduler/`):

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/single_add.onnx test/models/mixed_ops.onnx
```

### Verbose output

Show full cmake/make and test binary output for every model, not just failures:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --verbose
```

### Keep remote directories (debugging)

Preserve remote build directories after the run so you can SSH in and inspect:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --no-cleanup
# then: ssh root@192.168.100.8
#       ls /tmp/inference_hw_tests/
#       cat /tmp/inference_hw_tests/single_add/build/CMakeFiles/CMakeError.log
```

### Stop on first failure

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --fail-fast
```

---

## Reading the Summary

```
  Model                   Status                 Time  Steps
  ──────────────────────────────────────────────────────────
  single_add              PASSED                  3.4s  generate  upload  build  run
  mixed_ops               PASSED                  3.3s  generate  upload  build  run
  sat_div_pos             BUILD_ERROR             2.4s  generate  upload  build
```

| Status | Meaning |
|--------|---------|
| `PASSED` | All four steps completed; `test_inference` printed `PASSED` |
| `GENERATE_ERROR` | `inference_scheduler.py` failed locally |
| `UPLOAD_ERROR` | SFTP transfer to the board failed |
| `BUILD_ERROR` | `cmake` or `make` failed on the board |
| `RUN_ERROR` | `test_inference` ran but printed `FAILED` or exited non-zero |

For any non-`PASSED` result the full step output is printed below the table.

---

## Debugging Failures

### BUILD_ERROR

Run with `--no-cleanup`, SSH into the board, and check the build log:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/my_model.onnx --no-cleanup

ssh root@192.168.100.8
cd /tmp/inference_hw_tests/my_model/build
make test_inference          # re-run make interactively
cat CMakeFiles/CMakeError.log
```

Common causes:

- **C name conflict**: a tensor named after a C standard library function (e.g.,
  `div`, `log`, `exp`) generates a variable that shadows the library symbol.
  Fix: rename the initializer tensor in the model generator.
- **Missing driver files**: `driver/` is empty because neither `local.driver_dir`
  nor `remote.driver_dir` was set, or the path does not contain the expected
  `.c`/`.h` files.
- **XRT not found**: `xrt.h` is not in the system include path and `XRT_DIR` was
  not set. Add `"-DXRT_DIR=/opt/xilinx/xrt"` to `remote.cmake_args`.

### RUN_ERROR — value mismatch

The test binary prints the failing element index and both the expected and actual
values. Run with `--verbose` to see the full output:

```bash
.venv/bin/python run_remote_tests.py --config remote_config.json --verbose \
    --models test/models/my_model.onnx
```

A systematic 1-LSB difference between expected and actual usually indicates a
quantization rounding mismatch between the Python simulator and the hardware. See
`src/dtype.py` for the three-mode fixed-point rounding model:
- `quantize()` — round-to-nearest (weight initialisation)
- `truncate()` — floor toward −∞ (Add/Sub/Mul output narrowing, AP_TRN)
- `truncate_div()` — truncate toward zero (Div, C integer division semantics)

### UIO device not found

The prerequisite check reports `MISSING uio (VectorOPKernel: VectorOPKernel_0)`:

```bash
# On the board: list UIO device names
cat /sys/class/uio/uio*/name

# Expected for design_matmul_vecop:
# VectorOPKernel_0
# MatmulKernel_0
```

If the names differ (older bitstream or different DTSI), update `remote.uio_devices`
in your config to match. The driver scans `/sys/class/uio/uio*/name` for a string
match — these must be sysfs names, not `/dev/uioN` paths.

If no UIO devices appear at all, the DTBO overlay may not be loaded:

```bash
# Load the overlay (KV260 with fpgautil)
fpgautil -b design_matmul_vecop.bit.bin -o matmul_vecop.dtbo

# Verify
cat /sys/class/uio/uio*/name
```

---

## Performance Benchmarking (`run_remote_perf.py`)

`run_remote_perf.py` measures raw kernel throughput and latency on a physical
KV260. Unlike `run_remote_tests.py`, it does **not** check numerical correctness
— it only cares about how fast each kernel runs for a given set of parameters.

The script:
1. Generates a self-contained C benchmark project locally (four standalone
   binaries, one per kernel)
2. Uploads it to the board once, builds everything in one `cmake` + `make` pass
3. Runs each test case as a separate binary invocation and parses the JSON output
4. Prints a formatted table of latency (ms) and throughput (GB/s or GFLOPS)

Kernels whose driver files are absent are silently skipped — you can benchmark
only the kernels that are currently deployed.

---

### Prerequisites

Same SSH, toolchain, and XRT requirements as `run_remote_tests.py`. See
[Prerequisites](#prerequisites) above. Run `--check-only` to verify before
starting a long benchmark run.

Driver files must be available either locally (`local.driver_dirs`) or on the
board (`remote.driver_dirs`). The local path is the standard Vitis HLS output:

```
<axi_demo>/build/kernels/<kernel>/kv260/<target>/solution1/impl/ip/drivers/<KernelName>_v1_0/src/
```

---

### Config File — `perf_config.json`

`perf_config.json` extends the same schema as the correctness-test configs with
one additional top-level section: `benchmarks`.

#### Full key reference

All keys from the correctness-test config apply (see [Config File Reference](#config-file-reference)). Additional and modified keys:

| Key | Default | Description |
|-----|---------|-------------|
| `remote.work_dir` | `"/tmp/inference_hw_tests"` | Temporary build directory on the board |
| `remote.uio_devices` | `{}` | Per-kernel UIO sysfs names; required for kernel initialization |
| `local.driver_dirs` | `{}` | Per-kernel local HLS driver source paths; merged before upload |
| `run.timeout` | `60` | Per-benchmark binary execution timeout in seconds |
| `benchmarks.<Kernel>.enabled` | `true` | Set `false` to skip that kernel entirely |
| `benchmarks.<Kernel>.warmup` | `10` | Default warmup iterations for all cases in this kernel group |
| `benchmarks.<Kernel>.cases` | *(required)* | List of named-parameter case dicts; see [Case Fields](#case-fields). Kernels without a `cases` array are skipped. |

#### `benchmarks` section

Each kernel group has three keys:

```json
"benchmarks": {
  "VectorOPKernel": {
    "enabled": true,
    "warmup": 10,
    "cases": [ ... ]
  },
  "MatmulKernel": { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "ConvKernel":   { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "PoolingKernel":{ "enabled": true, "warmup": 10, "cases": [ ... ] }
}
```

If `"cases"` is omitted or empty that kernel is skipped entirely. If
`"enabled"` is `false` no cases from that kernel are loaded or run
(equivalent to `--kernels` without that kernel).

#### Case fields

Each element of `"cases"` is a JSON object with a `"label"` string plus the
kernel-specific numeric fields listed below. All numeric values are integers.
An optional `"warmup"` field overrides the per-kernel warmup for that case.

**VectorOPKernel** — element-wise op benchmark

| Field | Description |
|-------|-------------|
| `label` | Display name in the report |
| `op` | Opcode: 0=ADD 1=SUB 2=MUL 3=DIV 4=RELU 5=RELU6 |
| `size` | Elements per inner kernel call |
| `outer` | Outer loop count; `outer=1` is the non-broadcast case |
| `a_inc` | Stride for A between outer iterations (`size` to advance, `0` to repeat) |
| `b_inc` | Stride for B between outer iterations (`size` to advance, `0` to repeat) |
| `iters` | Timed iterations (after warmup) |

Binary ops touch 3 memory ports (A, B, C); unary ops (RELU, RELU6) touch 2.
The reported **GB/s** accounts for this: `ports × size × outer × 2B / lat`.

**MatmulKernel** — matrix multiply benchmark

| Field | Description |
|-------|-------------|
| `n` | Rows of A and C |
| `k` | Columns of A / rows of B (accumulation dimension) |
| `m` | Columns of B and C |
| `batch` | Batch size; `1` = single matrix multiply |
| `a_stride` | Elements between A batch slices (`n×k` for batched, `0` to broadcast A) |
| `b_stride` | Elements between B batch slices (`k×m` for batched, `0` to broadcast B) |
| `iters` | Timed iterations |

Reported metric: **GFLOPS** = `2 × batch × n × k × m / lat`.

**ConvKernel** — 2-D NCHW convolution benchmark

| Field | Description |
|-------|-------------|
| `batch` | Batch size |
| `in_ch` | Input channels |
| `in_h`, `in_w` | Input spatial dimensions |
| `out_ch` | Output channels |
| `kh`, `kw` | Kernel (filter) size |
| `stride_h`, `stride_w` | Convolution stride |
| `dilation_h`, `dilation_w` | Dilation factor (1 = standard conv) |
| `pad_top`, `pad_left` | Zero-padding; use `(k−1)/2` for same-padding |
| `has_bias` | 1 to include a bias buffer in the benchmark, 0 to skip |
| `is_dw` | 1 for depthwise (grouped, `in_ch=out_ch`), 0 for standard |
| `iters` | Timed iterations |

Output size is computed by the benchmark binary: `out_h = (in_h + 2×pad_top − dil×(kh−1) − 1) / stride_h + 1`.
Reported metric: **GFLOPS** = `2 × MACs / lat` where MACs = `batch × out_ch × out_h × out_w × ic × kh × kw`
(standard conv) or `batch × out_ch × out_h × out_w × kh × kw` (depthwise).

**PoolingKernel** — 2-D NCHW pooling benchmark

| Field | Description |
|-------|-------------|
| `batch`, `channels` | Batch and channel count |
| `in_h`, `in_w` | Input spatial dimensions |
| `pool_h`, `pool_w` | Pooling window size; set to `in_h×in_w` for global pool |
| `stride_h`, `stride_w` | Pooling stride |
| `pad_top`, `pad_left` | Zero-padding |
| `dil_h`, `dil_w` | Dilation (1 = standard) |
| `pool_type` | 0=MaxPool 1=AveragePool 2=LpPool |
| `lp_order` | P value for LpPool (1 or 2); ignored for other types |
| `count_include_pad` | 1 to include padding in the average denominator |
| `iters` | Timed iterations |

Reported metric: **GB/s** = `(x_bytes + y_bytes) / lat`.

---

### Running Benchmarks

#### Full benchmark run (all enabled kernels)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json
```

#### Specific kernels only (overrides `enabled` in config)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --kernels VectorOPKernel MatmulKernel
```

#### Override iteration and warmup counts

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --iters 500 --warmup 20
```

`--iters` overrides the per-case `iters` field for every case. `--warmup`
overrides the per-kernel `warmup` for every case.

#### Preflight check (no benchmarks)

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --check-only
```

#### Verbose — show error output for failed cases

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --verbose
```

#### Keep remote build for manual inspection

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --no-cleanup
# then: ssh root@192.168.100.8
#       ls /tmp/kernel_perf/kv260_perf/build/
#       sudo /tmp/kernel_perf/kv260_perf/build/bench_vectorop fabric ADD-1K 0 16384 1 0 0 100 5
```

---

### Reading the Report

After all cases run, the script prints a per-kernel table:

```
  VectorOPKernel
  ──────────────────────────────────────────────────────────────────────────────
  Label               Parameters                        Lat(ms)      GB/s
  ──────────────────────────────────────────────────────────────────────────────
  ADD-1K              ADD    size=1024    outer=1          0.0048    1.274
  ADD-4K              ADD    size=4096    outer=1          0.0092    2.672
  ADD-16K             ADD    size=16384   outer=1          0.0214    4.580
  ADD-64K             ADD    size=65536   outer=1          0.0731    5.366
  ADD-256K            ADD    size=262144  outer=1          0.2743    5.719
  ...
  RELU-bcast-8x16K    RELU   size=16384   outer=8          0.1634    4.820
  ──────────────────────────────────────────────────────────────────────────────
  peak GB/s                                                           5.719
  min latency                                              0.0048
  14/14 OK

  MatmulKernel
  ...
  peak GFLOPS                                                         8.441

  ── OVERALL: 41/41 cases passed ──
```

| Column | Meaning |
|--------|---------|
| `Lat(ms)` | Mean kernel execution time per call (wall-clock, after warmup) |
| `GB/s` | Memory bandwidth for VectorOPKernel and PoolingKernel |
| `GFLOPS` | Arithmetic throughput for MatmulKernel and ConvKernel |
| `peak GB/s` / `peak GFLOPS` | Best metric across all passing cases in the group |
| `min latency` | Shortest latency across all passing cases in the group |
| `ERR` | Case failed; run with `--verbose` to see the error message |

The latency includes `inference_buf_sync_from_device()` (cache invalidation after
each kernel write). For large buffers this adds a measurable but consistent
overhead; it is included because it is part of the real inference path.

---

### Disabling Kernels

To skip a kernel group without removing its cases from the config:

```json
"benchmarks": {
  "ConvKernel": { "enabled": false, "warmup": 10, "cases": [ ... ] }
}
```

The CLI `--kernels` flag takes precedence over `enabled`: passing
`--kernels VectorOPKernel` runs only VectorOPKernel regardless of which kernels
are marked `enabled` in the config.

---

### Adding Custom Test Cases

Append entries to the kernel's `"cases"` array in `perf_config.json`. Use the
field tables above to set the parameters. Example — a large square matmul:

```json
{"label": "512x512x512", "n": 512, "k": 512, "m": 512,
 "batch": 1, "a_stride": 0, "b_stride": 0, "iters": 5}
```

The `"cases"` array is required; a kernel with no `"cases"` key (or an empty
array) is skipped.

---

### Debugging Failed Cases

A case shows `ERR` when the benchmark binary exits non-zero. Common causes:

| Error | Cause | Fix |
|-------|-------|-----|
| `xclOpen(0) failed` | XRT not loaded / no bitstream active | Load the FPGA overlay: `fpgautil -b <bitstream>.bit.bin -o <overlay>.dtbo` |
| `init 'fabric' failed` | UIO device name mismatch | Check `cat /sys/class/uio/uio*/name` and update `remote.uio_devices` |
| `alloc failed` | DMA buffer allocation failed | Reduce `size` or number of concurrent allocations; check `dmesg` for CMA |
| `No such file` (binary missing) | Driver files not found at build time | Verify `local.driver_dirs` paths exist and contain all `x<kernel>*.c/.h` files; re-run with `--no-cleanup` and inspect cmake output |

Run with `--verbose` to see the first 5 lines of stderr from the failing binary:

```bash
.venv/bin/python run_remote_perf.py --config perf_config.json --verbose
```

Run with `--no-cleanup` to keep the build on the board and invoke the binary
manually for interactive debugging.

---

## Multi-Kernel Models (MatmulKernel + VectorOPKernel)

Models that combine both kernels (e.g., `MatMul → Relu`, `Add → MatMul`) require
the `design_matmul_vecop` bitstream and both UIO devices active on the board.

### 1. Generate mixed-kernel test models

```bash
.venv/bin/python test/gen_mixed_kernel_models.py
# Creates 14 models in test/models/:
#   mixed_matmul_relu.onnx          mixed_add_matmul.onnx
#   mixed_matmul_add_relu.onnx      mixed_two_layer_mlp.onnx
#   mixed_add_matmul_unaligned.onnx mixed_matmul_scale_bias.onnx
#   mixed_outer_matmul_relu.onnx    mixed_relu6_matmul_add.onnx
#   mixed_residual.onnx             mixed_batch_matmul_relu.onnx
#   mixed_sub_div_matmul.onnx       mixed_two_input_matmul.onnx
#   mixed_two_output.onnx           mixed_two_input_two_output.onnx
# The last three models exercise multiple graph inputs and/or outputs:
#   mixed_two_input_matmul:      two inputs (X1, X2), one output
#   mixed_two_output:            one input, two outputs (Yadd, Yrelu)
#   mixed_two_input_two_output:  two inputs (X1, X2), two outputs (Yadd, Yrelu)
```

### 2. Load the bitstream on the board

```bash
# On the KV260
fpgautil -b design_matmul_vecop.bit.bin -o matmul_vecop.dtbo
cat /sys/class/uio/uio*/name
# VectorOPKernel_0
# MatmulKernel_0
```

### 3. Use remote_config_mixed.json

```bash
# Edit paths for your local HLS output directories
$EDITOR remote_config_mixed.json

# Preflight check
.venv/bin/python run_remote_tests.py \
    --config remote_config_mixed.json --check-only

# Run all mixed-kernel models
.venv/bin/python run_remote_tests.py --config remote_config_mixed.json
```

The runner merges driver files from `local.driver_dirs.VectorOPKernel` and
`local.driver_dirs.MatmulKernel` before uploading.  The generated `CMakeLists.txt`
passes both UIO names as compile definitions:

```
cmake … -DINFERENCE_VECTOROPKERNEL_INSTANCE="VectorOPKernel_0" \
         -DINFERENCE_MATMULKERNEL_INSTANCE="MatmulKernel_0"
```

### 4. Config files by bitstream

| Config file | Bitstream | Kernels |
|-------------|-----------|---------|
| `remote_config_vectorop.json` | `vadd_kv260` | VectorOPKernel only |
| `remote_config_matmul.json` | `matmul_kv260` | MatmulKernel only |
| `remote_config_mixed.json` | `design_matmul_vecop` | Both |

---

## Full Example: Adding a New Model

1. Create the ONNX model in `test/gen_test_models.py` and re-generate:
   ```bash
   .venv/bin/python test/gen_test_models.py
   ```

2. Verify the Python simulation locally:
   ```bash
   .venv/bin/python -m pytest test/test_saturation.py -v
   ```

3. Add the model path to `remote_config.json`:
   ```json
   "models": [
     "test/models/sat_add_pos.onnx",
     "test/models/my_new_model.onnx"
   ]
   ```

4. Run on hardware:
   ```bash
   .venv/bin/python run_remote_tests.py --config remote_config.json \
       --models test/models/my_new_model.onnx
   ```

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

Copy the example and fill in your board details:

```bash
cp remote_config.json.example remote_config.json
$EDITOR remote_config.json
```

Minimum required change: set `ssh.host` to your board's IP or hostname.

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

---

## Multi-Kernel Models (MatmulKernel + VectorOPKernel)

Models that combine both kernels (e.g., `MatMul → Relu`, `Add → MatMul`) require
the `design_matmul_vecop` bitstream and both UIO devices active on the board.

### 1. Generate mixed-kernel test models

```bash
.venv/bin/python test/gen_mixed_kernel_models.py
# Creates: test/models/mixed_matmul_relu.onnx, mixed_add_matmul.onnx, etc.
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

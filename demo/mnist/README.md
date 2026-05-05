# MNIST KV260 demo

End-to-end MNIST inference demo for the KV260 FPGA platform.  Downloads the
MNIST test split and two pre-trained ONNX models (an MNIST convnet and a
LeNet variant) from a shared Google Drive folder, generates a self-contained
KV260 inference project for each model with `inference-scheduler`, copies
in the HLS driver sources, builds the project on the board over SSH, and
runs a benchmark that reports top-1 accuracy and per-image latency.

```
download_assets.py   →   generate_project.py   →   deploy_and_run.py
   MNIST + .onnx           per-model CMake             SSH upload, build,
                           project + bench_glue        run bench_mnist
```

## Layout

```
demo/mnist/
├── README.md
├── requirements.txt
├── mnist_config.json.example       — copy to mnist_config.json and edit
├── run_demo.py                     — one-shot orchestrator
├── scripts/
│   ├── download_assets.py          — fetch MNIST IDX + ONNX models
│   ├── generate_project.py         — schedule each ONNX → CMake project
│   └── deploy_and_run.py           — upload, build, run on KV260 over SSH
├── src/
│   └── bench_mnist.c               — benchmark host (compiled on the board)
├── assets/                         — populated by download_assets.py
│   ├── data/                       — MNIST IDX files
│   └── models/                     — ONNX files from Google Drive
└── build/                          — generator output and results.json
    ├── projects/<model>/           — generated CMake project per model
    │   ├── CMakeLists.txt          —   patched to build bench_mnist
    │   ├── include/inference.h     —   from inference-scheduler
    │   ├── src/inference*.c        —   from inference-scheduler
    │   ├── driver/                 —   HLS driver sources copied in
    │   └── test/
    │       ├── bench_mnist.c       —   copied from demo/mnist/src/
    │       └── bench_glue.h        —   generated; per-model glue + macros
    ├── logs/<model>.<step>.log     — per-step build/run output
    └── results.json                — final benchmark summary
```

`test/bench_glue.h` is **generated per model** by `scripts/generate_project.py`
— not committed in the repo.  Each copy bakes in the right
`INFERENCE_<INPUT>_SIZE` / `INFERENCE_<OUTPUT>_SIZE` macros and a
`bench_inference_init()` shim that calls `inference_init()` with the correct
number of UIO arguments for the kernels that model actually uses (e.g. 3 for
LeNet, 4 for the MNIST convnet).  That's why it's generated rather than
static: the I/O names and active-kernel set differ per model.

## Prerequisites

### Host (workstation that orchestrates the demo)

* Python 3.10+
* `pip install -r requirements.txt`
* HLS-generated driver sources for the four kernels.  These are produced by
  the top-level CMake build:

  ```bash
  # from the repo root
  mkdir -p build && cd build
  cmake -DAXI_BUS_WIDTH=128 ..
  make synthesize_kv260
  ```

  The default `mnist_config.json.example` assumes the standard build paths;
  override `local.driver_dirs` if you keep the build elsewhere.

### KV260 board

* Linux with the cormorant overlay loaded (so the four UIO devices appear).
  Verify on the board:

  ```bash
  cat /sys/class/uio/uio*/name
  # → fabric_vecop, fabric_matmul, fabric_conv, fabric_pool
  ```

* `gcc`, `cmake ≥ 3.19`, `make`
* XRT runtime available via `pkg-config xrt` or under `/opt/xilinx/xrt`
* Passwordless `sudo` for the SSH user (XRT requires root for
  buffer allocation).

## First-time setup

```bash
cd demo/mnist
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp mnist_config.json.example mnist_config.json
$EDITOR mnist_config.json          # set ssh.host, key_file, etc.
```

## Run the demo

One-liner that does download → generate → deploy:

```bash
.venv/bin/python run_demo.py
```

Or step-by-step (lets you iterate without re-downloading):

```bash
.venv/bin/python scripts/download_assets.py
.venv/bin/python scripts/generate_project.py
.venv/bin/python scripts/deploy_and_run.py --verbose
```

Sample output:

```
  ── MNIST KV260 BENCHMARK ──

  Model           Status         Acc  mean(ms)   p50(ms)   p99(ms)        IPS
  ─────────────────────────────────────────────────────────────────────────────
  mnist_convnet   OK          98.92%    21.287    21.286    21.309       47.0
```

The full per-image numbers are also written to `build/results.json`.

## Useful options

| Flag | Effect |
|------|--------|
| `--models lenet` | restrict generate/deploy to a subset of models |
| `--skip-download` | reuse cached MNIST + ONNX files |
| `--skip-deploy` | only regenerate the local CMake projects |
| `--force-download` | re-fetch all assets |
| `--no-cleanup` (deploy script) | keep the remote work_dir for inspection |
| `--verbose` | print full build / run output for failed steps |

## Tuning the run

Edit `mnist_config.json`:

* **`run.iters`** — number of test images to evaluate.  `0` means the full
  10 000-image test set.
* **`run.warmup`** — how many inferences to discard before the timed window.
* **`run.use_sudo`** — set to `false` if you SSH in directly as root.
* **`remote.cmake_args`** — extra `-D…` flags forwarded to the cross-build,
  e.g. `["-DBENCH_INPUT_BIAS=-128"]` if your model expects centred input.

## Troubleshooting

* **`ImportError: paramiko`** — `pip install -r requirements.txt`.
* **`gdown failed`** — Google Drive sometimes throttles.  Run
  `python3 -m gdown --folder <url> -O assets/models/` manually, or download
  the `.onnx` files via a browser and place them in `assets/models/`.
* **`Driver file not found: driver/xconvkernel.h`** during cmake — the HLS
  driver sources weren't on this host.  Either run `make synthesize_kv260`
  in the repo build, or point `local.driver_dirs` at an existing build
  output.
* **`UIO device 'fabric_pool' not found`** — the loaded overlay does not
  expose all four kernels.  Update `remote.uio_devices` to the names that
  appear in `cat /sys/class/uio/uio*/name`.
* **Accuracy near 10 %** — almost always an input encoding mismatch.  Try
  `BENCH_INPUT_BIAS=-128` in `remote.cmake_args` (centred input) or
  re-train / re-export the model with the expected input range.

# Inference Scheduler — Technical Reference

The inference scheduler is a Python code-generator that bridges the gap between
trained neural network models (ONNX format) and the hardware accelerators running
on the KV260.  It reads an ONNX graph, validates that every operator can be
executed by one of the supported hardware kernels, and emits a self-contained C
project that drives the IP through the auto-generated Xilinx driver APIs.

> **Full documentation** lives in
> [`inference-scheduler/doc/INFERENCE_SCHEDULER.md`](../inference-scheduler/doc/INFERENCE_SCHEDULER.md)
> and
> [`inference-scheduler/doc/ARCHITECTURE.md`](../inference-scheduler/doc/ARCHITECTURE.md).
> This file provides a quick orientation.

---

## Hardware Kernels

| Kernel | ONNX ops handled | Notes |
|--------|-----------------|-------|
| **VectorOPKernel** | `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(0,6)` | 1-D element-wise, II=1 |
| **MatmulKernel** | `MatMul` | Tiled 2-D matrix multiply |
| **ConvKernel** | `Conv` | 2-D NCHW convolution with optional bias |
| **PoolingKernel** | `MaxPool`, `AveragePool`, `LpPool`, `GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool` | 2-D NCHW pooling |

**Zero-cost transformations (no hardware call):**
- `Reshape` — output pointer is aliased to the source buffer; no data copy.
- `Gemm` — decomposed to `MatMul` + optional `Add` at model load time
  (`alpha=1, beta=1, transA=0, transB=0` required).

---

## Usage

```bash
cd inference-scheduler

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate all test ONNX models
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py

# Generate a complete C inference project from an ONNX model
.venv/bin/python inference_scheduler.py model.onnx --out-dir /tmp/out

# Run the full test suite (897 tests)
.venv/bin/python -m pytest test/ -v
```

---

## Architecture

```
inference_scheduler.py          CLI, argument parsing
└── src/
    ├── graph.py    OnnxGraph   load, shape inference, Gemm preprocessing,
    │                           tensor registry, node dispatch
    ├── tensor.py   TensorInfo  weight encoding, C declarations
    ├── nodes.py                ScheduledNode  (VectorOPKernel)
    │                           MatmulNode     (MatmulKernel)
    │                           ConvNode       (ConvKernel)
    │                           PoolNode       (PoolingKernel)
    │                           ReshapeNode    (buffer alias)
    └── codegen/    CodeGenerator
                    _core.py    tensor layout + DMA pool sizing
                    _header.py  include/inference.h
                    _source.py  src/inference.c  (weights, init, run)
                    _simulate.py  fixed-point forward simulation
                    _test.py    test/test_inference.c  (on-device smoke test)
                    _cmake.py   CMakeLists.txt
```

### OnnxGraph loading sequence

1. `onnx.load()` + `onnx.checker.check_model()` — structural validation.
2. `shape_inference.infer_shapes()` — fills intermediate tensor shapes.
3. `_preprocess_model()` — rewrites `Gemm` → `MatMul` + optional `Add`.
4. Build tensor registry (weights, inputs, intermediates, outputs).
5. Dispatch each node to `MatmulNode` / `ConvNode` / `PoolNode` / `ReshapeNode`
   / `ScheduledNode` based on `op_type`.

### Key data flow

```
model.onnx  →  OnnxGraph  →  CodeGenerator  →  C project
                                 │
                    ┌────────────┤
                    │            │
               TensorLayout   emit:
               (alloc sizes,   run_op()    VectorOPKernel calls
                strides,       run_matmul() MatmulKernel calls
                n_chunks)      run_conv()   ConvKernel calls
                               run_pool()   PoolingKernel calls
```

### DMA buffer management

- All buffers allocated in `inference_init()` from a contiguous DMA pool.
- `Reshape` output buffers are pointer-assigned (= source), never independently allocated.
- `inference_run()` flushes all graph inputs to DDR at the top, invalidates all
  graph outputs at the bottom. Internal intermediate buffers are never synced —
  the PL kernels access DDR directly via their AXI master ports.
- Weights are synced once at init; they never change.

---

## Data Type

Default: `ap_fixed<16,8>` — 16-bit two's complement, 8 integer bits, 8
fractional bits. Encoding: `1.0 = 0x0100`, `0.5 = 0x0080`, range `[-128, 127.996]`.

The `DataType` abstraction in `src/dtype.py` allows other types (e.g. `float32`)
to be plugged in without changing any other source file.

---

## Generated C API

```c
// inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES     16u
#define INFERENCE_BUF_POOL_SIZE_BYTES  N

// One per active kernel (only present kernels appear):
int  inference_init(const char *vectoropkernel_instance
                    [, const char *matmulkernel_instance]
                    [, const char *convkernel_instance]
                    [, const char *poolkernel_instance]);

// All graph inputs, then all graph outputs:
void inference_run(inference_buf_t *<input...>, inference_buf_t *<output...>);
void inference_deinit(void);
```

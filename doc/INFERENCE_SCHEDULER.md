# Inference Scheduler — Technical Reference

The inference scheduler is a Python code-generator that bridges the gap between
trained neural network models (ONNX format) and the `VectorOPKernel` HLS
accelerator running on the KV260.  It reads an ONNX graph, validates that every
operator can be executed by `VectorOPKernel`, and emits a self-contained C
source file that drives the IP through the auto-generated `XVectoropkernel`
driver API and the Xilinx bare-metal (`Xil_*`) library.

---

## Hardware Constraints

All design decisions flow from one central fact: the only accelerator available
is `VectorOPKernel`, a 1-D element-wise engine.

| Property | Value |
|----------|-------|
| Operations | Add, Sub, Mul, Div, Relu, Relu6 |
| Input arrays | Up to 2 (`a[]` and `b[]`); unary ops use `a[]` only |
| Data type | `ap_fixed<16,8>` (16-bit, 8 integer bits, 8 fractional bits) |
| Throughput | II=1 after AXI burst ramp-up |
| Control | AXI-Lite (`s_axi_ctrl`): `a_addr`, `b_addr`, `c_addr`, `size`, `op` |
| Memory | AXI4 master reads/writes directly to DDR; CPU cache coherence required |

Because the kernel is 1-D, all tensor shapes are flattened.  A tensor with
shape `[1, 3, 224, 224]` is treated as a flat array of 150 528 elements.

---

## Supported ONNX Operators

| ONNX `op_type` | Kernel op | Arity | Notes |
|---------------|-----------|-------|-------|
| `Add` | `OP_ADD` (0) | binary | |
| `Sub` | `OP_SUB` (1) | binary | |
| `Mul` | `OP_MUL` (2) | binary | |
| `Div` | `OP_DIV` (3) | binary | Division by zero returns 0 |
| `Relu` | `OP_RELU` (4) | unary | `max(x, 0)` |
| `Clip(min=0, max=6)` | `OP_RELU6` (5) | unary | Attribute- and input-based bounds both supported |

Any other `op_type` causes the tool to exit with a `SchedulerError`.

---

## Usage

```bash
cd inference-scheduler

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate test ONNX models (writes to test/models/)
.venv/bin/python test/gen_test_models.py

# Compile a model — output to stdout
.venv/bin/python inference_scheduler.py model.onnx

# Compile a model — output to file
.venv/bin/python inference_scheduler.py model.onnx -o inference.c

# Override the driver header path written into the generated #include
.venv/bin/python inference_scheduler.py model.onnx \
    --driver-include ../build/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src/xvectoropkernel.h
```

A summary of the scheduled graph is printed to `stderr`; the C code goes to
`stdout` or the specified file.

---

## Architecture

```
inference_scheduler.py          CLI, argument parsing
└── src/
    ├── graph.py    OnnxGraph   model loading, shape inference, tensor registry
    ├── tensor.py   TensorInfo  weight encoding, C declarations
    ├── nodes.py    ScheduledNode  op mapping, validation, call emission
    └── codegen.py  CodeGenerator  assembles the final .c file
```

### `OnnxGraph` (graph.py)

Construction sequence:

1. `onnx.load()` + `onnx.checker.check_model()` — basic sanity check.
2. `onnx.shape_inference.infer_shapes()` — populates `value_info` for every
   intermediate tensor; required because the ONNX spec allows models to omit
   intermediate shapes.
3. Build a `dict[name → TensorInfo]` covering:
   - `graph.initializer` — constant weights (have `data`)
   - `graph.input` minus initializers — true model inputs (no `data`)
   - `graph.value_info` — inferred intermediates (no `data`)
   - `graph.output` — model outputs (no `data`)
4. Wrap each `graph.node` in a `ScheduledNode`; fail fast on unsupported ops.

ONNX mandates that nodes appear in topological order, so no explicit sort is
needed.

### `TensorInfo` (tensor.py)

Holds tensor metadata and emits C declarations:

```python
@dataclass
class TensorInfo:
    onnx_name: str
    shape:     List[int]
    dtype:     str             # 'float32', 'int8', …
    data:      Optional[np.ndarray]  # None for non-constants

    def emit_weight_decl(self) -> str   # static const uint16_t name[N] = {...};
    def emit_buffer_decl(self) -> str   # static Data_t name[N];
```

**Weight encoding** converts `float32` model weights to `ap_fixed<16,8>`
bit patterns in a single NumPy pass:

```python
clipped = np.clip(data, -128.0, 127.99609375)
bits    = np.round(clipped * 256).astype(np.int16).view(np.uint16)
```

This matches how Xilinx HLS stores `ap_fixed<16,8>` values in memory:
each element is a 16-bit two's-complement integer equal to `round(v × 256)`.
The encoding is the same as the one described in `doc/SIMULATION_ISSUES.md`
(section "ap_fixed<16,8> encoding"), so weight arrays produced by the
scheduler can be directly compared with the reference model in that document.

### `ScheduledNode` (nodes.py)

Wraps one `onnx.NodeProto` and handles two concerns:

**Validation** (`validate()`):
- Confirms the op is in the supported set.
- For `Clip`: checks `min=0, max=6`; handles both attribute-style (opset < 11)
  and input-style (opset ≥ 11) bounds; strips bound inputs so `emit_call()`
  only sees the data tensor.
- Checks that all input and output tensors have the same number of elements
  (required by the flat 1-D kernel).

**Emission** (`emit_call()`):
```c
/* binary */
run_op(a_name, b_name, c_name, N, VECTOROP_ADD);

/* unary */
run_op(a_name, NULL, c_name, N, VECTOROP_RELU);
```

### `CodeGenerator` (codegen.py)

Assembles the final C file in fixed section order:

| Section | Contents |
|---------|----------|
| File header | Tool version, model name, timestamp, op summary |
| Includes | `xvectoropkernel.h`, `xil_cache.h`, `xil_types.h`, `string.h`, `stdint.h` |
| Defines | `typedef uint16_t Data_t`, `BYTES_PER_ELEM 2u`, `VECTOROP_*` constants |
| Weight arrays | One `static const uint16_t` array per ONNX initializer |
| Buffers | Comments for model inputs/outputs; `static Data_t` for intermediates |
| Kernel instance | `static XVectoropkernel s_kernel;` |
| `run_op()` | Configure + cache flush + start + poll + invalidate |
| `inference_init()` | Wraps `XVectoropkernel_Initialize()` |
| `inference_run()` | Sequential `run_op()` calls, one per node |

---

## Generated Code Walkthrough

Given the `mixed_ops` test model (Add → Mul → Clip(0,6) on a `[1, 64]` tensor):

```c
/* [0] Add(X, bias1) -> add_Y  shape=[1, 64] */
run_op(X, bias1, add_Y, 64u, VECTOROP_ADD);

/* [1] Mul(add_Y, scale) -> mul_Y  shape=[1, 64] */
run_op(add_Y, scale, mul_Y, 64u, VECTOROP_MUL);

/* [2] Clip(mul_Y) -> clip_Y  shape=[1, 64] */
run_op(mul_Y, NULL, clip_Y, 64u, VECTOROP_RELU6);
```

`run_op()` handles all hardware interaction:

```c
static void run_op(const Data_t *a, const Data_t *b, Data_t *c,
                   unsigned size, unsigned op)
{
    /* Write AXI-Lite registers */
    XVectoropkernel_Set_a(&s_kernel, (u64)(UINTPTR)a);
    XVectoropkernel_Set_b(&s_kernel, (u64)(UINTPTR)b);
    XVectoropkernel_Set_c(&s_kernel, (u64)(UINTPTR)c);
    XVectoropkernel_Set_size(&s_kernel, size);
    XVectoropkernel_Set_op(&s_kernel, op);

    /* Flush CPU cache → DDR before the AXI master reads */
    Xil_DCacheFlushRange((INTPTR)a, (INTPTR)(size * BYTES_PER_ELEM));
    if (b != NULL)
        Xil_DCacheFlushRange((INTPTR)b, (INTPTR)(size * BYTES_PER_ELEM));

    /* Start and poll */
    XVectoropkernel_Start(&s_kernel);
    while (!XVectoropkernel_IsDone(&s_kernel)) {}

    /* Invalidate CPU cache so the CPU sees what the kernel wrote */
    Xil_DCacheInvalidateRange((INTPTR)c, (INTPTR)(size * BYTES_PER_ELEM));
}
```

### Cache coherence

The Cortex-A53 on the KV260 has a hardware-managed L1/L2 cache, but
`VectorOPKernel`'s AXI master ports bypass the cache and access DDR
directly.  Without explicit cache maintenance:

- A write to `a[]` in the CPU cache may not reach DDR before the kernel reads
  it → stale data in the accelerator.
- A write to `c[]` by the kernel may be overwritten by a stale cache line when
  the CPU next reads `c[]` → CPU sees old data.

`Xil_DCacheFlushRange()` writes dirty cache lines back to DDR before the
kernel starts.  `Xil_DCacheInvalidateRange()` discards cache lines for the
output region after the kernel finishes, forcing the CPU to re-fetch from DDR
on the next read.

Weight arrays (`static const uint16_t`) are in the `.rodata` section.
Because they are written once at link time and never modified, a single flush
at `inference_init()` time is sufficient — but the current implementation
flushes them on every `run_op()` call for simplicity.  If throughput matters
this can be optimised by flushing weights once during initialisation.

---

## Integration into a Bare-Metal Application

1. **Include the generated file** in your SDK project alongside the
   `XVectoropkernel` driver sources.

2. **Call `inference_init()`** once after hardware initialisation:
   ```c
   if (inference_init("VectorOPKernel") != XST_SUCCESS) {
       /* handle error */
   }
   ```
   In bare-metal builds, pass the device ID (cast to `const char *`) or
   adjust the call to use `XVectoropkernel_CfgInitialize()` directly.

3. **Call `inference_run()`** with pointers to input and output buffers:
   ```c
   Data_t input[64];
   Data_t output[64];
   /* populate input[] … */
   inference_run(input, output);
   /* read output[] … */
   ```

4. **Buffer alignment**: the AXI-Lite `b_addr` register is set even for unary
   ops (passed as `NULL` = 0x0).  Ensure `VectorOPKernel` ignores `gmem1`
   transactions when `op` ≥ `OP_RELU`; the kernel source confirms it does.

---

## Limitations

| Limitation | Reason |
|------------|--------|
| Only element-wise ops supported | `VectorOPKernel` has no convolution, matmul, or reduction hardware |
| All tensors must have the same shape | The kernel processes `a[]`, `b[]`, `c[]` with a single `size` counter |
| No dynamic shapes | Shape inference is done at compile time; `size` is a compile-time constant in the generated call |
| Batch size = 1 | The flat 1-D model means the full tensor (all batch elements) is processed in one invocation |
| `ap_fixed<16,8>` only | Weight encoding targets this specific type; changing the data type requires rebuilding the IP and updating `BYTES_PER_ELEM` |
| Sequential execution | Nodes are executed one at a time; no inter-layer pipelining |

---

## Adding a New ONNX Operator

To support a new operation, two changes are needed:

1. **Add a new `Op` code to `VectorOPKernel`** in `kernel/VectorOP.cpp` and
   `include/VectorOP.h`, then re-synthesise.

2. **Register the op in `inference-scheduler/src/nodes.py`**:
   ```python
   _ONNX_OP_MAP["NewOp"] = (OP_NEW, arity)
   ```
   If the op requires attribute parsing or special validation, add a branch in
   `ScheduledNode.validate()`.

---

## Test Infrastructure

```
test/
├── gen_test_models.py   Creates four ONNX models in test/models/
│                          single_add.onnx   — Add + weight initializer
│                          relu_chain.onnx   — Add → Relu
│                          mixed_ops.onnx    — Add → Mul → Clip(0,6)
│                          unsupported.onnx  — Conv (triggers SchedulerError)
└── test_scheduler.py    28 unit tests covering:
                           TestTensorEncoding  — ap_fixed<16,8> bit patterns
                           TestGraphParsing    — node count, op mapping, error path
                           TestCodeGen         — presence of all generated sections
                           TestCLI             — exit codes, file output
```

Run with:
```bash
.venv/bin/python -m pytest test/test_scheduler.py -v
```

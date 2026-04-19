# Inference Scheduler — Internal Architecture

This document describes how the scheduler works internally: how it processes the
ONNX graph, how it reasons about shapes and broadcasting, how it generates C code,
and how to extend it.

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Source Layout](#2-source-layout)
3. [Step 1 — ONNX Graph Parsing (OnnxGraph)](#3-step-1--onnx-graph-parsing-onnxgraph)
4. [Step 2 — Tensor Classification (TensorInfo)](#4-step-2--tensor-classification-tensorinfo)
5. [Step 3 — Op Mapping (ScheduledNode)](#5-step-3--op-mapping-schedulednode)
6. [Broadcasting Algorithm](#6-broadcasting-algorithm)
7. [Step 4 — Allocation Sizing (_CoreMixin)](#7-step-4--allocation-sizing-_coremixin)
8. [Step 5 — Code Generation (CodeGenerator)](#8-step-5--code-generation-codegenerator)
9. [DataType Abstraction](#9-datatype-abstraction)
10. [Fixed-Point Simulation (_SimulateMixin)](#10-fixed-point-simulation-_simulatemixin)
11. [Cache Coherency Model](#11-cache-coherency-model)
12. [Extending the System](#12-extending-the-system)

---

## 1. High-Level Pipeline

```
model.onnx
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  OnnxGraph (src/graph.py)                                            │
│                                                                      │
│  1. Load ONNX proto, validate with onnx.checker.check_model()        │
│  2. Run shape inference (onnx.shape_inference.infer_shapes)          │
│  3. Build tensor registry: weights, inputs, intermediates, outputs   │
│  4. For each node: ScheduledNode.from_onnx_node()                    │
│     - Map op_type → (op_code, arity)                                 │
│     - Validate broadcasting constraints                              │
│     - Compute: outer_count, chunk_size, aligned_chunk_size           │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼ OnnxGraph (nodes, tensors)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  CodeGenerator._CoreMixin (src/codegen/_core.py)                     │
│                                                                      │
│  5. _compute_alloc_sizes()                                           │
│     - Seed with natural sizes (numel)                                │
│     - Override for broadcast nodes (padded stride layouts)           │
│     - Forward-propagate padding through non-broadcast nodes          │
│  6. _compute_pool_bytes() — total DMA memory needed                  │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼ alloc_sizes dict
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Code Emission (mixins in src/codegen/)                              │
│                                                                      │
│  _HeaderMixin  →  include/inference.h   (types, macros, API decls)  │
│  _SourceMixin  →  src/inference.c       (weight arrays, init, run)  │
│  _BufImplMixin →  src/inference_buf.c   (DMA alloc, cache sync)     │
│  _SimulateMixin + _TestMixin → test/test_inference.c                 │
│  _CmakeMixin   →  CMakeLists.txt                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source Layout

```
inference_scheduler.py   CLI entry point (argparse + file I/O)
src/
  dtype.py               DataType ABC + ApFixed, Float32 implementations
  tensor.py              TensorInfo dataclass — metadata + C code emitters
  nodes.py               ScheduledNode dataclass — op mapping + C call emitters
  graph.py               OnnxGraph — ONNX loading, shape inference, node scheduling
  codegen/
    __init__.py          CodeGenerator class (assembles all mixins via MRO)
    _core.py             _CoreMixin: __init__, alloc sizes, pool size, helpers
    _header.py           _HeaderMixin: generate_header()
    _source.py           _SourceMixin: generate_source()
    _buf_impl.py         _BufImplMixin: generate_buf_impl(), generate_setup_script()
    _simulate.py         _SimulateMixin: fixed-point simulation, expected GT arrays
    _test.py             _TestMixin: generate_test()
    _cmake.py            _CmakeMixin: generate_cmake()
    _banners.py          _banner(), _file_banner() — section header helpers
```

---

## 3. Step 1 — ONNX Graph Parsing (OnnxGraph)

`OnnxGraph` in `src/graph.py` wraps the ONNX protobuf loading and produces a
clean, typed view of the computation graph.

### Loading and Shape Inference

```python
model = onnx.load(model_path)
onnx.checker.check_model(model)          # validate structural correctness
model = shape_inference.infer_shapes(model)  # fill in intermediate shapes
```

The shape inference step is critical. Without it, intermediate tensors (the
outputs of each ONNX node that feed into the next) have no shape information.
After `infer_shapes`, every tensor's shape is available in `model.graph.value_info`.

### Tensor Registry

The scheduler builds a flat dictionary `{name → TensorInfo}` covering all four
categories of tensor in the ONNX graph:

| Category | Source in ONNX proto | Role |
|----------|---------------------|------|
| **Weights** | `graph.initializer` | Constant values (model parameters); have `data != None` |
| **Inputs** | `graph.input` minus initializers | Caller-supplied at runtime; have `data == None` |
| **Intermediates** | `graph.value_info` | Produced by one node, consumed by the next; scratch buffers |
| **Outputs** | `graph.output` | Final results written to caller-supplied buffers |

> **Why "minus initializers"?** Older ONNX opsets sometimes list constant weights
> in both `graph.input` and `graph.initializer`. The scheduler skips duplicates:
> any name already in the initializer set is not treated as a true graph input.

---

## 4. Step 2 — Tensor Classification (TensorInfo)

`TensorInfo` in `src/tensor.py` is a simple dataclass:

```python
@dataclass
class TensorInfo:
    onnx_name: str          # original ONNX tensor name
    shape:     List[int]    # e.g. [1, 64, 32, 32]
    dtype:     str          # ONNX dtype string, e.g. 'float32'
    data:      np.ndarray   # None for non-constant tensors
```

Key derived properties:

- **`numel`**: total number of elements (`product(shape)`)
- **`c_name`**: C identifier derived from `onnx_name` — non-alphanumeric characters
  are replaced with underscores, leading digits get a `t_` prefix
- **`is_weight`**: `data is not None`
- **`is_large_weight`**: `is_weight and numel > LARGE_WEIGHT_THRESHOLD (4096)`

### Code Emission Methods

`TensorInfo` knows how to produce the C declarations for its data:

| Method | When used | Output |
|--------|-----------|--------|
| `emit_weight_decl(dtype)` | Small weight tensor | `static const uint16_t _rom_bias[N] = {...};` + DMA pointer |
| `emit_weight_decl_strided(outer, stride, dtype)` | Weight used alongside a strided buffer | Same but with alignment padding between blocks |
| `emit_large_weight_ptr_decl()` | Large weight tensor | Just the DMA pointer; data loaded from `.dat` at runtime |
| `emit_buffer_decl()` | Intermediate tensor | `static inference_buf_t *name = NULL;` |

---

## 5. Step 3 — Op Mapping (ScheduledNode)

`ScheduledNode` in `src/nodes.py` wraps one ONNX `NodeProto` and converts it
to one or more VectorOPKernel invocations.

### Op Mapping Table

```python
_ONNX_OP_MAP = {
    "Add":  (OP_ADD,   arity=2),
    "Sub":  (OP_SUB,   arity=2),
    "Mul":  (OP_MUL,   arity=2),
    "Div":  (OP_DIV,   arity=2),
    "Relu": (OP_RELU,  arity=1),
    "Clip": (OP_RELU6, arity=1),   # validated for (min=0, max=6)
}
```

### Clip Validation

The ONNX `Clip` operator has configurable `min` and `max` bounds. The hardware
only implements `RELU6 = clip(x, 0, 6)`. The scheduler validates the bounds:

- **ONNX opset < 11**: bounds stored as attributes `min` and `max`
- **ONNX opset ≥ 11**: bounds stored as constant input tensors at indices 1 and 2

If either `min ≠ 0` or `max ≠ 6`, a `SchedulerError` is raised.

### ScheduledNode Fields

```python
@dataclass
class ScheduledNode:
    onnx_node:          onnx.NodeProto
    op_code:            int            # OP_ADD … OP_RELU6
    arity:              int            # 1 = unary, 2 = binary
    inputs:             List[TensorInfo]
    output:             TensorInfo
    index:              int            # sequential position in graph
    align_elems:        int            # ALIGN_BYTES / bytes_per_elem

    # Set by validate():
    outer_count:        int   # > 1 = broadcasting; loop iteration count
    chunk_size:         int   # data elements per kernel call
    aligned_chunk_size: int   # chunk_size rounded up to align_elems
    a_advances:         bool  # True if input A strides through the output
    b_advances:         bool  # True if input B strides through the output
```

### Code Emission

`emit_call()` produces the C kernel invocation:

**Non-broadcast case** (`outer_count == 1`):
```c
    run_op(X, bias, Y, 256u, VECTOROP_ADD);
```

**Broadcast case** (`outer_count > 1`):
```c
    for (unsigned _i = 0u; _i < 4u; _i++) {
        run_op_at(X, _i * INFERENCE_Y_CHUNK_STRIDE,
                  bias, 0u,
                  Y,  _i * INFERENCE_Y_CHUNK_STRIDE,
                  INFERENCE_Y_CHUNK, VECTOROP_ADD);
    }
```

Here `bias` does not advance (`b_advances = False`) — it repeats at offset 0
every iteration (a single-row bias added to each of 4 output rows). `X` and `Y`
both advance — they each hold 4 chunks strided by `CHUNK_STRIDE`.

---

## 6. Broadcasting Algorithm

This is the most algorithmically complex part of the scheduler.

### When Does Broadcasting Apply?

A binary ONNX op (Add, Sub, Mul, Div) broadcasts when one input has fewer
elements than the output. The canonical example is adding a per-channel bias
to a batch of feature maps:

```
X shape:    [4, 32, 64]   → 8192 elements (batch of 4, 32 rows, 64 cols)
bias shape: [1, 1, 64]    →   64 elements (one 64-element row, repeated)
Y shape:    [4, 32, 64]   → 8192 elements
```

ONNX multidirectional broadcasting would compute `Y[b,r,c] = X[b,r,c] + bias[0,0,c]`.

### The Trailing-Contiguous Constraint

VectorOPKernel processes a flat 1-D array per call. It cannot do arbitrary
striding or gather/scatter. To map ONNX broadcasting to sequential flat calls,
the scheduler requires that broadcast dimensions form a **contiguous leading block**
after right-aligning the shapes.

**Right-alignment**: a shorter shape is left-padded with 1s to match the output rank.

```
output: [4, 32, 64]
bias:   [1,  1, 64]   ← right-aligned, broadcast dims are {0, 1}, matching dim is {2}
```

The rule: all broadcast (size-1) dims must come before all matching dims. No
matching dim may appear before a broadcast dim in the aligned shape.

**Valid examples:**

| Input shape | Output shape | Outer count | Chunk | Notes |
|-------------|-------------|-------------|-------|-------|
| `[64]` | `[4, 32, 64]` | 128 | 64 | Leading 128 dims broadcast |
| `[1, 64]` | `[4, 32, 64]` | 128 | 64 | Same after right-align |
| `[1, 1, 64]` | `[4, 32, 64]` | 128 | 64 | Explicit 1s |
| `[32, 64]` | `[4, 32, 64]` | 4 | 2048 | Outer dim broadcasts |
| `[4, 32, 64]` | `[4, 32, 64]` | 1 | 8192 | No broadcast (exact match) |

**Invalid examples (rejected with SchedulerError):**

| Input shape | Output shape | Problem |
|-------------|-------------|---------|
| `[4, 1, 64]` | `[4, 32, 64]` | Broadcast dim (1) appears between two matching dims (4, 64) |
| `[3, 64]` | `[4, 32, 64]` | `3 ≠ 32` and `3 ≠ 1` — neither matching nor broadcast |
| both inputs smaller | any | Only one input may broadcast per op |

### Computing Broadcast Parameters

Given a valid broadcast, `_broadcast_info()` computes:

```python
outer_count = output.numel // t.numel   # how many times to repeat t
chunk_size  = t.numel                   # elements per kernel call
aligned_chunk_size = ceil_to(chunk_size, align_elems)
                     # round up to 16-byte alignment boundary
```

For `bias [64]` → `output [4, 32, 64]`:
```
outer_count        = 8192 // 64  = 128
chunk_size         = 64
aligned_chunk_size = 64          (64 is already a multiple of 8 for ap_fixed<16,8>)
```

For `bias [6]` → `output [4, 32, 6]` with ap_fixed<16,8> (align_elems=8):
```
outer_count        = 768 // 6  = 128
chunk_size         = 6
aligned_chunk_size = 8          (next multiple of 8 above 6)
gap_elements       = 2          (slots 6 and 7 are zero-padded)
```

### Physical Memory Layout

When `aligned_chunk_size > chunk_size`, the DMA buffer has gap elements between
data blocks:

```
Buffer for bias [6], allocated as 128 × 8 = 1024 elements:

offset 0:  [ data[0] data[1] data[2] data[3] data[4] data[5]  0  0 ]  ← block 0
offset 8:  [ data[0] data[1] data[2] data[3] data[4] data[5]  0  0 ]  ← block 1
offset 16: [ data[0] data[1] data[2] data[3] data[4] data[5]  0  0 ]  ← block 2
...
```

Each block starts at a 16-byte-aligned physical address. The kernel is told
`size = 6` so it reads only the 6 data elements and ignores the 2 gap slots.

The Python `_simulate.py` and `_test.py` modules both use the same strided layout
when filling inputs and comparing expected outputs, ensuring CPU-side simulation
and on-device execution produce identical memory patterns.

### Two Inputs That Both Stride

In a non-broadcast binary op where the output buffer was padded by an earlier
broadcast node, both inputs may appear to "stride" even though neither is
technically broadcasting. The scheduler handles this in `_compute_alloc_sizes()`:
sizes are propagated forward through the topological node order, and all co-inputs
of a non-broadcast node are raised to the same padded alloc size.

---

## 7. Step 4 — Allocation Sizing (_CoreMixin)

`_compute_alloc_sizes()` returns `{onnx_name: alloc_size_in_elements}` for every
tensor that needs a DMA buffer.

### The Algorithm

**Pass 1 — Seed with natural sizes**: every tensor starts at its `numel`.

**Pass 2 — Broadcast nodes**: for each node with `outer_count > 1`:
- Output: `alloc = outer_count × aligned_chunk_size`
- Advancing input (strides through output): same `alloc`
- Repeating input (fixed at offset 0): just `aligned_chunk_size` (one block)

**Pass 3 — Non-broadcast propagation**: processed in topological order.
For a non-broadcast node, find the maximum alloc among its inputs. If that
exceeds the output's current alloc, raise the output. Then raise every input
to the same alloc. This handles the case where a Relu follows a broadcast Add:
the Relu's input (the Add's output) has a padded alloc, so the Relu's output
must match, and the `run_op()` call gets `size = padded_alloc` to process all
elements (data + gaps) in one sweep.

**Why process gaps in non-broadcast ops?** The gaps contain zeros (set by
`memset` at allocation time or padded in the ROM array). Running Relu over zeros
still produces zeros — correct. Running Add with a zero-padded partner and a
zero-padded output is also a no-op on the gaps. This avoids complicating
downstream nodes with chunk-by-chunk loops.

### Pool Size Calculation

`_compute_pool_bytes()` sums `align64(alloc × bytes_per_elem)` over all tensors,
then rounds up to a 4 KiB page boundary. This is the minimum contiguous DMA
region needed and is exposed as `INFERENCE_BUF_POOL_SIZE_BYTES`.

---

## 8. Step 5 — Code Generation (CodeGenerator)

`CodeGenerator` is assembled from several mixins using Python's MRO:

```python
class CodeGenerator(
    _HeaderMixin, _SourceMixin, _BufImplMixin,
    _SimulateMixin, _TestMixin, _CmakeMixin,
    _CoreMixin,           # __init__ must be last
):
    pass
```

All mixins share state through `self`:
- `self._graph` — the `OnnxGraph` instance
- `self._dtype` — the active `DataType` (default: `AP_FIXED_16_8`)
- `self._alloc_sizes` — computed by `_CoreMixin.__init__`
- `self._embed_large_weights`, `self._embed_large_expected` — CLI flags

### _SourceMixin — inference.c

The source file is assembled from sections:

```
_file_banner()                     ← auto-generated header with model info
_source_includes()                 ← #include "inference.h", xvectoropkernel.h, string.h
_source_op_defines()               ← #define VECTOROP_ADD 0u …
_weight_arrays()                   ← ROM arrays + DMA pointers for each weight
_buffer_declarations()             ← comments for I/O, DMA pointers for intermediates
_kernel_instance()                 ← static XVectoropkernel s_kernel;
_run_op_helper()                   ← static void run_op(…) and/or run_op_at(…)
[_load_weight_helper()]            ← only when large weights exist
_init_function()                   ← inference_init() + inference_deinit()
_inference_function()              ← inference_run()
```

### _SourceMixin — run_op() and run_op_at()

`run_op()` is emitted when any node uses a single flat kernel call
(`outer_count == 1`). `run_op_at()` is emitted when any node broadcasts
(`outer_count > 1`). Both may be emitted in the same file if the graph
contains a mix of regular and broadcast ops.

```c
/* run_op(): whole-buffer dispatch */
static void run_op(inference_buf_t *a, inference_buf_t *b,
                   inference_buf_t *c, unsigned size, unsigned op)
{
    XVectoropkernel_Set_a(&s_kernel, inference_buf_phys(a));
    XVectoropkernel_Set_b(&s_kernel, b ? inference_buf_phys(b) : (u64)0);
    XVectoropkernel_Set_c(&s_kernel, inference_buf_phys(c));
    XVectoropkernel_Set_size(&s_kernel, size);
    XVectoropkernel_Set_op(&s_kernel, op);
    XVectoropkernel_Start(&s_kernel);
    while (!XVectoropkernel_IsDone(&s_kernel)) {}
}

/* run_op_at(): offset-based dispatch for broadcasting loops */
static void run_op_at(inference_buf_t *a, unsigned a_off,
                      inference_buf_t *b, unsigned b_off,
                      inference_buf_t *c, unsigned c_off,
                      unsigned size, unsigned op)
{
    XVectoropkernel_Set_a(&s_kernel,
        inference_buf_phys(a) + (uint64_t)a_off * INFERENCE_BYTES_PER_ELEM);
    /* … same for b and c … */
    XVectoropkernel_Start(&s_kernel);
    while (!XVectoropkernel_IsDone(&s_kernel)) {}
}
```

Notice that `run_op()` passes **physical addresses** to the kernel registers.
The kernel's AXI master ports use these physical addresses to read/write DDR
directly. The CPU never sees these transfers — it only writes to the AXI-Lite
control registers and polls the done flag.

---

## 9. DataType Abstraction

`DataType` in `src/dtype.py` is an abstract base class that encapsulates
everything type-specific. Adding a new element type requires only subclassing
it; no changes to `nodes.py`, `graph.py`, or any codegen mixin are needed.

```python
class DataType(ABC):
    @property
    def bytes_per_elem(self) -> int: ...     # 2 for ap_fixed<16,8>, 4 for float32
    @property
    def align_elems(self) -> int: ...        # ALIGN_BYTES // bytes_per_elem
    @property
    def c_type(self) -> str: ...             # "uint16_t" or "float"
    @property
    def c_array_type(self) -> str: ...       # type for static ROM arrays
    @property
    def np_storage(self) -> np.dtype: ...    # numpy dtype matching raw DMA storage

    def quantize(self, x: np.ndarray) -> np.ndarray: ...     # float64 → float64 (rounded)
    def ramp_to_float(self, positions) -> np.ndarray: ...    # C test ramp → float64
    def float_to_storage(self, x) -> np.ndarray: ...         # float64 → raw storage dtype
    def encode_weight(self, data) -> List[str]: ...          # float array → C literal list
    def dat_bytes(self, data) -> bytes: ...                  # float array → binary .dat
    def c_display(self, ptr, idx) -> str: ...                # C printf display expression
    def c_fill_rhs(self, pos_expr) -> str: ...               # C ramp-fill RHS expression
    def format_literal(self, storage_val) -> str: ...        # single value → C literal
```

### ap_fixed<16,8> (default)

`ApFixed(W=16, I=8)` — 16-bit signed two's-complement, 8 integer bits, 8 fractional bits.

| Property | Value |
|----------|-------|
| Representable range | −128 to +127.99609375 |
| Quantization step | 1/256 ≈ 0.00390625 |
| Scale factor | 256 |
| Encoding of 1.0 | `0x0100` |
| Encoding of 0.5 | `0x0080` |
| Encoding of −1.0 | `0xFF00` (two's complement) |
| bytes_per_elem | 2 |
| align_elems | 8 (16 bytes / 2 bytes per elem) |

Quantization: `encoded = round(clip(x, -128, 127.996) * 256)` stored as `int16_t`.

### float32

`Float32()` — IEEE 754 single precision.

| Property | Value |
|----------|-------|
| bytes_per_elem | 4 |
| align_elems | 4 (16 bytes / 4 bytes per elem) |
| Quantization | Round-trip through `float32` (matches hardware float precision) |

### Ramp Fill Correspondence

The C test harness fills input buffers with:
```c
p[i] = (Data_t)(i & 0xFFFFu);
```

For `ap_fixed<16,8>` this creates the sequence:
```
p[0] = 0x0000  → 0.0
p[1] = 0x0001  → 1/256 ≈ 0.0039
p[2] = 0x0002  → 2/256 ≈ 0.0078
...
p[256] = 0x0100  → 1.0
p[384] = 0x0180  → 1.5
```

The Python `ramp_to_float(positions)` method replicates this exactly:
```python
uint_vals = (positions & 0xFFFF).astype(np.uint16)
int_vals  = uint_vals.view(np.int16)        # reinterpret as signed
return int_vals.astype(np.float64) / 256.0  # decode fixed-point
```

This bit-exact correspondence between the C ramp fill and the Python simulation
is how the scheduler guarantees that the embedded GT arrays in `test_inference.c`
always match what the hardware will produce.

---

## 10. Fixed-Point Simulation (_SimulateMixin)

The simulation in `src/codegen/_simulate.py` forward-passes the ONNX graph using
numpy, quantizing intermediate results at each node boundary to mimic the
hardware's element-wise saturate-and-round behavior.

### Simulation Flow

```
1. Seed weights: for each weight tensor, quantize(float_data) → simulated array
2. Seed inputs:  ramp_to_float(positions) → simulated input array
   (positions account for broadcast stride layout)
3. For each ScheduledNode in topological order:
   a. Fetch input arrays by onnx_name
   b. Apply numpy operation (a + b, a * b, np.maximum, etc.)
   c. quantize(result) → output array (clips + rounds to representable grid)
   d. Store as arrays[output.onnx_name]
4. Return all arrays (inputs, weights, intermediates, outputs)
```

### Quantization at Each Step

The hardware VectorOPKernel applies `saturate_cast<Data_t>` after every
element-wise operation. In `ap_fixed<16,8>` arithmetic:

```
add(0x7F00, 0x0100) = 0x8000  (128 + 0.5... but wait, 128 overflows)
→ saturate to max: 0x7FFF  (127.996)
```

The Python `quantize()` method mirrors this exactly:
```python
def quantize(self, x):
    clipped = np.clip(x.astype(np.float64), self._min_val, self._max_val)
    return np.round(clipped * self._scale) / self._scale
```

### Expected Storage Layout

After simulation, the expected output for each tensor is converted from logical
float64 values back into the raw DMA-buffer layout via `_expected_storage()`:

- **Non-broadcast tensors**: contiguous encoding, `dtype.float_to_storage(flat)`
- **Broadcast tensors**: strided layout with zero gaps — same structure as the
  weight ROM arrays and the C test ramp fill

The encoded storage array is then either:
- Embedded as a C array literal (`_emit_expected_c()`) for small tensors
- Written to `expected/<name>.dat` and loaded at runtime for large tensors

---

## 11. Cache Coherency Model

The KV260 has a Cortex-A53 CPU with L1/L2 data caches and a Xilinx FPGA PL
(Programmable Logic) fabric. They share DDR but **the PL is not cache-coherent
with the CPU caches** — the CPU's view of memory may be stale if the PL has
written to DDR, and vice versa.

Two operations maintain coherency:

| Function | Direction | When | Implementation |
|----------|-----------|------|----------------|
| `sync_to_device(buf)` | CPU cache → DDR | Before PL reads | Linux: `xclSyncBO(TO_DEVICE)` / Bare-metal: `Xil_DCacheFlushRange` |
| `sync_from_device(buf)` | DDR → CPU cache | After PL writes | Linux: `xclSyncBO(FROM_DEVICE)` / Bare-metal: `Xil_DCacheInvalidateRange` |

### Sync Policy in Generated Code

The scheduler places sync calls at the minimal required boundaries:

```
inference_init():
  for each weight tensor:
    memcpy(virtual_ptr, rom_data, ...)   ← CPU writes weight data
    sync_to_device(weight_buf)           ← flush once; weights never change

inference_run(X, Y):
  sync_to_device(X)                      ← flush user input(s) before first op
  [all kernel calls — no sync inside]
  sync_from_device(Y)                    ← invalidate output(s) after last op
```

**Internal buffers (intermediates) are never synced.** After the first kernel
writes an intermediate result to DDR, the next kernel reads it from DDR directly
via its AXI master port — the CPU cache is never involved in this data path.
Syncing intermediates would be wasteful and is architecturally unnecessary.

**Weights are synced once at init.** They are read-only after initialization, so
there is no need to flush them before every `inference_run()` call.

**`run_op()` and `run_op_at()` perform no sync.** They are pure dispatch helpers:
write registers, start, poll. The comment at the top of `run_op()` in the
generated code explicitly states this contract so callers know what to expect
if they ever call `run_op` manually.

---

## 12. Extending the System

### Adding a New ONNX Operator

The VectorOPKernel hardware currently supports 6 opcodes (0–5). If the hardware
is extended with a new opcode, add the mapping in `src/nodes.py`:

```python
OP_ABS  = 6   # hypothetical new op
OP_NAMES[OP_ABS] = "VECTOROP_ABS"
_ONNX_OP_MAP["Abs"] = (OP_ABS, 1)   # arity=1 (unary)
```

No other files need to change for a unary op. For a binary op, the broadcasting
logic in `_broadcast_info()` already handles the general case.

### Adding a New Data Type

Subclass `DataType` in `src/dtype.py`:

```python
class Int8(DataType):
    @property
    def name(self) -> str:      return "int8"
    @property
    def bytes_per_elem(self):   return 1
    @property
    def c_type(self):           return "int8_t"
    @property
    def c_array_type(self):     return "int8_t"
    @property
    def np_storage(self):       return np.int8

    def quantize(self, x):
        return np.clip(np.round(x), -128, 127).astype(np.float64)

    def ramp_to_float(self, positions):
        return (positions & 0xFF).astype(np.int8).astype(np.float64)

    def float_to_storage(self, x):
        return np.clip(np.round(x), -128, 127).astype(np.int8)

    def dat_bytes(self, data):
        return self.float_to_storage(data.flatten().astype(np.float64)).tobytes()

    def c_display(self, ptr, idx):
        return f"(double){ptr}[{idx}]"

    def c_fill_rhs(self, pos_expr):
        return f"(Data_t)({pos_expr} & 0xFFu)"

INT8: DataType = Int8()
```

Then pass it to `OnnxGraph` and `CodeGenerator`:

```python
from src.dtype import INT8
g  = OnnxGraph("model.onnx", dtype=INT8)
cg = CodeGenerator(g, "model.onnx", dtype=INT8)
```

The rest of the scheduler adapts automatically: `bytes_per_elem` controls
buffer sizing, `align_elems` controls alignment, and the C code emitters use
`c_type` for the `Data_t` typedef.

### Adding a New Generated File

Add a new mixin in `src/codegen/`:

```python
# src/codegen/_myfile.py
class _MyFileMixin:
    def generate_myfile(self) -> str:
        # self._graph, self._dtype, self._alloc_sizes all available
        return "/* auto-generated */\n"
```

Add it to `CodeGenerator` in `src/codegen/__init__.py`:

```python
from ._myfile import _MyFileMixin

class CodeGenerator(_HeaderMixin, _SourceMixin, ..., _MyFileMixin, _CoreMixin):
    pass
```

Call it in `inference_scheduler.py`:

```python
myfile = gen.generate_myfile()
_write(os.path.join(out_dir, "src", "myfile.c"), myfile)
```

### Handling a New ONNX Opset

If a future ONNX opset changes the encoding of a supported op (as happened with
`Clip` between opset 10 and 11), update `_get_clip_bounds()` or the relevant
extraction logic in `ScheduledNode.validate()`. All other files remain unchanged.

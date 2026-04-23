# CLAUDE.md — inference-scheduler

Python code-generator that parses an ONNX model and emits a complete C project
that runs inference on the Xilinx KV260 FPGA using the VectorOPKernel IP core.

## Quick Start

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate test ONNX models (required before running tests)
.venv/bin/python test/gen_test_models.py

# Run all tests
.venv/bin/python -m pytest test/ -v

# Generate a C inference project
.venv/bin/python inference_scheduler.py test/models/mixed_ops.onnx --out-dir /tmp/out
```

## CLI

```
python inference_scheduler.py <model.onnx> [options]

Options:
  --out-dir DIR              Output directory (default: ./<stem>_inference/)
  --driver-dir DIR           Copy XVectoropkernel driver sources from this path
  --embed-large-weights      Inline all weights as C arrays (skip .dat files)
  --embed-large-expected     Inline all GT arrays in test_inference.c
```

## Generated Project Layout

```
<out_dir>/
├── CMakeLists.txt            INFERENCE_TARGET=BARE_METAL|LINUX
├── include/inference.h       Public API: Data_t, size macros, init/run declarations
├── src/
│   ├── inference.c           Weight ROM arrays, run_op() helper, init/run bodies
│   └── inference_buf.c       DMA buffer alloc/sync (Linux XRT or bare-metal Xil)
├── test/test_inference.c     On-device test: ramp fill → run → compare vs GT
├── scripts/check_inference_setup.sh
├── driver/                   XVectoropkernel sources (copied or stub README)
├── weights/                  External .dat files for large weight tensors (> 4096 elems)
└── expected/                 External .dat files for large GT expected arrays (> 4096 elems)
```

## Source Layout

```
inference_scheduler.py   CLI entry point
requirements.txt
src/
  dtype.py               DataType abstraction (ap_fixed<W,I>, float32)
  layout.py              TensorLayout frozen dataclass (numel, alloc, n_chunks, chunk, stride)
  tensor.py              TensorInfo: metadata + C declaration emitters
  nodes.py               ScheduledNode: ONNX op → VectorOPKernel call
  graph.py               OnnxGraph: ONNX parsing, shape inference, tensor registry
  codegen/
    __init__.py          CodeGenerator (assembles all mixins)
    _core.py             _compute_tensor_layouts() → TensorLayout; large-tensor lists, pool sizing
    _header.py           generate_header()  → include/inference.h
    _source.py           generate_source()  → src/inference.c
    _buf_impl.py         generate_buf_impl() → src/inference_buf.c
    _simulate.py         Fixed-point forward simulation; generate_expected_dat()
    _test.py             generate_test()    → test/test_inference.c
    _cmake.py            generate_cmake()   → CMakeLists.txt
    _banners.py          File-header banner helpers
test/
  gen_test_models.py     Build all test ONNX models
  helpers.py             _model(), _models_exist() shared by test modules
  models/                Pre-generated ONNX models (single_add.onnx, etc.)
  test_*.py              pytest test modules (591 tests total)
```

## Key Abstractions

### DataType (`src/dtype.py`)

Encapsulates everything type-specific. The rest of the codebase is type-agnostic.

```python
from src.dtype import AP_FIXED_16_8, FLOAT32, ApFixed

g  = OnnxGraph("model.onnx", dtype=AP_FIXED_16_8)   # default
cg = CodeGenerator(g, "model.onnx", dtype=AP_FIXED_16_8)
```

| Type | `name` | `bytes_per_elem` | `align_elems` | Range |
|------|--------|-----------------|----------------|-------|
| `AP_FIXED_16_8` | `ap_fixed<16,8>` | 2 | 8 | [-128, 127.996] |
| `FLOAT32` | `float32` | 4 | 4 | IEEE 754 |

Key methods:
- `quantize(x)` — round float64 array to representable grid (used in simulation)
- `encode_weight(data)` → list of C literal strings (`"0x0100"`)
- `float_to_storage(x)` → numpy array with `np_storage` dtype
- `dat_bytes(data)` → little-endian bytes for external `.dat` files
- `c_display(ptr, idx)` → C expression for `printf("%.4f")` display
- `c_fill_rhs(pos_expr)` → RHS of test ramp-fill assignment

Adding a new type: subclass `DataType`, implement all abstract methods, pass
the instance to `OnnxGraph` and `CodeGenerator`.

### ScheduledNode (`src/nodes.py`)

Maps one ONNX operator to one or more VectorOPKernel invocations.

Supported ops and opcodes:

| ONNX op | Opcode | Arity | Notes |
|---------|--------|-------|-------|
| `Add` | `OP_ADD` (0) | binary | |
| `Sub` | `OP_SUB` (1) | binary | |
| `Mul` | `OP_MUL` (2) | binary | |
| `Div` | `OP_DIV` (3) | binary | |
| `Relu` | `OP_RELU` (4) | unary | b=NULL |
| `Clip(min=0,max=6)` | `OP_RELU6` (5) | unary | exact bounds required |
| `MatMul` | — | binary | dispatched to MatmulKernel |

**Broadcasting**: One input per binary op may broadcast. Rules:
- Right-align input shape to output shape
- All broadcast dimensions (size 1) must form a **contiguous leading block**
- Valid: `[1, 1, 64]` broadcasts to `[4, 32, 64]` (dims 0,1 broadcast)
- Invalid: `[4, 1, 64]` to `[4, 32, 64]` (broadcast dim between matching dims)

When broadcasting, `emit_call()` generates a for-loop calling `run_op_at()`:
```c
for (unsigned _i = 0u; _i < 4u; _i++) {
    run_op_at(X, _i * INFERENCE_Y_CHUNK_STRIDE, bias, 0u,
              Y, _i * INFERENCE_Y_CHUNK_STRIDE, INFERENCE_Y_CHUNK, VECTOROP_ADD);
}
```

### CodeGenerator (`src/codegen/`)

Multi-mixin class. `_CoreMixin.__init__` computes padded allocation sizes for all
tensors accounting for broadcast alignment gaps. All other mixins read `self._alloc_sizes`.

Allocation rules (`_compute_tensor_layouts` → `TensorLayout`):
- Phase 1: all tensors seeded as `TensorLayout.flat(numel)`
- Phase 2: broadcast VectorOP nodes set advancing/repeating layouts with alignment-padded strides
- Phase 3: layout propagates forward through non-broadcast, non-MatmulNode chains
- MatmulNode reads row strides directly from `TensorLayout.gap` at emit time

### Cache Coherency Model

The kernel's AXI master reads/writes DDR using **physical addresses** programmed
into AXI-Lite registers. CPU cache must be explicitly managed:

- `inference_buf_sync_to_device(buf)` — flush CPU cache → DDR (before kernel reads)
- `inference_buf_sync_from_device(buf)` — invalidate CPU cache (after kernel writes)

**Contract**:
- `inference_init()`: syncs each weight buffer **once** after `memcpy` from ROM
- `inference_run()`: syncs all graph **inputs** at the top, all graph **outputs** at the bottom
- `run_op()`: performs **no sync** — pure AXI-Lite register writes + poll
- Internal kernel-to-kernel buffers (intermediates): **no sync ever needed**

### Large Tensor Handling

Tensors exceeding the threshold are written to external binary `.dat` files and
loaded at runtime via `fread()`:

| Threshold | Constant | Files |
|-----------|----------|-------|
| `LARGE_WEIGHT_THRESHOLD = 4096` | in `tensor.py` | `weights/<c_name>.dat` |
| `LARGE_EXPECTED_THRESHOLD = 4096` | in `codegen/_simulate.py` | `expected/<c_name>.dat` |

Both `.dat` files contain little-endian elements in the strided DMA-buffer layout
(alignment gaps are zero-filled for broadcast tensors).

## Generated C API

```c
// inference.h
typedef uint16_t Data_t;              // ap_fixed<16,8>
#define INFERENCE_BYTES_PER_ELEM  2u
#define INFERENCE_ALIGN_BYTES     16u
#define INFERENCE_ALIGN_ELEMS     8u
#define INFERENCE_INPUT_SIZE      N   // alloc size for graph input(s)
#define INFERENCE_OUTPUT_SIZE     N   // alloc size for graph output(s)

// For broadcast nodes only:
#define INFERENCE_<TENSOR>_CHUNK        chunk_size
#define INFERENCE_<TENSOR>_CHUNK_STRIDE INFERENCE_ALIGN_UP(chunk_size, align)

int  inference_init(const char *instance_name);  // alloc DMA bufs, load weights
void inference_run(inference_buf_t *in, inference_buf_t *out);
// signature lists all graph inputs then all graph outputs;
// models may have multiple of each.
void inference_deinit(void);

// inference_buf.c (platform-specific)
inference_buf_t *inference_buf_alloc(unsigned size_elements);
void             inference_buf_free(inference_buf_t *buf);
void            *inference_buf_ptr(inference_buf_t *buf);   // virtual address (CPU)
uint64_t         inference_buf_phys(inference_buf_t *buf);  // physical address (AXI)
void             inference_buf_sync_to_device(inference_buf_t *buf);
void             inference_buf_sync_from_device(inference_buf_t *buf);
```

## Testing

Tests live in `test/`. Run with `pytest`:

```bash
.venv/bin/python -m pytest test/ -v                        # all tests
.venv/bin/python -m pytest test/test_source.py -v          # generated inference.c
.venv/bin/python -m pytest test/test_broadcast.py -v       # broadcast logic
.venv/bin/python -m pytest test/ -k "test_relu" -v         # filter by name
```

Most test classes are decorated `@unittest.skipUnless(_models_exist(), ...)` —
run `test/gen_test_models.py` first if tests are skipped.

## Driver Sources

The XVectoropkernel driver is generated by Vitis HLS synthesis. When
`--driver-dir` is omitted, `driver/` is left empty with a `README.md`.
Required files: `xvectoropkernel.h`, `xvectoropkernel_hw.h`,
`xvectoropkernel.c`, `xvectoropkernel_sinit.c`, `xvectoropkernel_linux.c`.

Source path after synthesis:
```
<axi_demo>/build/kv260/vadd_kv260/solution1/impl/ip/drivers/VectorOPKernel_v1_0/src/
```

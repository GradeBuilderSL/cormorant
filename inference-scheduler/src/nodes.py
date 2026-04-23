"""
Supported ONNX operators and their mapping to VectorOPKernel op codes,
plus MatmulNode for the separate MatmulKernel IP.

Each ScheduledNode wraps one onnx.NodeProto, holds resolved TensorInfo
references for its inputs/outputs, and knows how to emit the corresponding
run_op() / run_op_at() call in the generated C file.

MatmulNode wraps a MatMul ONNX op and emits run_matmul() calls for
the XMatmulkernel driver API.

Supported ONNX ops (VectorOPKernel)
------------------------------------
  Add                 → OP_ADD   (binary)
  Sub                 → OP_SUB   (binary)
  Mul                 → OP_MUL   (binary)
  Div                 → OP_DIV   (binary)
  Relu                → OP_RELU  (unary)
  Clip(min=0, max=6)  → OP_RELU6 (unary)

Supported ONNX ops (MatmulKernel)
----------------------------------
  MatMul              → run_matmul()  (2-D and batched 3-D)

All other ops raise SchedulerError.

Multidirectional broadcasting
------------------------------
Binary ops support partial ONNX multidirectional broadcasting: one input
may be smaller than the output, provided its shape right-aligns to the
output and any broadcasted (size-1 or absent) dimensions form a contiguous
leading block (no interleaved matching and broadcast dims).

When broadcasting is detected the node emits a C for-loop calling
run_op_at() (offset-based, no internal sync) instead of a single run_op()
call.  The chunk stride is rounded up to INFERENCE_ALIGN_BYTES so every
iteration starts at a 16-byte-aligned physical address.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import onnx

from .tensor import TensorInfo


class SchedulerError(Exception):
    """Raised when the ONNX graph cannot be compiled to VectorOPKernel calls."""


# ------------------------------------------------------------------ #
# Op-code constants (must match VectorOP.h)                          #
# ------------------------------------------------------------------ #

OP_ADD   = 0
OP_SUB   = 1
OP_MUL   = 2
OP_DIV   = 3
OP_RELU  = 4
OP_RELU6 = 5

# Human-readable names for comments
OP_NAMES = {
    OP_ADD:   "VECTOROP_ADD",
    OP_SUB:   "VECTOROP_SUB",
    OP_MUL:   "VECTOROP_MUL",
    OP_DIV:   "VECTOROP_DIV",
    OP_RELU:  "VECTOROP_RELU",
    OP_RELU6: "VECTOROP_RELU6",
}

# ONNX op_type → (op_code, arity)
# arity 2 = binary (reads a and b), arity 1 = unary (reads a only)
_ONNX_OP_MAP = {
    "Add":  (OP_ADD,   2),
    "Sub":  (OP_SUB,   2),
    "Mul":  (OP_MUL,   2),
    "Div":  (OP_DIV,   2),
    "Relu": (OP_RELU,  1),
    # Clip is matched by name but validated for (0,6) attributes below
    "Clip": (OP_RELU6, 1),
}

# ------------------------------------------------------------------ #
# Alignment constants                                                 #
#                                                                     #
# _ALIGN_BYTES is a hardware requirement (AXI burst boundary).       #
# _BYTES_PER_ELEM must match INFERENCE_BYTES_PER_ELEM in the         #
# generated inference.h — change both together when the data type    #
# changes (e.g. ap_fixed<16,8> → float32 means 2 → 4).              #
# _ALIGN_ELEMS is derived and must not be hardcoded elsewhere.       #
# ------------------------------------------------------------------ #

_ALIGN_BYTES = 16   # AXI burst alignment requirement (hardware-fixed)
# _BYTES_PER_ELEM and _ALIGN_ELEMS are data-type-dependent.
# They are injected per-node via ScheduledNode.from_onnx_node(align_elems=...)
# rather than kept as module-level constants.


# ------------------------------------------------------------------ #
# Clip attribute extraction                                           #
# ------------------------------------------------------------------ #

def _get_clip_bounds(node: onnx.NodeProto) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract Clip min/max from either:
      - Opset < 11: stored as attributes 'min' and 'max'
      - Opset >= 11: stored as optional input tensors (indices 1 and 2)
    Returns (min_val, max_val); None means the bound is not present.
    """
    attrs = {a.name: a for a in node.attribute}
    if "min" in attrs or "max" in attrs:
        mn = attrs["min"].f if "min" in attrs else None
        mx = attrs["max"].f if "max" in attrs else None
        return mn, mx
    return None, None


# ------------------------------------------------------------------ #
# Broadcasting helper                                                 #
# ------------------------------------------------------------------ #

def _broadcast_info(
    t: TensorInfo, output: TensorInfo, align_elems: int = 8
) -> Tuple[int, int, int, bool]:
    """
    Compute how tensor *t* broadcasts to *output*.

    Returns (outer_count, chunk_size, aligned_chunk_size, advances):

      outer_count         Number of loop iterations.  1 = no broadcast
                          (t covers the whole output in one call).
      chunk_size          t.numel — data elements in one chunk.
      aligned_chunk_size  chunk_size rounded up to align_elems so every
                          loop iteration starts at an INFERENCE_ALIGN_BYTES-
                          aligned physical address.  May be > chunk_size,
                          producing gap elements between data blocks.
      advances            True  → t strides through the output (outer_count=1,
                                  or t.numel == output.numel).
                          False → t repeats at offset 0 each iteration
                                  (t.numel < output.numel).

    Raises SchedulerError if t's shape is incompatible with broadcasting
    to output under the trailing-contiguous constraint.
    """
    if t.numel == output.numel:
        # Exact match — no broadcasting, single call covers everything.
        return (1, output.numel, output.numel, True)

    if output.numel % t.numel != 0:
        raise SchedulerError(
            f"Tensor '{t.onnx_name}' (numel={t.numel}) is not a factor of "
            f"output '{output.onnx_name}' (numel={output.numel}); "
            f"cannot broadcast."
        )

    outer_count = output.numel // t.numel
    chunk_size  = t.numel

    # Right-align t.shape to output.shape, padding missing leading dims with 1.
    pad       = len(output.shape) - len(t.shape)
    t_aligned = [1] * pad + list(t.shape)

    # Trailing-contiguous constraint:
    # After right-alignment each t dim must be either:
    #   - Equal to the corresponding output dim  (matching)
    #   - 1                                      (broadcast)
    # AND all broadcast dims (1s) must form a contiguous LEADING block —
    # no matching dim may appear before a broadcast dim.
    found_match = False
    for td, od in zip(t_aligned, output.shape):
        if td == od:
            found_match = True
        elif td == 1:
            if found_match:
                raise SchedulerError(
                    f"Tensor '{t.onnx_name}' shape={t.shape} cannot be broadcast "
                    f"to output '{output.onnx_name}' shape={output.shape}: "
                    f"broadcast dimensions must form a contiguous leading block "
                    f"(found a broadcast dim after a matching dim)."
                )
        else:
            raise SchedulerError(
                f"Tensor '{t.onnx_name}' shape={t.shape} cannot be broadcast "
                f"to output '{output.onnx_name}' shape={output.shape}: "
                f"dimension {td} does not match output dimension {od} and is not 1."
            )

    # Round chunk up to align_elems so each iteration's physical start address
    # is INFERENCE_ALIGN_BYTES-aligned.  The extra elements are gap/padding and
    # are never read or written by VectorOPKernel.
    aligned_chunk_size = (chunk_size + align_elems - 1) & ~(align_elems - 1)

    return (outer_count, chunk_size, aligned_chunk_size, False)


# ------------------------------------------------------------------ #
# ScheduledNode                                                       #
# ------------------------------------------------------------------ #

@dataclass
class ScheduledNode:
    """One ONNX operator mapped to one or more VectorOPKernel invocations."""

    kernel_name: ClassVar[str] = "VectorOPKernel"

    onnx_node:   onnx.NodeProto
    op_code:     int                   # OP_ADD … OP_RELU6
    arity:       int                   # 1 = unary, 2 = binary
    inputs:      List[TensorInfo]      # resolved input tensors
    output:      TensorInfo            # single output tensor
    index:       int = 0               # sequential index in graph
    align_elems: int = 8               # elements per AXI alignment boundary

    # ------------------------------------------------------------------ #
    # Broadcast fields — populated by validate()                          #
    # ------------------------------------------------------------------ #

    # outer_count > 1 means broadcasting: the kernel is invoked outer_count
    # times, each processing chunk_size elements at an aligned stride.
    outer_count:        int  = field(default=1,    init=False)
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    # Which inputs stride through the output (True) vs repeat at offset 0 (False).
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,           # name → TensorInfo
        index:       int,
        align_elems: int = 8,
    ) -> "ScheduledNode":
        op_type = node.op_type

        if op_type not in _ONNX_OP_MAP:
            raise SchedulerError(
                f"Node '{node.name or op_type}' (op_type='{op_type}') is not "
                f"supported by VectorOPKernel.\n"
                f"Supported ops: {sorted(_ONNX_OP_MAP)}"
            )

        op_code, arity = _ONNX_OP_MAP[op_type]

        # Resolve input tensors (skip empty optional inputs)
        inputs = []
        for inp_name in node.input:
            if inp_name == "":
                continue
            if inp_name not in tensors:
                raise SchedulerError(
                    f"Tensor '{inp_name}' referenced by node "
                    f"'{node.name or op_type}' was not found in the graph."
                )
            inputs.append(tensors[inp_name])

        # Resolve output tensor
        if not node.output or node.output[0] == "":
            raise SchedulerError(
                f"Node '{node.name or op_type}' has no output tensor."
            )
        out_name = node.output[0]
        if out_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{out_name}' of node '{node.name or op_type}' "
                f"was not found in the graph."
            )
        output = tensors[out_name]

        sn = cls(
            onnx_node=node,
            op_code=op_code,
            arity=arity,
            inputs=inputs,
            output=output,
            index=index,
            align_elems=align_elems,
        )
        sn.validate()
        return sn

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        op_type = self.onnx_node.op_type

        # Clip: only allow (min=0, max=6)
        if op_type == "Clip":
            mn, mx = _get_clip_bounds(self.onnx_node)
            if mn is None and len(self.inputs) > 1:
                mn_tensor = self.inputs[1]
                if mn_tensor.is_weight and mn_tensor.data is not None:
                    mn = float(mn_tensor.data.flat[0])
            if mx is None and len(self.inputs) > 2:
                mx_tensor = self.inputs[2]
                if mx_tensor.is_weight and mx_tensor.data is not None:
                    mx = float(mx_tensor.data.flat[0])

            if mn != 0.0 or mx != 6.0:
                raise SchedulerError(
                    f"Clip node '{self.onnx_node.name}' has bounds "
                    f"(min={mn}, max={mx}). Only Clip(0, 6) maps to "
                    f"VECTOROP_RELU6."
                )
            self.inputs = [self.inputs[0]]
            self.arity  = 1

        # Binary ops need exactly 2 inputs; unary need exactly 1
        n_data = len(self.inputs)
        if self.arity == 2 and n_data != 2:
            raise SchedulerError(
                f"Binary op '{op_type}' in node '{self.onnx_node.name}' "
                f"requires 2 inputs, got {n_data}."
            )
        if self.arity == 1 and n_data != 1:
            raise SchedulerError(
                f"Unary op '{op_type}' in node '{self.onnx_node.name}' "
                f"requires 1 input, got {n_data}."
            )

        if self.arity == 1:
            # Unary ops: input must match output exactly — no broadcasting.
            t = self.inputs[0]
            if t.numel != self.output.numel:
                raise SchedulerError(
                    f"Shape mismatch in unary node "
                    f"'{self.onnx_node.name or op_type}': "
                    f"input '{t.onnx_name}' has {t.numel} elements but output "
                    f"'{self.output.onnx_name}' has {self.output.numel} elements."
                )
            self.outer_count        = 1
            self.chunk_size         = self.output.numel
            self.aligned_chunk_size = self.output.numel
            self.a_advances         = True
            self.b_advances         = True

        else:
            # Binary ops: check each input for trailing-contiguous broadcast.
            ae = self.align_elems
            a_outer, _, _, a_advances = _broadcast_info(self.inputs[0], self.output, ae)
            b_outer, _, _, b_advances = _broadcast_info(self.inputs[1], self.output, ae)

            if not a_advances and not b_advances:
                raise SchedulerError(
                    f"Binary node '{self.onnx_node.name or op_type}': "
                    f"both inputs are smaller than the output — "
                    f"only one input may broadcast per operation."
                )

            outer_count = max(a_outer, b_outer)
            chunk_size  = self.output.numel // outer_count
            aligned_chunk_size = (
                (chunk_size + ae - 1) & ~(ae - 1)
            )

            self.outer_count        = outer_count
            self.chunk_size         = chunk_size
            self.aligned_chunk_size = aligned_chunk_size
            self.a_advances         = a_advances
            self.b_advances         = b_advances

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        in_names  = ", ".join(t.onnx_name for t in self.inputs)
        broadcast = (
            f"  (broadcast \u00d7{self.outer_count})"
            if self.outer_count > 1 else ""
        )
        return (
            f"    /* [{self.index}] {self.onnx_node.op_type}"
            f"({in_names}) -> {self.output.onnx_name}"
            f"  shape={self.output.shape}{broadcast} */"
        )

    def emit_call(self, op_size: Optional[int] = None) -> str:
        """
        Emit the C kernel call(s) for this node.

        Cache sync is handled entirely by the caller (inference_run):
          - sync_to_device for graph inputs at the top of inference_run()
          - sync_from_device for graph outputs at the bottom of inference_run()
          - weights were flushed once in inference_init()
          - internal (kernel-to-kernel) buffers need no sync

        op_size overrides chunk_size for the non-broadcast case when the
        output buffer is larger than numel due to upstream alignment padding.
        """
        a  = self.inputs[0].c_name
        b  = self.inputs[1].c_name if self.arity == 2 else "NULL"
        c  = self.output.c_name
        op = OP_NAMES[self.op_code]

        if self.outer_count == 1:
            size = op_size if op_size is not None else self.chunk_size
            return f"    run_op({a}, {b}, {c}, {size}u, {op});"

        # Broadcasting: loop over outer_count chunks with run_op_at().
        # The chunk stride uses the CHUNK_STRIDE macro (INFERENCE_ALIGN_UP of
        # CHUNK) so every iteration starts at an INFERENCE_ALIGN_BYTES-aligned
        # physical address.  Gap elements between data blocks are never touched
        # by VectorOPKernel (it receives the exact chunk_size as 'size').
        chunk_macro  = f"INFERENCE_{c.upper()}_CHUNK"
        stride_macro = f"INFERENCE_{c.upper()}_CHUNK_STRIDE"
        n     = self.outer_count
        a_off = f"_i * {stride_macro}" if self.a_advances else "0u"
        b_off = (f"_i * {stride_macro}" if self.b_advances else "0u") \
                if self.arity == 2 else "0u"
        c_off = f"_i * {stride_macro}"

        return "\n".join([
            f"    for (unsigned _i = 0u; _i < {n}u; _i++) {{",
            f"        run_op_at({a}, {a_off},"
            f" {b}, {b_off},"
            f" {c}, {c_off},"
            f" {chunk_macro}, {op});",
            "    }",
        ])


# ------------------------------------------------------------------ #
# MatmulNode                                                           #
# ------------------------------------------------------------------ #

@dataclass
class MatmulNode:
    """One ONNX MatMul operator mapped to one or more XMatmulkernel invocations.

    kernel_name identifies this node's hardware kernel for the registry lookup.

    Supports four shapes:

      2-D   A[N,K]        @ B[K,M]           → Y[N,M]
      3-D   A[b,N,K]      @ B[b or 1,K,M]    → Y[b,N,M]   (B may broadcast)
      4-D   A[b1,b2,N,K]  @ B[b1,b2,K,M]     → Y[b1,b2,N,M]   (flat batch)
      4D×3D A[b1,b2,N,K]  @ B[b2,K,M]        → Y[b1,b2,N,M]   (outer loop)

    The 4D×3D case uses an outer loop of b1 calls to XMatmulkernel (each with
    batch=b2), emitting run_matmul_at() with element offsets per iteration.

    Fields
    ------
    n, k, m, batch          inner matrix dimensions and inner batch count
    a/b/c_batch_stride      element stride inside each XMatmulkernel call
    outer_count             outer loop iterations (1 for 2-D/3-D/flat-4-D)
    a/b/c_outer_stride      element offset advance per outer iteration
                            (b_outer_stride == 0 when B has no outer dim)

    Compatibility fields (always fixed):
    chunk_size, aligned_chunk_size, a_advances, b_advances, arity
    """

    kernel_name: ClassVar[str] = "MatmulKernel"

    onnx_node:      onnx.NodeProto
    inputs:         List[TensorInfo]   # [A, B]
    output:         TensorInfo
    index:          int = 0
    align_elems:    int = 8

    # Inner matrix dimensions (used in every kernel call)
    n:              int = 0   # output rows        (A.shape[-2])
    k:              int = 0   # inner dimension    (A.shape[-1] == B.shape[-2])
    m:              int = 0   # output columns     (B.shape[-1])
    batch:          int = 1   # inner batch count  (1 for 2-D; b for 3-D/4-D)
    a_batch_stride: int = 0   # A elements between consecutive inner-batch slices
    b_batch_stride: int = 0   # B elements between consecutive inner-batch slices
    c_batch_stride: int = 0   # Y elements between consecutive inner-batch slices

    # Outer loop parameters (set when A has a leading batch dim absent from B)
    outer_count:    int = 1   # outer loop iterations; 1 means a single call
    a_outer_stride: int = 0   # A elements advanced per outer iteration
    b_outer_stride: int = 0   # B elements advanced per outer iteration (0 if B broadcasts)
    c_outer_stride: int = 0   # Y elements advanced per outer iteration

    # Compatibility shims for _compute_alloc_sizes / _broadcast_io_map.
    # These are always derived constants — never set by callers.
    chunk_size:         int  = field(default=0,    init=False)
    aligned_chunk_size: int  = field(default=0,    init=False)
    a_advances:         bool = field(default=True, init=False)
    b_advances:         bool = field(default=True, init=False)
    arity:              int  = field(default=2,    init=False)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_onnx_node(
        cls,
        node:        onnx.NodeProto,
        tensors:     dict,
        index:       int,
        align_elems: int = 8,
    ) -> "MatmulNode":
        if node.op_type != "MatMul":
            raise SchedulerError(
                f"MatmulNode.from_onnx_node() called with op_type='{node.op_type}'"
            )

        # Resolve inputs
        if len(node.input) != 2:
            raise SchedulerError(
                f"MatMul node '{node.name}' must have exactly 2 inputs, "
                f"got {len(node.input)}."
            )
        a_name, b_name = node.input[0], node.input[1]
        for tname in (a_name, b_name):
            if tname not in tensors:
                raise SchedulerError(
                    f"Tensor '{tname}' referenced by MatMul node "
                    f"'{node.name}' not found in graph."
                )
        a_info = tensors[a_name]
        b_info = tensors[b_name]

        # Resolve output
        if not node.output or node.output[0] == "":
            raise SchedulerError(f"MatMul node '{node.name}' has no output.")
        out_name = node.output[0]
        if out_name not in tensors:
            raise SchedulerError(
                f"Output tensor '{out_name}' of MatMul node '{node.name}' "
                f"not found in graph."
            )
        out_info = tensors[out_name]

        # Derive n, k, m, batch, and strides from shapes using ONNX
        # numpy-style batched matmul broadcasting.
        a_shape = tuple(a_info.shape)
        b_shape = tuple(b_info.shape)

        a_rank = len(a_shape)
        b_rank = len(b_shape)

        if a_rank < 2 or b_rank < 2 or a_rank > 5 or b_rank > 5:
            raise SchedulerError(
                f"MatMul node '{node.name}': rank 2–5 supported, "
                f"got A{list(a_shape)} @ B{list(b_shape)}."
            )

        n_val = a_shape[-2]
        k_val = a_shape[-1]
        m_val = b_shape[-1]
        if b_shape[-2] != k_val:
            raise SchedulerError(
                f"MatMul shape mismatch in node '{node.name}': "
                f"A K={k_val} != B K={b_shape[-2]} "
                f"in A{list(a_shape)} @ B{list(b_shape)}."
            )

        a_batch = a_shape[:-2]   # leading batch dims of A; () when A is 2-D
        b_batch = b_shape[:-2]   # leading batch dims of B; () when B is 2-D

        # ---- Special cases: one input is 2-D (no batch dims) -------------
        # The 2-D side broadcasts across all batch dimensions of the other.
        # The kernel's batch loop handles this with a batch stride of 0 for
        # the 2-D operand; no outer loop is needed.

        if len(a_batch) == 0 and len(b_batch) == 0:
            # A[N,K] @ B[K,M] — plain matrix multiply
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=1, a_batch_stride=0, b_batch_stride=0, c_batch_stride=0,
            )

        if len(b_batch) == 0:
            # A[*batch,N,K] @ B[K,M] — B broadcasts across all of A's batch dims
            batch = 1
            for d in a_batch:
                batch *= d
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=n_val * k_val,
                b_batch_stride=0,
                c_batch_stride=n_val * m_val,
            )

        if len(a_batch) == 0:
            # A[N,K] @ B[*batch,K,M] — A broadcasts across all of B's batch dims
            batch = 1
            for d in b_batch:
                batch *= d
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=0,
                b_batch_stride=k_val * m_val,
                c_batch_stride=n_val * m_val,
            )

        # ---- General case: both inputs have batch dims -------------------
        # Right-align batch dimensions and apply ONNX broadcast rules.
        #
        # Batch dims are split into two contiguous blocks:
        #
        #   outer block (leading)  — exactly one input has size 1; the other
        #                advances element-by-element, driving an outer loop.
        #                All outer dims must be from the SAME input.
        #
        #   shared block (trailing) — both inputs have the same (non-1) size.
        #                The kernel's built-in batch loop iterates this block.
        #
        # outer_count = product of leading (outer) output dims
        # batch       = product of trailing (shared) output dims
        #
        # outer_count == 1 → emit run_matmul()     (no outer loop)
        # outer_count >  1 → emit a for-loop calling run_matmul_at()

        max_len = max(len(a_batch), len(b_batch))
        a_ext   = (1,) * (max_len - len(a_batch)) + a_batch
        b_ext   = (1,) * (max_len - len(b_batch)) + b_batch

        # Validate broadcastability and compute the output batch shape.
        out_batch: list = []
        for ad, bd in zip(a_ext, b_ext):
            if ad == bd:
                out_batch.append(ad)
            elif ad == 1:
                out_batch.append(bd)
            elif bd == 1:
                out_batch.append(ad)
            else:
                raise SchedulerError(
                    f"MatMul '{node.name}': batch dims {list(a_batch)} and "
                    f"{list(b_batch)} are not broadcastable."
                )

        # Find the split between the outer block and the shared block.
        # outer_a_advances: True  → A is the advancing side (b_ext has 1s)
        #                   False → B is the advancing side (a_ext has 1s)
        outer_a_advances: Optional[bool] = None
        split = max_len   # tentative: all dims are outer

        for i, (ad, bd) in enumerate(zip(a_ext, b_ext)):
            if ad > 1 and bd > 1:
                # Both present → shared block starts here
                split = i
                break
            if ad == 1 and bd == 1:
                # Degenerate size-1 on both sides → treat as shared
                split = i
                break
            # Exactly one is 1 → outer broadcast dim
            advances = (ad > 1)
            if outer_a_advances is None:
                outer_a_advances = advances
            elif outer_a_advances != advances:
                raise SchedulerError(
                    f"MatMul '{node.name}': mixed-direction outer broadcast "
                    f"in batch dims {list(a_batch)} vs {list(b_batch)} "
                    f"is not supported."
                )

        outer_count = 1
        for d in out_batch[:split]:
            outer_count *= d

        batch = 1
        for d in out_batch[split:]:
            batch *= d

        if batch <= 1:
            a_batch_stride = 0
            b_batch_stride = 0
            c_batch_stride = 0
        else:
            a_batch_stride = (n_val * k_val
                              if any(a_ext[i] > 1 for i in range(split, max_len))
                              else 0)
            b_batch_stride = (k_val * m_val
                              if any(b_ext[i] > 1 for i in range(split, max_len))
                              else 0)
            c_batch_stride = n_val * m_val

        if outer_count <= 1:
            return cls(
                onnx_node=node, inputs=[a_info, b_info], output=out_info,
                index=index, align_elems=align_elems,
                n=n_val, k=k_val, m=m_val,
                batch=batch,
                a_batch_stride=a_batch_stride,
                b_batch_stride=b_batch_stride,
                c_batch_stride=c_batch_stride,
            )

        # Outer loop — the advancing side steps through outer-count blocks;
        # the other side repeats at offset 0 every iteration.
        if outer_a_advances:
            a_outer_stride = batch * n_val * k_val
            b_outer_stride = 0
        else:
            a_outer_stride = 0
            b_outer_stride = batch * k_val * m_val
        c_outer_stride = batch * n_val * m_val

        return cls(
            onnx_node=node, inputs=[a_info, b_info], output=out_info,
            index=index, align_elems=align_elems,
            n=n_val, k=k_val, m=m_val,
            batch=batch,
            a_batch_stride=a_batch_stride,
            b_batch_stride=b_batch_stride,
            c_batch_stride=c_batch_stride,
            outer_count=outer_count,
            a_outer_stride=a_outer_stride,
            b_outer_stride=b_outer_stride,
            c_outer_stride=c_outer_stride,
        )

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        a = self.inputs[0].onnx_name
        b = self.inputs[1].onnx_name
        batch_str = f", batch={self.batch}" if self.batch > 1 else ""
        outer_str = (
            f"  (outer_loop\u00d7{self.outer_count})"
            if self.outer_count > 1 else ""
        )
        return (
            f"    /* [{self.index}] MatMul({a}, {b}) -> {self.output.onnx_name}"
            f"  [{self.n},{self.k}]x[{self.k},{self.m}]->[{self.n},{self.m}]"
            f"{batch_str}{outer_str} */"
        )

    def emit_call(self, op_size: Optional[int] = None,
                  a_row_stride: int = 0, c_row_stride: int = 0) -> str:
        """Emit the run_matmul() or run_matmul_at() call for this node.

        op_size is accepted for interface compatibility but ignored.

        a_row_stride / c_row_stride are non-zero when the A input or Y output
        DMA buffer has alignment-gap padding (e.g. A was produced by a
        broadcast VectorOP, or Y feeds one downstream).  In that case the
        kernel is called with batch=N, n=1 so that the batch stride acts as
        a per-row stride, allowing the kernel to skip alignment gaps.
        """
        a = self.inputs[0].c_name
        b = self.inputs[1].c_name
        c = self.output.c_name

        if self.outer_count == 1:
            if a_row_stride != 0 or c_row_stride != 0:
                # Row-strided decomposition: N calls of (1 × K) × (K × M).
                # a_batch_stride = a_row_stride (or natural K if only c is strided)
                # c_batch_stride = c_row_stride (or natural M if only a is strided)
                eff_a = a_row_stride if a_row_stride != 0 else self.k
                eff_c = c_row_stride if c_row_stride != 0 else self.m
                return (
                    f"    run_matmul({a}, {b}, {c},\n"
                    f"               1u, {self.k}u, {self.m}u, {self.n}u,\n"
                    f"               {eff_a}u, {self.b_batch_stride}u, {eff_c}u);"
                )
            return (
                f"    run_matmul({a}, {b}, {c},\n"
                f"               {self.n}u, {self.k}u, {self.m}u, {self.batch}u,\n"
                f"               {self.a_batch_stride}u,"
                f" {self.b_batch_stride}u,"
                f" {self.c_batch_stride}u);"
            )

        # 4D×3D broadcasting: outer loop over the b1 dimension.
        # A and Y advance by one b2-block per iteration; B repeats at offset 0.
        return "\n".join([
            f"    for (unsigned _i = 0u; _i < {self.outer_count}u; _i++) {{",
            f"        run_matmul_at({a}, _i * {self.a_outer_stride}u,",
            f"                      {b}, _i * {self.b_outer_stride}u,",
            f"                      {c}, _i * {self.c_outer_stride}u,",
            f"                      {self.n}u, {self.k}u, {self.m}u, {self.batch}u,",
            f"                      {self.a_batch_stride}u, {self.b_batch_stride}u,"
            f" {self.c_batch_stride}u);",
            "    }",
        ])

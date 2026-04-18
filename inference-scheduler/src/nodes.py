"""
Supported ONNX operators and their mapping to VectorOPKernel op codes.

Each ScheduledNode wraps one onnx.NodeProto, holds resolved TensorInfo
references for its inputs/outputs, and knows how to emit the corresponding
run_op() / run_op_at() call in the generated C file.

Supported ONNX ops
------------------
  Add                 → OP_ADD   (binary)
  Sub                 → OP_SUB   (binary)
  Mul                 → OP_MUL   (binary)
  Div                 → OP_DIV   (binary)
  Relu                → OP_RELU  (unary)
  Clip(min=0, max=6)  → OP_RELU6 (unary)

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
from typing import List, Optional, Tuple

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

_ALIGN_BYTES    = 16   # AXI burst alignment requirement (hardware-fixed)
_BYTES_PER_ELEM = 2    # must match INFERENCE_BYTES_PER_ELEM in inference.h
_ALIGN_ELEMS    = _ALIGN_BYTES // _BYTES_PER_ELEM   # derived; never hardcode this


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
    t: TensorInfo, output: TensorInfo
) -> Tuple[int, int, int, bool]:
    """
    Compute how tensor *t* broadcasts to *output*.

    Returns (outer_count, chunk_size, aligned_chunk_size, advances):

      outer_count         Number of loop iterations.  1 = no broadcast
                          (t covers the whole output in one call).
      chunk_size          t.numel — data elements in one chunk.
      aligned_chunk_size  chunk_size rounded up to _ALIGN_ELEMS so every
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

    # Round chunk up to _ALIGN_ELEMS so each iteration's physical start address
    # is INFERENCE_ALIGN_BYTES-aligned.  The extra elements are gap/padding and
    # are never read or written by VectorOPKernel.
    aligned_chunk_size = (chunk_size + _ALIGN_ELEMS - 1) & ~(_ALIGN_ELEMS - 1)

    return (outer_count, chunk_size, aligned_chunk_size, False)


# ------------------------------------------------------------------ #
# ScheduledNode                                                       #
# ------------------------------------------------------------------ #

@dataclass
class ScheduledNode:
    """One ONNX operator mapped to one or more VectorOPKernel invocations."""

    onnx_node:  onnx.NodeProto
    op_code:    int                   # OP_ADD … OP_RELU6
    arity:      int                   # 1 = unary, 2 = binary
    inputs:     List[TensorInfo]      # resolved input tensors
    output:     TensorInfo            # single output tensor
    index:      int = 0               # sequential index in graph

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
        node:    onnx.NodeProto,
        tensors: dict,           # name → TensorInfo
        index:   int,
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
            a_outer, _, _, a_advances = _broadcast_info(self.inputs[0], self.output)
            b_outer, _, _, b_advances = _broadcast_info(self.inputs[1], self.output)

            if not a_advances and not b_advances:
                raise SchedulerError(
                    f"Binary node '{self.onnx_node.name or op_type}': "
                    f"both inputs are smaller than the output — "
                    f"only one input may broadcast per operation."
                )

            outer_count = max(a_outer, b_outer)
            chunk_size  = self.output.numel // outer_count
            aligned_chunk_size = (
                (chunk_size + _ALIGN_ELEMS - 1) & ~(_ALIGN_ELEMS - 1)
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
        a  = self.inputs[0].c_name
        b  = self.inputs[1].c_name if self.arity == 2 else "NULL"
        c  = self.output.c_name
        op = OP_NAMES[self.op_code]

        if self.outer_count == 1:
            # Non-broadcast: run_op() handles cache syncs internally.
            # op_size overrides self.chunk_size when the output buffer is
            # larger than numel due to upstream broadcast alignment padding.
            size = op_size if op_size is not None else self.chunk_size
            return f"    run_op({a}, {b}, {c}, {size}u, {op});"

        # Broadcasting: sync whole buffers once, then loop with run_op_at().
        # The chunk stride uses the CHUNK_STRIDE macro (INFERENCE_ALIGN_UP of
        # CHUNK) so every iteration starts at an INFERENCE_ALIGN_BYTES-aligned
        # physical address.  Gap elements between data blocks are never touched
        # by VectorOPKernel (it receives the exact chunk_size as 'size').
        chunk_macro  = f"INFERENCE_{c.upper()}_CHUNK"
        stride_macro = f"INFERENCE_{c.upper()}_CHUNK_STRIDE"
        n    = self.outer_count
        a_off = f"_i * {stride_macro}" if self.a_advances else "0u"
        b_off = (f"_i * {stride_macro}" if self.b_advances else "0u") \
                if self.arity == 2 else "0u"
        c_off = f"_i * {stride_macro}"

        lines = [f"    inference_buf_sync_to_device({a});"]
        if self.arity == 2:
            lines.append(f"    inference_buf_sync_to_device({b});")
        lines.append(
            f"    for (unsigned _i = 0u; _i < {n}u; _i++) {{"
        )
        lines.append(
            f"        run_op_at({a}, {a_off},"
            f" {b}, {b_off},"
            f" {c}, {c_off},"
            f" {chunk_macro}, {op});"
        )
        lines.append("    }")
        lines.append(f"    inference_buf_sync_from_device({c});")
        return "\n".join(lines)

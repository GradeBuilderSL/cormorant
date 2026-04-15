"""
Supported ONNX operators and their mapping to VectorOPKernel op codes.

Each ScheduledNode wraps one onnx.NodeProto, holds resolved TensorInfo
references for its inputs/outputs, and knows how to emit the corresponding
run_op() call in the generated C file.

Supported ONNX ops
------------------
  Add                 → OP_ADD   (binary)
  Sub                 → OP_SUB   (binary)
  Mul                 → OP_MUL   (binary)
  Div                 → OP_DIV   (binary)
  Relu                → OP_RELU  (unary)
  Clip(min=0, max=6)  → OP_RELU6 (unary)

All other ops raise SchedulerError.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

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


def _get_clip_bounds(node: onnx.NodeProto) -> tuple[Optional[float], Optional[float]]:
    """
    Extract Clip min/max from either:
      - Opset < 11: stored as attributes 'min' and 'max'
      - Opset >= 11: stored as optional input tensors (indices 1 and 2)
    Returns (min_val, max_val); None means the bound is not present.
    """
    # Attribute-based (opset < 11)
    attrs = {a.name: a for a in node.attribute}
    if "min" in attrs or "max" in attrs:
        mn = attrs["min"].f  if "min" in attrs else None
        mx = attrs["max"].f  if "max" in attrs else None
        return mn, mx
    # Input-based bounds are resolved by OnnxGraph from initializers;
    # return None here and let ScheduledNode.validate() re-check after
    # input tensors are bound.
    return None, None


@dataclass
class ScheduledNode:
    """One ONNX operator mapped to a single VectorOPKernel invocation."""

    onnx_node:  onnx.NodeProto
    op_code:    int                   # OP_ADD … OP_RELU6
    arity:      int                   # 1 = unary, 2 = binary
    inputs:     List[TensorInfo]      # resolved input tensors
    output:     TensorInfo            # single output tensor
    index:      int = 0               # sequential index in graph

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
            # For input-based bounds (opset >= 11), check initializer data
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
            # Clip's first input is the data tensor; bounds inputs are constants
            # that are not passed to run_op(), so trim inputs to just the data.
            self.inputs = [self.inputs[0]]
            self.arity  = 1

        # Shape consistency: all inputs must have the same numel as the output
        out_numel = self.output.numel
        for t in self.inputs:
            if t.numel != out_numel:
                raise SchedulerError(
                    f"Shape mismatch in node '{self.onnx_node.name or op_type}': "
                    f"input '{t.onnx_name}' has {t.numel} elements but output "
                    f"'{self.output.onnx_name}' has {out_numel} elements. "
                    f"VectorOPKernel requires all tensors to have the same size."
                )

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

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_comment(self) -> str:
        in_names = ", ".join(t.onnx_name for t in self.inputs)
        return (
            f"    /* [{self.index}] {self.onnx_node.op_type}"
            f"({in_names}) -> {self.output.onnx_name}"
            f"  shape={self.output.shape} */"
        )

    def emit_call(self) -> str:
        a = self.inputs[0].c_name
        b = self.inputs[1].c_name if self.arity == 2 else "NULL"
        c = self.output.c_name
        n = self.output.numel
        op = OP_NAMES[self.op_code]
        return f"    run_op({a}, {b}, {c}, {n}u, {op});"

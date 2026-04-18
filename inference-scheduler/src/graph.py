"""
ONNX model loading and graph resolution.

OnnxGraph
---------
  1. Loads the model and runs shape inference so every intermediate tensor
     has a known shape.
  2. Builds a flat dict of TensorInfo objects covering:
       - constant weights  (model.graph.initializer)
       - graph inputs      (model.graph.input)
       - graph outputs     (model.graph.output)
       - intermediate      (model.graph.value_info, produced by shape inference)
  3. Wraps each NodeProto in a ScheduledNode (validates op support and shapes).
  4. Exposes the ordered node list ready for code generation.
"""

from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import shape_inference, TensorProto

from .tensor import TensorInfo
from .nodes  import ScheduledNode, SchedulerError
from .dtype  import DataType, AP_FIXED_16_8


# ------------------------------------------------------------------ #
# ONNX dtype → numpy dtype string                                     #
# ------------------------------------------------------------------ #

_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT:   "float32",
    TensorProto.DOUBLE:  "float64",
    TensorProto.INT8:    "int8",
    TensorProto.INT16:   "int16",
    TensorProto.INT32:   "int32",
    TensorProto.INT64:   "int64",
    TensorProto.UINT8:   "uint8",
    TensorProto.UINT16:  "uint16",
    TensorProto.UINT32:  "uint32",
    TensorProto.UINT64:  "uint64",
    TensorProto.FLOAT16: "float16",
    TensorProto.BOOL:    "bool",
}


def _onnx_dtype_name(onnx_dtype: int) -> str:
    return _ONNX_DTYPE_MAP.get(onnx_dtype, f"onnx_dtype_{onnx_dtype}")


def _shape_from_type_proto(tp: onnx.TypeProto) -> List[int]:
    """Extract a concrete integer shape list from a TypeProto."""
    if not tp.HasField("tensor_type"):
        return []
    shape = tp.tensor_type.shape
    if shape is None:
        return []
    dims = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            # Symbolic / dynamic dimension — use 0 as placeholder
            dims.append(0)
    return dims


class OnnxGraph:
    """Parsed, validated, and resolved ONNX computation graph."""

    def __init__(self, model_path: str,
                 dtype: DataType = None) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _dtype      = dtype if dtype is not None else AP_FIXED_16_8
        align_elems = _dtype.align_elems

        # Load and validate
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Run shape inference so every intermediate tensor gets a shape
        model = shape_inference.infer_shapes(model)

        graph = model.graph

        # ---------------------------------------------------------- #
        # Build the tensor registry                                    #
        # ---------------------------------------------------------- #
        self._tensors: Dict[str, TensorInfo] = {}

        # 1. Constant weights / initializers
        for init in graph.initializer:
            arr = nph.to_array(init).copy()
            ti  = TensorInfo(
                onnx_name=init.name,
                shape=list(arr.shape),
                dtype=_onnx_dtype_name(init.data_type),
                data=arr.astype(np.float32),   # always store as float32
            )
            self._tensors[init.name] = ti

        # 2. Graph inputs (may overlap with initializers for older opsets)
        for vi in graph.input:
            if vi.name in self._tensors:
                continue  # already registered as initializer
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # 3. Intermediate tensors (shape-inferred by onnx.shape_inference)
        for vi in graph.value_info:
            if vi.name in self._tensors:
                continue
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # 4. Graph outputs
        for vi in graph.output:
            if vi.name in self._tensors:
                continue
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # ---------------------------------------------------------- #
        # Identify model boundaries                                    #
        # ---------------------------------------------------------- #
        # Graph inputs that are NOT in the initializer set are true
        # model inputs (data that the caller supplies at run time).
        init_names = {init.name for init in graph.initializer}
        self._input_names: List[str] = [
            vi.name for vi in graph.input if vi.name not in init_names
        ]
        self._output_names: List[str] = [
            vi.name for vi in graph.output
        ]

        # ---------------------------------------------------------- #
        # Resolve nodes                                               #
        # ---------------------------------------------------------- #
        self._nodes: List[ScheduledNode] = []
        for idx, node in enumerate(graph.node):
            sn = ScheduledNode.from_onnx_node(node, self._tensors, idx, align_elems)
            self._nodes.append(sn)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> List[ScheduledNode]:
        return self._nodes

    @property
    def input_tensors(self) -> List[TensorInfo]:
        return [self._tensors[n] for n in self._input_names]

    @property
    def output_tensors(self) -> List[TensorInfo]:
        return [self._tensors[n] for n in self._output_names]

    @property
    def weight_tensors(self) -> List[TensorInfo]:
        """All constant initializer tensors, in declaration order."""
        seen = set()
        weights = []
        for sn in self._nodes:
            for t in sn.inputs:
                if t.is_weight and t.onnx_name not in seen:
                    seen.add(t.onnx_name)
                    weights.append(t)
        return weights

    @property
    def intermediate_tensors(self) -> List[TensorInfo]:
        """Non-constant, non-input, non-output tensors (writable buffers)."""
        boundary = (
            {t.onnx_name for t in self.input_tensors}
            | {t.onnx_name for t in self.output_tensors}
            | {t.onnx_name for t in self.weight_tensors}
        )
        seen = set()
        result = []
        for sn in self._nodes:
            for t in [sn.output] + sn.inputs:
                if t.onnx_name not in boundary and t.onnx_name not in seen:
                    seen.add(t.onnx_name)
                    result.append(t)
        return result

    def get_tensor(self, name: str) -> TensorInfo:
        return self._tensors[name]

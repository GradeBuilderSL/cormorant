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
import onnx.helper as onnx_helper
import onnx.numpy_helper as nph
from onnx import shape_inference, TensorProto

from typing import Union
from .tensor import TensorInfo
from .nodes  import (ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode,
                     POOL_OP_TYPES, VECTOROP_OP_TYPES, RESHAPE_OP_TYPES, SchedulerError)
from .dtype  import DataType, AP_FIXED_16_8
from ._matmul_hw_config import MM_MAX_K

_ALL_SUPPORTED_OP_TYPES: frozenset = (
    {"MatMul", "Conv", "Gemm"} | POOL_OP_TYPES | VECTOROP_OP_TYPES | RESHAPE_OP_TYPES
)


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

    @staticmethod
    def _preprocess_model(model: onnx.ModelProto):
        """Rewrite Gemm → MatMul + Add (when applicable) so downstream node
        classes only see the supported op set.

        Returns ``(rewritten_model, gemm_decomposed_count)``.  The count is
        zero for graphs that contained no Gemm nodes (in which case the
        original model is returned unmodified).
        """
        """
        Simplify the ONNX graph before scheduling:

          1. Decompose Gemm (alpha=1, beta=1, transA=0) into MatMul + Add
             so existing MatmulNode / ScheduledNode handle it.  When
             transB=1, the constant B initializer is transposed offline
             and a new "<B>_T" initializer is appended so the rewritten
             MatMul can read it as a row-major tensor (transB=0
             semantics).  transB=1 with a non-constant B is rejected
             — runtime transpose is not supported.

        Requires that shape inference has already been run on the model
        so that intermediate shapes are available for the new MatMul output.
        """
        graph = model.graph

        # Build a shape map from all known tensors (inputs, outputs, value_info,
        # and initializers — initializers don't appear in value_info).
        shape_map: Dict[str, List[int]] = {}
        for init in graph.initializer:
            arr = nph.to_array(init)
            shape_map[init.name] = list(arr.shape)
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            dims = [
                d.dim_value if d.HasField("dim_value") else 0
                for d in vi.type.tensor_type.shape.dim
            ]
            shape_map[vi.name] = dims

        gemm_counter = [0]
        new_nodes: List[onnx.NodeProto] = []
        new_value_info: List[onnx.ValueInfoProto] = []

        for node in graph.node:
            if node.op_type != "Gemm":
                new_nodes.append(node)
                continue

            attrs = {a.name: a for a in node.attribute}
            alpha  = attrs["alpha"].f  if "alpha"  in attrs else 1.0
            beta   = attrs["beta"].f   if "beta"   in attrs else 1.0
            transA = attrs["transA"].i if "transA" in attrs else 0
            transB = attrs["transB"].i if "transB" in attrs else 0

            if abs(alpha - 1.0) > 1e-6 or abs(beta - 1.0) > 1e-6:
                raise SchedulerError(
                    f"Gemm node '{node.name}': alpha={alpha}, beta={beta}. "
                    f"Only alpha=1, beta=1 is supported."
                )
            if transA != 0:
                raise SchedulerError(
                    f"Gemm node '{node.name}': transA={transA}. "
                    f"Only transA=0 is supported (A is a runtime tensor; "
                    f"offline transpose is not feasible)."
                )
            if transB not in (0, 1):
                raise SchedulerError(
                    f"Gemm node '{node.name}': transB={transB}. "
                    f"Must be 0 or 1."
                )

            A = node.input[0]
            B = node.input[1]
            C = node.input[2] if len(node.input) >= 3 and node.input[2] else None
            Y = node.output[0]

            # transB=1: transpose the constant B initializer offline so the
            # rewritten MatMul reads it row-major.  We append a new
            # "<B>_T" initializer rather than mutating B in place so any
            # other consumer of the original tensor is left untouched.
            if transB == 1:
                b_init = next(
                    (init for init in graph.initializer if init.name == B),
                    None,
                )
                if b_init is None:
                    raise SchedulerError(
                        f"Gemm node '{node.name}': transB=1 requires B '{B}' "
                        f"to be a constant initializer; runtime transpose is "
                        f"not supported."
                    )
                arr = nph.to_array(b_init)
                if arr.ndim != 2:
                    raise SchedulerError(
                        f"Gemm node '{node.name}': transB=1 with non-2D B "
                        f"(shape={list(arr.shape)}) is not supported."
                    )
                new_B = f"{B}_T"
                if not any(init.name == new_B for init in graph.initializer):
                    graph.initializer.append(
                        nph.from_array(arr.T.copy(), name=new_B)
                    )
                    shape_map[new_B] = list(arr.T.shape)
                B = new_B

            gemm_counter[0] += 1
            n = gemm_counter[0]

            if C:
                # Gemm → MatMul(A,B)→tmp  +  Add(tmp,C)→Y
                tmp = f"_gemm_mm_out_{n}"
                # Infer tmp shape: A[-2] × B[-1]
                a_shape = shape_map.get(A, [])
                b_shape = shape_map.get(B, [])
                if len(a_shape) >= 2 and len(b_shape) >= 2:
                    tmp_shape = a_shape[:-1] + [b_shape[-1]]
                else:
                    tmp_shape = []
                if tmp_shape:
                    new_value_info.append(
                        onnx_helper.make_tensor_value_info(
                            tmp, TensorProto.FLOAT, tmp_shape
                        )
                    )
                new_nodes.append(
                    onnx_helper.make_node("MatMul", inputs=[A, B], outputs=[tmp],
                                          name=f"_gemm_matmul_{n}")
                )
                new_nodes.append(
                    onnx_helper.make_node("Add", inputs=[tmp, C], outputs=[Y],
                                          name=f"_gemm_add_{n}")
                )
            else:
                # No bias: Gemm → MatMul(A,B)→Y
                new_nodes.append(
                    onnx_helper.make_node("MatMul", inputs=[A, B], outputs=[Y],
                                          name=f"_gemm_matmul_{n}")
                )

        if gemm_counter[0] == 0:
            return model, 0  # nothing changed

        new_graph = onnx_helper.make_graph(
            new_nodes,
            graph.name,
            list(graph.input),
            list(graph.output),
            initializer=list(graph.initializer),
            value_info=list(graph.value_info) + new_value_info,
        )
        new_model = onnx_helper.make_model(
            new_graph, opset_imports=list(model.opset_import)
        )
        new_model.ir_version = model.ir_version
        return new_model, gemm_counter[0]

    @staticmethod
    def _is_pointwise_conv(node: onnx.NodeProto) -> bool:
        """True iff ``node`` is a vanilla 1×1 ONNX Conv: kernel 1×1, stride 1,
        pad 0, dilation 1, group 1.  Anything else (3×3, depthwise, strided,
        padded) is rejected and stays on ConvKernel."""
        if node.op_type != "Conv":
            return False
        attrs = {a.name: a for a in node.attribute}

        def _ints(name, default):
            return list(attrs[name].ints) if name in attrs else default

        if _ints("kernel_shape", [1, 1])      != [1, 1]:       return False
        if _ints("strides",      [1, 1])      != [1, 1]:       return False
        if _ints("pads",         [0, 0, 0, 0]) != [0, 0, 0, 0]: return False
        if _ints("dilations",    [1, 1])      != [1, 1]:       return False
        if (attrs["group"].i if "group" in attrs else 1) != 1: return False
        return True

    @staticmethod
    def _rewrite_pointwise_conv_as_matmul(
        model: onnx.ModelProto,
        mm_max_k: int = MM_MAX_K,
        max_bias_tile_elems: int = 65536,
    ):
        """Rewrite pointwise (1×1) Conv nodes as MatMul (+ optional bias Add).

        For each eligible ``Conv`` node:

          1. Reshape  W [OC, IC, 1, 1]   → W2 [OC, IC]            (alias)
          2. Reshape  X [N,  IC, H,  W]  → X2 [N, IC, H*W]        (alias)
          3. MatMul   W2 × X2            → Y2 [N, OC, H*W]
          4. (bias)   Tile B[OC] to B2[OC, H*W] offline,
                      Add Y2 + B2        → Y2b [N, OC, H*W]
                      (broadcast on the leading N dim — fits the
                       contiguous-leading-block rule the VectorOPKernel
                       scheduler expects)
          5. Reshape  Y2(b) [N,OC,H*W]   → Y [N, OC, H, W]        (alias)

        A Conv is rewritten iff all of:
          * ``_is_pointwise_conv(node)`` (1×1, stride=1, pad=0, dil=1, group=1)
          * static rank-4 shapes for X and W with all dims known
          * ``IC ≤ mm_max_k``                       (MatmulKernel constraint)
          * if biased: bias is a constant initializer and
            ``OC * H * W ≤ max_bias_tile_elems``    (avoid weight blowup)

        Otherwise the original Conv is left untouched (and stays on ConvKernel).

        Returns ``(rewritten_model, count)``.
        """
        graph = model.graph

        # Shape map covering initializers + value_info + inputs + outputs.
        shape_map: Dict[str, List[int]] = {}
        for init in graph.initializer:
            shape_map[init.name] = list(nph.to_array(init).shape)
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            dims = [
                d.dim_value if d.HasField("dim_value") else 0
                for d in vi.type.tensor_type.shape.dim
            ]
            shape_map[vi.name] = dims

        inits_by_name = {init.name: idx for idx, init in enumerate(graph.initializer)}

        counter            = [0]
        new_nodes          : List[onnx.NodeProto]      = []
        new_value_info     : List[onnx.ValueInfoProto] = []
        new_initializers   : List[onnx.TensorProto]    = []

        for node in graph.node:
            if not OnnxGraph._is_pointwise_conv(node):
                new_nodes.append(node)
                continue

            X  = node.input[0]
            Wt = node.input[1]
            Bt = node.input[2] if len(node.input) >= 3 and node.input[2] else None
            Y  = node.output[0]

            x_shape = shape_map.get(X,  [])
            w_shape = shape_map.get(Wt, [])
            if (len(x_shape) != 4 or len(w_shape) != 4
                    or any(d <= 0 for d in x_shape)
                    or any(d <= 0 for d in w_shape)):
                new_nodes.append(node)
                continue

            N, IC, H, W = x_shape
            OC, _, kH, kW = w_shape
            if (kH, kW) != (1, 1):
                new_nodes.append(node)   # shape disagrees with kernel_shape attr; bail
                continue
            if IC > mm_max_k:
                new_nodes.append(node)   # MatmulKernel can't handle this IC
                continue

            biased = Bt is not None
            if biased:
                # Need the bias as a constant so we can tile it offline.
                if Bt not in inits_by_name:
                    new_nodes.append(node)
                    continue
                tile_elems = OC * H * W
                if tile_elems > max_bias_tile_elems:
                    new_nodes.append(node)
                    continue

            counter[0] += 1
            n = counter[0]

            def _mk_shape_init(name: str, shape):
                arr = np.array(list(shape), dtype=np.int64)
                new_initializers.append(nph.from_array(arr, name=name))

            # 1. Reshape W [OC, IC, 1, 1] → W2 [OC, IC]
            W2       = f"_pwconv_{n}_W2"
            W2_shape = f"_pwconv_{n}_W2_shape"
            _mk_shape_init(W2_shape, [OC, IC])
            new_nodes.append(onnx_helper.make_node(
                "Reshape", inputs=[Wt, W2_shape], outputs=[W2],
                name=f"_pwconv_{n}_reshape_W"))
            new_value_info.append(onnx_helper.make_tensor_value_info(
                W2, TensorProto.FLOAT, [OC, IC]))

            # 2. Reshape X [N, IC, H, W] → X2 [N, IC, H*W]
            X2       = f"_pwconv_{n}_X2"
            X2_shape = f"_pwconv_{n}_X2_shape"
            _mk_shape_init(X2_shape, [N, IC, H * W])
            new_nodes.append(onnx_helper.make_node(
                "Reshape", inputs=[X, X2_shape], outputs=[X2],
                name=f"_pwconv_{n}_reshape_X"))
            new_value_info.append(onnx_helper.make_tensor_value_info(
                X2, TensorProto.FLOAT, [N, IC, H * W]))

            # 3. MatMul: W2 [OC, IC] @ X2 [N, IC, H*W] → Y2 [N, OC, H*W]
            Y2 = f"_pwconv_{n}_Y2"
            new_nodes.append(onnx_helper.make_node(
                "MatMul", inputs=[W2, X2], outputs=[Y2],
                name=f"_pwconv_{n}_matmul"))
            new_value_info.append(onnx_helper.make_tensor_value_info(
                Y2, TensorProto.FLOAT, [N, OC, H * W]))

            post = Y2
            if biased:
                # 4. Tile bias [OC] → [OC, H*W] so the Add broadcasts on the
                #    LEADING N dim only (contiguous-leading-block rule).
                b_init = graph.initializer[inits_by_name[Bt]]
                b_arr  = nph.to_array(b_init).astype(np.float32)
                if b_arr.shape != (OC,):
                    new_nodes.append(node)
                    counter[0] -= 1
                    continue
                b_tiled = np.tile(b_arr.reshape(OC, 1), (1, H * W)).astype(np.float32)
                B2 = f"_pwconv_{n}_B_tiled"
                new_initializers.append(nph.from_array(b_tiled, name=B2))

                Y2b = f"_pwconv_{n}_Y2b"
                new_nodes.append(onnx_helper.make_node(
                    "Add", inputs=[Y2, B2], outputs=[Y2b],
                    name=f"_pwconv_{n}_bias_add"))
                new_value_info.append(onnx_helper.make_tensor_value_info(
                    Y2b, TensorProto.FLOAT, [N, OC, H * W]))
                post = Y2b

            # 5. Reshape post [N, OC, H*W] → Y [N, OC, H, W]  (alias)
            Y_shape = f"_pwconv_{n}_Y_shape"
            _mk_shape_init(Y_shape, [N, OC, H, W])
            new_nodes.append(onnx_helper.make_node(
                "Reshape", inputs=[post, Y_shape], outputs=[Y],
                name=f"_pwconv_{n}_reshape_Y"))
            # Y already has value_info from the original Conv output.

        if counter[0] == 0:
            return model, 0

        new_graph = onnx_helper.make_graph(
            new_nodes,
            graph.name,
            list(graph.input),
            list(graph.output),
            initializer=list(graph.initializer) + new_initializers,
            value_info=list(graph.value_info) + new_value_info,
        )
        new_model = onnx_helper.make_model(
            new_graph, opset_imports=list(model.opset_import)
        )
        new_model.ir_version = model.ir_version
        return new_model, counter[0]

    def __init__(self, model_path: str,
                 dtype: DataType = None,
                 rewrite_pointwise_conv: bool = True) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _dtype      = dtype if dtype is not None else AP_FIXED_16_8
        align_elems = _dtype.align_elems

        # Load and validate
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Run shape inference so every intermediate tensor gets a shape
        model = shape_inference.infer_shapes(model)

        # Simplify: decompose Gemm → MatMul + Add.  The count is exposed via
        # ``self.gemm_decomposed_count`` so the report generator can list it
        # as an applied transformation.
        model, self.gemm_decomposed_count = OnnxGraph._preprocess_model(model)

        # Rewrite 1×1 (pointwise) Conv → MatMul (+ optional bias Add).  On
        # by default — opt out with rewrite_pointwise_conv=False (CLI:
        # --no-rewrite-pointwise-conv).  ConvKernel's line-buffer /
        # patch-buffer machinery is pure overhead for K=1; MatmulKernel
        # handles these much more cleanly and lets the scheduler overlap
        # pointwise + depthwise on separate lanes.
        if rewrite_pointwise_conv:
            model, self.pointwise_conv_rewritten_count = (
                OnnxGraph._rewrite_pointwise_conv_as_matmul(model)
            )
        else:
            self.pointwise_conv_rewritten_count = 0

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
        self._nodes: List[Union[ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode]] = []
        for idx, node in enumerate(graph.node):
            if node.op_type == "MatMul":
                sn = MatmulNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type == "Conv":
                sn = ConvNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type in POOL_OP_TYPES:
                sn = PoolNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type in RESHAPE_OP_TYPES:
                sn = ReshapeNode.from_onnx_node(node, self._tensors, idx, align_elems)
            else:
                if node.op_type not in VECTOROP_OP_TYPES:
                    raise SchedulerError(
                        f"Node '{node.name or node.op_type}' "
                        f"(op_type='{node.op_type}') is not supported.\n"
                        f"Supported ops: {sorted(_ALL_SUPPORTED_OP_TYPES)}"
                    )
                sn = ScheduledNode.from_onnx_node(node, self._tensors, idx, align_elems)
            self._nodes.append(sn)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> List[Union[ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode]]:
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

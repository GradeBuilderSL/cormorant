"""Tests for Reshape (buffer-alias) and Gemm (→MatMul+Add) transformations."""

import os
import sys
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto, shape_inference

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import (ReshapeNode, MatmulNode, ScheduledNode, SchedulerError,
                          ConvNode, PoolNode)
from src.tensor  import TensorInfo


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _m(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _models_exist() -> bool:
    return (os.path.isfile(_m("reshape_flatten.onnx"))
            and os.path.isfile(_m("squeeze_then_matmul.onnx")))


def _gen(name: str) -> CodeGenerator:
    path = _m(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


# ═══════════════════════════════════════════════════════════════════
# Helpers — build stub TensorInfo without a real ONNX model
# ═══════════════════════════════════════════════════════════════════

def _tensor(name: str, shape: list) -> TensorInfo:
    return TensorInfo(onnx_name=name, shape=shape, dtype="float32", data=None)


# ═══════════════════════════════════════════════════════════════════
# Gemm preprocessing: _preprocess_model()
# ═══════════════════════════════════════════════════════════════════

class TestGemmPreprocess(unittest.TestCase):
    """_preprocess_model() decomposes Gemm → MatMul + Add at the ONNX proto level."""

    def _make_gemm_model(self, *, with_bias: bool,
                         transA: int = 0, transB: int = 0,
                         alpha: float = 1.0, beta: float = 1.0):
        w = np.eye(4, dtype=np.float32)
        inits = [nph.from_array(w, name="W")]
        inputs_proto = ["X", "W"]
        if with_bias:
            b = np.zeros((1, 4), dtype=np.float32)
            inits.append(nph.from_array(b, name="B"))
            inputs_proto.append("B")
        node = oh.make_node(
            "Gemm", inputs=inputs_proto, outputs=["Y"],
            transA=transA, transB=transB, alpha=alpha, beta=beta,
        )
        graph = oh.make_graph(
            [node], "test_gemm",
            inputs=[oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            outputs=[oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
            initializer=inits,
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        return shape_inference.infer_shapes(model)

    def test_gemm_with_bias_becomes_matmul_add(self):
        model = self._make_gemm_model(with_bias=True)
        result = OnnxGraph._preprocess_model(model)
        ops = [n.op_type for n in result.graph.node]
        self.assertIn("MatMul", ops)
        self.assertIn("Add",    ops)
        self.assertNotIn("Gemm", ops)

    def test_gemm_no_bias_becomes_matmul_only(self):
        model = self._make_gemm_model(with_bias=False)
        result = OnnxGraph._preprocess_model(model)
        ops = [n.op_type for n in result.graph.node]
        self.assertIn("MatMul", ops)
        self.assertNotIn("Gemm", ops)
        self.assertNotIn("Add",  ops)

    def test_gemm_intermediate_in_value_info(self):
        """MatMul output tensor from Gemm decomposition appears in value_info."""
        model  = self._make_gemm_model(with_bias=True)
        result = OnnxGraph._preprocess_model(model)
        vi_names = {vi.name for vi in result.graph.value_info}
        # The MatMul output should be registered so tensor registry can find it
        matmul_out = next(
            n.output[0] for n in result.graph.node if n.op_type == "MatMul"
        )
        self.assertIn(matmul_out, vi_names)

    def test_gemm_matmul_feeds_add(self):
        """Add node's first input == MatMul node's output."""
        model  = self._make_gemm_model(with_bias=True)
        result = OnnxGraph._preprocess_model(model)
        mm_out = next(n.output[0] for n in result.graph.node if n.op_type == "MatMul")
        add_in = next(n.input[0]  for n in result.graph.node if n.op_type == "Add")
        self.assertEqual(mm_out, add_in)

    def test_no_gemm_returns_same_model(self):
        """_preprocess_model returns the original model unchanged when no Gemm nodes."""
        w    = np.eye(4, dtype=np.float32)
        init = nph.from_array(w, name="W")
        node = oh.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
        graph = oh.make_graph(
            [node], "matmul_only",
            inputs=[oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            outputs=[oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
            initializer=[init],
        )
        model  = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model  = shape_inference.infer_shapes(model)
        result = OnnxGraph._preprocess_model(model)
        self.assertIs(result, model)   # identical object when nothing changed

    def test_gemm_transA_raises(self):
        model = self._make_gemm_model(with_bias=False, transA=1)
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph._preprocess_model(model)
        self.assertIn("transA=1", str(cm.exception))

    def test_gemm_alpha_raises(self):
        model = self._make_gemm_model(with_bias=True, alpha=2.0)
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph._preprocess_model(model)
        self.assertIn("alpha=2.0", str(cm.exception))

    def test_gemm_chain_produces_two_matmul_add_pairs(self):
        """Two chained Gemms both decompose."""
        w1 = nph.from_array(np.eye(4, dtype=np.float32), name="W1")
        b1 = nph.from_array(np.zeros((1, 4), dtype=np.float32), name="B1")
        w2 = nph.from_array(np.eye(4, dtype=np.float32), name="W2")
        b2 = nph.from_array(np.zeros((1, 4), dtype=np.float32), name="B2")
        g1 = oh.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["H"])
        g2 = oh.make_node("Gemm", inputs=["H", "W2", "B2"], outputs=["Y"])
        graph = oh.make_graph(
            [g1, g2], "gemm_chain",
            inputs=[oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            outputs=[oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
            initializer=[w1, b1, w2, b2],
        )
        model  = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model  = shape_inference.infer_shapes(model)
        result = OnnxGraph._preprocess_model(model)
        ops    = [n.op_type for n in result.graph.node]
        self.assertEqual(ops.count("MatMul"), 2)
        self.assertEqual(ops.count("Add"),    2)
        self.assertNotIn("Gemm", ops)


# ═══════════════════════════════════════════════════════════════════
# ReshapeNode: construction and emit methods
# ═══════════════════════════════════════════════════════════════════

class TestReshapeNodeUnit(unittest.TestCase):
    """ReshapeNode.from_onnx_node and emit methods (no ONNX model file needed)."""

    def _make_node(self, src_shape, out_shape) -> ReshapeNode:
        """Build a ReshapeNode from stub TensorInfo objects."""
        src = _tensor("src", src_shape)
        out = _tensor("out", out_shape)
        tensors = {"src": src, "out": out, "shape": _tensor("shape", [len(out_shape)])}
        node = oh.make_node("Reshape", inputs=["src", "shape"], outputs=["out"])
        return ReshapeNode.from_onnx_node(node, tensors, index=0, align_elems=8)

    def test_valid_reshape_created(self):
        sn = self._make_node([1, 4, 4, 4], [1, 64])
        self.assertIsInstance(sn, ReshapeNode)

    def test_numel_mismatch_raises(self):
        with self.assertRaises(SchedulerError) as cm:
            self._make_node([1, 4, 4, 4], [1, 100])   # 64 ≠ 100
        self.assertIn("numel", str(cm.exception).lower())

    def test_kernel_name_empty(self):
        sn = self._make_node([1, 64], [1, 8, 8])
        self.assertEqual(sn.kernel_name, "")

    def test_compatibility_shims(self):
        sn = self._make_node([1, 4, 4, 4], [1, 64])
        self.assertEqual(sn.outer_count,        1)
        self.assertEqual(sn.chunk_size,         0)
        self.assertEqual(sn.aligned_chunk_size, 0)
        self.assertTrue(sn.a_advances)
        self.assertTrue(sn.b_advances)
        self.assertEqual(sn.arity, 1)

    def test_emit_call_empty(self):
        sn = self._make_node([1, 4, 4, 4], [1, 64])
        self.assertEqual(sn.emit_call({}), "")

    def test_emit_comment_contains_shapes(self):
        sn = self._make_node([1, 4, 4, 4], [1, 64])
        comment = sn.emit_comment()
        self.assertIn("[1, 4, 4, 4]", comment)
        self.assertIn("[1, 64]",      comment)
        self.assertIn("alias",        comment)

    def test_emit_comment_contains_names(self):
        sn = self._make_node([1, 4, 4, 4], [1, 64])
        comment = sn.emit_comment()
        self.assertIn("src", comment)
        self.assertIn("out", comment)


# ═══════════════════════════════════════════════════════════════════
# OnnxGraph node dispatch
# ═══════════════════════════════════════════════════════════════════

@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestGraphDispatch(unittest.TestCase):
    """OnnxGraph dispatches Reshape → ReshapeNode and Gemm → MatmulNode + ScheduledNode."""

    def test_reshape_flatten_creates_reshape_node(self):
        g = OnnxGraph(_m("reshape_flatten.onnx"))
        self.assertEqual(len(g.nodes), 1)
        self.assertIsInstance(g.nodes[0], ReshapeNode)

    def test_reshape_then_matmul_node_types(self):
        g = OnnxGraph(_m("reshape_then_matmul.onnx"))
        types = [type(sn) for sn in g.nodes]
        self.assertIn(ReshapeNode,  types)
        self.assertIn(MatmulNode,   types)

    def test_gemm_no_bias_creates_matmul_only(self):
        g = OnnxGraph(_m("gemm_no_bias.onnx"))
        self.assertEqual(len(g.nodes), 1)
        self.assertIsInstance(g.nodes[0], MatmulNode)

    def test_gemm_with_bias_creates_matmul_plus_add(self):
        g = OnnxGraph(_m("gemm_with_bias.onnx"))
        types = [type(sn) for sn in g.nodes]
        self.assertIn(MatmulNode,    types)
        self.assertIn(ScheduledNode, types)

    def test_gemm_with_bias_node_order(self):
        g = OnnxGraph(_m("gemm_with_bias.onnx"))
        self.assertIsInstance(g.nodes[0], MatmulNode)
        self.assertIsInstance(g.nodes[1], ScheduledNode)

    def test_gemm_chain_four_nodes(self):
        """Two Gemm(with bias) → 2×MatmulNode + 2×ScheduledNode(Add)."""
        g = OnnxGraph(_m("gemm_chain.onnx"))
        self.assertEqual(len(g.nodes), 4)
        self.assertIsInstance(g.nodes[0], MatmulNode)
        self.assertIsInstance(g.nodes[1], ScheduledNode)
        self.assertIsInstance(g.nodes[2], MatmulNode)
        self.assertIsInstance(g.nodes[3], ScheduledNode)

    def test_reshape_gemm_pipeline_node_types(self):
        g = OnnxGraph(_m("reshape_gemm_pipeline.onnx"))
        types = [type(sn) for sn in g.nodes]
        self.assertIn(ScheduledNode, types)   # Relu
        self.assertIn(ReshapeNode,   types)
        self.assertIn(MatmulNode,    types)

    def test_gemm_transA_raises_on_load(self):
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(_m("gemm_transA_unsupported.onnx"))
        self.assertIn("transA=1", str(cm.exception))

    def test_reshape_node_input_output_shapes(self):
        g = OnnxGraph(_m("reshape_flatten.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ReshapeNode)
        self.assertEqual(sn.inputs[0].shape, [1, 4, 4, 4])
        self.assertEqual(sn.output.shape,    [1, 64])


# ═══════════════════════════════════════════════════════════════════
# TensorLayout: reshape output excluded from propagation phases
# ═══════════════════════════════════════════════════════════════════

@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestReshapeLayouts(unittest.TestCase):
    """ReshapeNode output has a flat layout; it is excluded from phases 2 & 3."""

    def test_reshape_output_flat(self):
        gen = _gen("reshape_flatten.onnx")
        z_lay = gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.alloc,    z_lay.numel)

    def test_reshape_source_alloc_equals_numel(self):
        gen = _gen("reshape_flatten.onnx")
        x_lay = gen._layouts["X"]
        self.assertEqual(x_lay.alloc, x_lay.numel)
        self.assertEqual(x_lay.numel, 1 * 4 * 4 * 4)

    def test_reshape_then_matmul_intermediate_flat(self):
        """Z (Reshape output) must be flat even though it feeds MatMul."""
        gen   = _gen("reshape_then_matmul.onnx")
        z_lay = gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.alloc,    z_lay.numel)

    def test_reshape_aliases_property(self):
        gen     = _gen("reshape_flatten.onnx")
        aliases = gen._reshape_aliases
        # "Z" (output of Reshape) should map to the C name of "X" (source)
        self.assertIn("Z", aliases)
        self.assertEqual(aliases["Z"], gen._graph.nodes[0].inputs[0].c_name)

    def test_reshape_gemm_pipeline_aliases(self):
        """Reshape in a multi-node graph: alias map contains the reshape output."""
        gen = _gen("reshape_gemm_pipeline.onnx")
        sn  = next(n for n in gen._graph.nodes if isinstance(n, ReshapeNode))
        aliases = gen._reshape_aliases
        self.assertIn(sn.output.onnx_name, aliases)

    def test_no_reshape_empty_aliases(self):
        """Model with no Reshape nodes → empty alias dict."""
        gen = _gen("gemm_with_bias.onnx")
        self.assertEqual(gen._reshape_aliases, {})


# ═══════════════════════════════════════════════════════════════════
# Generated inference.c: reshape alias + Gemm decomposition
# ═══════════════════════════════════════════════════════════════════

@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestReshapeGemmSource(unittest.TestCase):
    """Generated inference.c handles reshape aliases and Gemm-decomposed nodes."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    # ---- Reshape alias in inference_init() ----

    def test_reshape_alias_assignment_in_init(self):
        """Reshape output is pointer-assigned (= source), not alloc'd.

        reshape_then_matmul: Z is an intermediate (fed into MatMul), so the
        alias assignment appears in inference_init().  In reshape_flatten, Z
        is the graph output (caller-supplied), so no internal allocation or
        alias assignment is needed.
        """
        s = self._src("reshape_then_matmul.onnx")
        self.assertIn("reshape alias", s)
        self.assertNotIn("inference_buf_alloc(Z", s)

    def test_reshape_alias_not_freed_in_deinit(self):
        """Reshape output buffer is NULLed but never freed."""
        s = self._src("reshape_then_matmul.onnx")
        self.assertIn("reshape alias — not owned", s)
        reshape_sn = next(
            n for n in OnnxGraph(_m("reshape_then_matmul.onnx")).nodes
            if isinstance(n, ReshapeNode)
        )
        alias_name = reshape_sn.output.c_name
        self.assertNotIn(f"inference_buf_free({alias_name})", s)

    def test_no_run_reshape_helper(self):
        """No 'static void run_reshape(' in generated source."""
        s = self._src("reshape_flatten.onnx")
        self.assertNotIn("static void run_reshape(", s)

    def test_reshape_emit_call_blank_in_run(self):
        """inference_run() body has comment for reshape but no function call."""
        s = self._src("reshape_flatten.onnx")
        self.assertIn("buffer alias, no hardware call", s)

    # ---- Gemm decomposes to MatMul + Add ----

    def test_gemm_no_bias_run_matmul_called(self):
        s = self._src("gemm_no_bias.onnx")
        self.assertIn("run_matmul(", s)

    def test_gemm_no_bias_no_run_op(self):
        """Pure Gemm(no bias) → only MatmulKernel, no VectorOP Add."""
        s = self._src("gemm_no_bias.onnx")
        self.assertNotIn("static void run_op(", s)

    def test_gemm_with_bias_both_helpers(self):
        """Gemm(with bias) → run_matmul + run_op (for the Add)."""
        s = self._src("gemm_with_bias.onnx")
        self.assertIn("static void run_matmul(", s)
        self.assertIn("static void run_op(",     s)

    def test_gemm_with_bias_add_call_present(self):
        """The Add node emitted by Gemm decomposition calls run_op with VECTOROP_ADD."""
        s = self._src("gemm_with_bias.onnx")
        self.assertIn("VECTOROP_ADD", s)

    def test_reshape_then_matmul_headers(self):
        """reshape + MatMul model includes xmatmulkernel.h only (no VOP)."""
        s = self._src("reshape_then_matmul.onnx")
        self.assertIn('#include "xmatmulkernel.h"', s)
        self.assertNotIn('#include "xvectoropkernel.h"', s)

    def test_gemm_chain_two_matmul_calls(self):
        s = self._src("gemm_chain.onnx")
        # "run_matmul(" also appears in comments and the function definition;
        # count only indented call sites inside inference_run().
        self.assertEqual(s.count("    run_matmul("), 2)

    def test_reshape_gemm_pipeline_all_ops_present(self):
        """Relu+Reshape+Gemm(bias) → run_op (relu) + run_matmul + run_op (add)."""
        s = self._src("reshape_gemm_pipeline.onnx")
        self.assertIn("static void run_op(",     s)
        self.assertIn("static void run_matmul(", s)
        self.assertIn("reshape alias",           s)
        self.assertIn("VECTOROP_RELU",           s)
        self.assertIn("VECTOROP_ADD",            s)

    # ---- Active kernels ----

    def test_reshape_flatten_no_active_kernel(self):
        """Reshape-only model has no active hardware kernel."""
        gen   = _gen("reshape_flatten.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, [])

    def test_gemm_with_bias_active_kernels(self):
        gen   = _gen("gemm_with_bias.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertIn("MatmulKernel",    names)
        self.assertIn("VectorOPKernel",  names)


# ═══════════════════════════════════════════════════════════════════
# Simulation: numeric correctness
# ═══════════════════════════════════════════════════════════════════

@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestReshapeGemmSimulate(unittest.TestCase):
    """Forward-pass simulation produces correct numeric outputs."""

    def _sim(self, name: str, inputs: dict | None = None) -> dict:
        gen = _gen(name)
        if inputs is None:
            return gen._simulate()
        return gen.simulate(inputs)

    def test_reshape_flatten_output_shape(self):
        arrays = self._sim("reshape_flatten.onnx")
        self.assertIn("Z", arrays)
        self.assertEqual(arrays["Z"].shape, (1, 64))

    def test_reshape_preserves_values(self):
        """Reshape just reinterprets memory — values are identical."""
        gen = _gen("reshape_flatten.onnx")
        x_q = gen._dtype.quantize(
            np.arange(64, dtype=np.float64).reshape(1, 4, 4, 4) * (1.0/256)
        )
        result = gen.simulate({"X": x_q})
        np.testing.assert_array_equal(
            result["Z"].flatten(), x_q.flatten(),
            err_msg="Reshape changed values"
        )

    def test_gemm_no_bias_output_shape(self):
        arrays = self._sim("gemm_no_bias.onnx")
        self.assertIn("Y", arrays)
        self.assertEqual(arrays["Y"].shape, (1, 8))

    def test_gemm_with_bias_output_shape(self):
        arrays = self._sim("gemm_with_bias.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 8))

    def test_gemm_with_bias_adds_bias(self):
        """Y = X @ W + B.  With X=ones, W=1/64*ones, B=0.5: Y ≈ 1.0 + 0.5 = 1.5."""
        gen  = _gen("gemm_with_bias.onnx")
        x_q  = gen._dtype.quantize(np.ones((1, 64), dtype=np.float64))
        y    = gen.simulate({"X": x_q})["Y"]
        # X@W = 1.0 (64 × 1/64), + bias 0.5 → 1.5; allow ap_fixed rounding
        np.testing.assert_allclose(y, 1.5, atol=0.02)

    def test_gemm_chain_output_shape(self):
        arrays = self._sim("gemm_chain.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 8))

    def test_gemm_chain_two_layers_correct(self):
        """X[1,32] → Gemm(W1=1/32) → H=1.0 → Gemm(W2=1/16) → Y=1.0."""
        gen  = _gen("gemm_chain.onnx")
        x_q  = gen._dtype.quantize(np.ones((1, 32), dtype=np.float64))
        y    = gen.simulate({"X": x_q})["Y"]
        np.testing.assert_allclose(y, 1.0, atol=0.02)

    def test_reshape_then_matmul_output_shape(self):
        arrays = self._sim("reshape_then_matmul.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 8))

    def test_reshape_then_matmul_correct(self):
        """X[1,4,4,4]=1.0 → flatten → MatMul(W=1/64) → Y=1.0."""
        gen  = _gen("reshape_then_matmul.onnx")
        x_q  = gen._dtype.quantize(np.ones((1, 4, 4, 4), dtype=np.float64))
        y    = gen.simulate({"X": x_q})["Y"]
        np.testing.assert_allclose(y, 1.0, atol=0.02)

    def test_reshape_gemm_pipeline_output_shape(self):
        arrays = self._sim("reshape_gemm_pipeline.onnx")
        self.assertEqual(arrays["Y"].shape, (1, 4))

    def test_reshape_gemm_pipeline_relu_then_fc(self):
        """Relu zero-clamps negatives before the FC layer."""
        gen  = _gen("reshape_gemm_pipeline.onnx")
        # All-ones input: Relu is a no-op, MatMul(W=1/128)*1.0 = 1.0, bias=0
        x_q  = gen._dtype.quantize(np.ones((1, 8, 4, 4), dtype=np.float64))
        y    = gen.simulate({"X": x_q})["Y"]
        np.testing.assert_allclose(y, 1.0, atol=0.02)


# ═══════════════════════════════════════════════════════════════════
# Squeeze / Unsqueeze: unit construction
# ═══════════════════════════════════════════════════════════════════

class TestSqueezeNodeUnit(unittest.TestCase):
    """Squeeze and Unsqueeze nodes are constructed as ReshapeNode (buffer alias)."""

    def _make_squeeze(self, src_shape, out_shape) -> ReshapeNode:
        src = _tensor("src", src_shape)
        out = _tensor("out", out_shape)
        tensors = {"src": src, "out": out}
        node = oh.make_node("Squeeze", inputs=["src"], outputs=["out"], axes=[2, 3])
        return ReshapeNode.from_onnx_node(node, tensors, index=0, align_elems=8)

    def _make_unsqueeze(self, src_shape, out_shape) -> ReshapeNode:
        src = _tensor("src", src_shape)
        out = _tensor("out", out_shape)
        tensors = {"src": src, "out": out}
        node = oh.make_node("Unsqueeze", inputs=["src"], outputs=["out"], axes=[2, 3])
        return ReshapeNode.from_onnx_node(node, tensors, index=0, align_elems=8)

    def test_squeeze_creates_reshape_node(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        self.assertIsInstance(sn, ReshapeNode)

    def test_unsqueeze_creates_reshape_node(self):
        sn = self._make_unsqueeze([1, 4], [1, 4, 1, 1])
        self.assertIsInstance(sn, ReshapeNode)

    def test_squeeze_kernel_name_empty(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        self.assertEqual(sn.kernel_name, "")

    def test_squeeze_emit_call_empty(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        self.assertEqual(sn.emit_call({}), "")

    def test_squeeze_emit_comment_says_squeeze(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        self.assertIn("Squeeze", sn.emit_comment())

    def test_unsqueeze_emit_comment_says_unsqueeze(self):
        sn = self._make_unsqueeze([1, 4], [1, 4, 1, 1])
        self.assertIn("Unsqueeze", sn.emit_comment())

    def test_squeeze_emit_comment_contains_alias(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        self.assertIn("alias", sn.emit_comment())

    def test_squeeze_emit_comment_contains_shapes(self):
        sn = self._make_squeeze([1, 4, 1, 1], [1, 4])
        comment = sn.emit_comment()
        self.assertIn("[1, 4, 1, 1]", comment)
        self.assertIn("[1, 4]", comment)

    def test_squeeze_numel_mismatch_raises(self):
        with self.assertRaises(SchedulerError) as cm:
            self._make_squeeze([1, 4, 1, 1], [1, 8])   # 4 ≠ 8
        self.assertIn("numel", str(cm.exception).lower())


# ═══════════════════════════════════════════════════════════════════
# Squeeze / Unsqueeze: graph dispatch and generated code
# ═══════════════════════════════════════════════════════════════════

@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestSqueezeGraphDispatch(unittest.TestCase):
    """OnnxGraph dispatches Squeeze/Unsqueeze → ReshapeNode."""

    def test_squeeze_then_matmul_node_types(self):
        g = OnnxGraph(_m("squeeze_then_matmul.onnx"))
        types = [type(sn) for sn in g.nodes]
        self.assertIn(PoolNode,    types)   # GlobalAveragePool
        self.assertIn(ReshapeNode, types)   # Squeeze
        self.assertIn(MatmulNode,  types)

    def test_squeeze_node_is_reshape_node(self):
        g  = OnnxGraph(_m("squeeze_then_matmul.onnx"))
        sn = next(n for n in g.nodes if isinstance(n, ReshapeNode))
        self.assertEqual(sn.onnx_node.op_type, "Squeeze")

    def test_squeeze_input_output_shapes(self):
        g  = OnnxGraph(_m("squeeze_then_matmul.onnx"))
        sn = next(n for n in g.nodes if isinstance(n, ReshapeNode))
        self.assertEqual(sn.inputs[0].shape, [1, 4, 1, 1])
        self.assertEqual(sn.output.shape,    [1, 4])

    def test_unsqueeze_then_relu_node_types(self):
        g = OnnxGraph(_m("unsqueeze_then_relu.onnx"))
        types = [type(sn) for sn in g.nodes]
        self.assertIn(ReshapeNode,   types)   # Unsqueeze
        self.assertIn(ScheduledNode, types)   # Relu

    def test_unsqueeze_node_is_reshape_node(self):
        g  = OnnxGraph(_m("unsqueeze_then_relu.onnx"))
        sn = next(n for n in g.nodes if isinstance(n, ReshapeNode))
        self.assertEqual(sn.onnx_node.op_type, "Unsqueeze")

    def test_squeeze_then_matmul_alias_present(self):
        gen = _gen("squeeze_then_matmul.onnx")
        s   = gen.generate_source()
        self.assertIn("reshape alias", s)

    def test_squeeze_comment_says_squeeze_in_source(self):
        gen = _gen("squeeze_then_matmul.onnx")
        s   = gen.generate_source()
        self.assertIn("Squeeze(", s)

    def test_unsqueeze_comment_says_unsqueeze_in_source(self):
        gen = _gen("unsqueeze_then_relu.onnx")
        s   = gen.generate_source()
        self.assertIn("Unsqueeze(", s)


@unittest.skipUnless(_models_exist(), "Run test/gen_reshape_gemm_models.py first")
class TestSqueezeSimulate(unittest.TestCase):
    """Squeeze/Unsqueeze simulation preserves values."""

    def test_squeeze_then_matmul_output_shape(self):
        gen    = _gen("squeeze_then_matmul.onnx")
        arrays = gen._simulate()
        self.assertEqual(arrays["Y"].shape, (1, 8))

    def test_squeeze_then_matmul_correct(self):
        """X[1,4,4,4]=1.0 → GAP → [1,4,1,1]=1.0 → Squeeze → [1,4] → MatMul(W=0.25) → Y=1.0."""
        gen = _gen("squeeze_then_matmul.onnx")
        x_q = gen._dtype.quantize(np.ones((1, 4, 4, 4), dtype=np.float64))
        y   = gen.simulate({"X": x_q})["Y"]
        np.testing.assert_allclose(y, 1.0, atol=0.02)

    def test_unsqueeze_then_relu_output_shape(self):
        gen    = _gen("unsqueeze_then_relu.onnx")
        arrays = gen._simulate()
        self.assertEqual(arrays["Y"].shape, (1, 4, 1, 1))

    def test_unsqueeze_preserves_values(self):
        """Unsqueeze just reinterprets memory — values are identical after Relu(positive)."""
        gen = _gen("unsqueeze_then_relu.onnx")
        x_q = gen._dtype.quantize(np.ones((1, 4), dtype=np.float64))
        y   = gen.simulate({"X": x_q})["Y"]
        np.testing.assert_allclose(y.flatten(), 1.0, atol=0.01)

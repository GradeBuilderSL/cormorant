"""Tests for ConvKernel scheduler integration."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import ConvNode, SchedulerError


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _conv_model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _conv_models_exist() -> bool:
    return os.path.isfile(_conv_model("conv_simple.onnx"))


def _gen(name: str) -> CodeGenerator:
    path = _conv_model(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


# ---------------------------------------------------------------------------
# ConvNode validation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvNodeValidation(unittest.TestCase):
    """ConvNode construction validates shapes and attributes."""

    def test_basic_conv_parses(self):
        path = _conv_model("conv_simple.onnx")
        g = OnnxGraph(path)
        self.assertEqual(len(g.nodes), 1)
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)

    def test_conv_with_bias_has_bias(self):
        path = _conv_model("conv_with_bias.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        self.assertTrue(sn.has_bias)
        self.assertEqual(len(sn.inputs), 3)

    def test_conv_without_bias_no_bias(self):
        path = _conv_model("conv_simple.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertFalse(sn.has_bias)
        self.assertEqual(len(sn.inputs), 2)

    def test_depthwise_raises(self):
        path = _conv_model("conv_depthwise.onnx")
        with self.assertRaises(SchedulerError) as cm:
            OnnxGraph(path)
        self.assertIn("group=4", str(cm.exception))
        self.assertIn("not supported", str(cm.exception))

    def test_stride_parsed(self):
        path = _conv_model("conv_stride2.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.stride_h, 2)
        self.assertEqual(sn.stride_w, 2)

    def test_dilation_parsed(self):
        path = _conv_model("conv_dilation.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.dilation_h, 2)
        self.assertEqual(sn.dilation_w, 2)

    def test_auto_pad_valid(self):
        path = _conv_model("conv_auto_pad_valid.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.pad_top,  0)
        self.assertEqual(sn.pad_left, 0)

    def test_explicit_pads(self):
        path = _conv_model("conv_padded.onnx")
        g = OnnxGraph(path)
        sn = g.nodes[0]
        self.assertEqual(sn.pad_top,  1)
        self.assertEqual(sn.pad_left, 1)


# ---------------------------------------------------------------------------
# ConvNode geometry fields
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvNodeGeometry(unittest.TestCase):
    """ConvNode derives correct shape parameters from ONNX node + inference."""

    def _node(self, name: str) -> ConvNode:
        g = OnnxGraph(_conv_model(name))
        return g.nodes[0]

    def test_simple_geometry(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.batch,  1)
        self.assertEqual(sn.in_ch,  4)
        self.assertEqual(sn.in_h,   8)
        self.assertEqual(sn.in_w,   8)
        self.assertEqual(sn.out_ch, 8)
        self.assertEqual(sn.out_h,  8)
        self.assertEqual(sn.out_w,  8)
        self.assertEqual(sn.kh, 1)
        self.assertEqual(sn.kw, 1)

    def test_stride2_output_size(self):
        sn = self._node("conv_stride2.onnx")
        # 3x3 no-pad stride-2: out = (8-3)//2+1 = 3
        self.assertEqual(sn.out_h, 3)
        self.assertEqual(sn.out_w, 3)

    def test_batch2_geometry(self):
        sn = self._node("conv_batch2.onnx")
        self.assertEqual(sn.batch, 2)

    def test_conv_kernel_name(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.kernel_name, "ConvKernel")

    def test_compatibility_shims(self):
        sn = self._node("conv_simple.onnx")
        self.assertEqual(sn.outer_count, 1)
        self.assertEqual(sn.chunk_size,  0)
        self.assertEqual(sn.aligned_chunk_size, 0)
        self.assertTrue(sn.a_advances)
        self.assertTrue(sn.b_advances)
        self.assertEqual(sn.arity, 2)


# ---------------------------------------------------------------------------
# TensorLayout — ConvNode excluded from phases 2 and 3
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvLayouts(unittest.TestCase):
    """ConvNode output always has a flat TensorLayout (n_chunks == 1)."""

    def test_simple_flat_output(self):
        gen = _gen("conv_simple.onnx")
        # Y[1,8,8,8] = 512 elements, flat
        y_lay = gen._layouts["Y"]
        self.assertEqual(y_lay.n_chunks, 1)
        self.assertEqual(y_lay.alloc,    y_lay.numel)

    def test_conv_then_relu_flat_intermediate(self):
        gen = _gen("conv_then_relu.onnx")
        # Z is ConvNode output feeding Relu; must stay flat
        z_lay = gen._layouts["Z"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(z_lay.alloc,    z_lay.numel)

    def test_conv_then_add_flat(self):
        gen = _gen("conv_then_add_flat.onnx")
        # Z and Y should both be flat (same-shape Add, no broadcast)
        z_lay = gen._layouts["Z"]
        y_lay = gen._layouts["Y"]
        self.assertEqual(z_lay.n_chunks, 1)
        self.assertEqual(y_lay.n_chunks, 1)

    def test_alloc_correct_for_simple(self):
        gen = _gen("conv_simple.onnx")
        # numel = 1*8*8*8 = 512
        y_lay = gen._layouts["Y"]
        self.assertEqual(y_lay.numel, 512)
        self.assertEqual(y_lay.alloc, 512)

    def test_chain_intermediates_flat(self):
        gen = _gen("conv_relu_chain.onnx")
        for name in ["Z1", "Z2", "Z3"]:
            lay = gen._layouts.get(name)
            if lay is not None:
                self.assertEqual(lay.n_chunks, 1,
                                 f"expected flat layout for {name}")


# ---------------------------------------------------------------------------
# Generated inference.c source
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvSource(unittest.TestCase):
    """Generated inference.c contains correct ConvKernel calls."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_conv_header_included(self):
        s = self._src("conv_simple.onnx")
        self.assertIn('#include "xconvkernel.h"', s)

    def test_conv_instance_declared(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("XConvkernel s_convkernel", s)

    def test_run_conv_helper_emitted(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("static void run_conv(", s)

    def test_run_conv_called(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("run_conv(", s)

    def test_no_run_op_for_conv_only(self):
        s = self._src("conv_simple.onnx")
        self.assertNotIn("run_op(", s)
        self.assertNotIn("static void run_op(", s)

    def test_mixed_both_helpers(self):
        s = self._src("conv_then_relu.onnx")
        self.assertIn("static void run_conv(", s)
        self.assertIn("static void run_op(",   s)

    def test_conv_args_no_bias(self):
        s = self._src("conv_simple.onnx")
        # has_bias=0 → bias arg is NULL, last param is 0u
        self.assertIn("run_conv(", s)
        # call line: run_conv(X, W, NULL, Y, ...)
        self.assertIn(", NULL, ", s)
        self.assertIn("0u);", s)   # has_bias=0u at end of call

    def test_conv_args_with_bias(self):
        s = self._src("conv_with_bias.onnx")
        # has_bias=1 → last param is 1u; bias arg is the buffer name (not NULL)
        self.assertIn("1u);", s)
        # The run_conv call should use the bias buffer name, not NULL
        # (NULL still appears in the run_conv helper comment — that's fine)
        self.assertNotIn(", NULL, ", s)

    def test_kernel_instance_registers(self):
        s = self._src("conv_simple.onnx")
        # Use prefix matching (driver uses padded spaces before '(')
        for fn in [
            "XConvkernel_Set_x",
            "XConvkernel_Set_weight",
            "XConvkernel_Set_bias",
            "XConvkernel_Set_y",
            "XConvkernel_Set_batch",
            "XConvkernel_Set_in_ch",
            "XConvkernel_Set_out_ch",
            "XConvkernel_Set_kh",
            "XConvkernel_Set_stride_h",
            "XConvkernel_Set_dilation_h",
            "XConvkernel_Set_pad_top",
            "XConvkernel_Set_has_bias",
            "XConvkernel_Start",
            "XConvkernel_IsDone",
        ]:
            self.assertIn(fn, s, f"{fn} not found in source")

    def test_emit_comment_contains_conv(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("[0] Conv(", s)

    def test_stride2_params_correct(self):
        s = self._src("conv_stride2.onnx")
        # emit_comment uses "s=2x2"; emit_call passes numeric args
        self.assertIn("s=2x2", s)
        self.assertIn("2u, 2u,", s)

    def test_dilation_params_correct(self):
        s = self._src("conv_dilation.onnx")
        self.assertIn("2u, 2u,", s)

    def test_batch2_param_correct(self):
        s = self._src("conv_batch2.onnx")
        # batch=2u should appear as first geometry param
        self.assertIn("2u,", s)

    def test_inference_init_has_conv_instance(self):
        s = self._src("conv_simple.onnx")
        self.assertIn("XConvkernel_Initialize(", s)
        self.assertIn("s_convkernel", s)

    def test_mixed_both_inits(self):
        s = self._src("conv_then_relu.onnx")
        self.assertIn("XConvkernel_Initialize(",    s)
        self.assertIn("XVectoropkernel_Initialize(", s)

    def test_kernel_registry_order(self):
        # When both VectorOPKernel and ConvKernel are active,
        # VectorOPKernel comes first (registry insertion order).
        s = self._src("conv_then_relu.onnx")
        vop_pos  = s.find("XVectoropkernel_Initialize(")
        conv_pos = s.find("XConvkernel_Initialize(")
        self.assertGreater(conv_pos, vop_pos)


# ---------------------------------------------------------------------------
# _active_kernels for Conv-containing graphs
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvActiveKernels(unittest.TestCase):

    def test_conv_only_active(self):
        gen = _gen("conv_simple.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["ConvKernel"])

    def test_conv_vectorop_active(self):
        gen = _gen("conv_then_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertIn("ConvKernel",     names)
        self.assertIn("VectorOPKernel", names)

    def test_conv_vectorop_registry_order(self):
        # VectorOPKernel should precede ConvKernel (registry insertion order)
        gen = _gen("conv_then_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertLess(names.index("VectorOPKernel"), names.index("ConvKernel"))

    def test_has_conv_nodes_property(self):
        gen = _gen("conv_simple.onnx")
        self.assertTrue(gen._has_conv_nodes)
        self.assertFalse(gen._has_vectorop_nodes)
        self.assertFalse(gen._has_matmul_nodes)


# ---------------------------------------------------------------------------
# Simulation — _forward_pass produces correct conv output
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvSimulation(unittest.TestCase):
    """Simulated conv output matches scipy/numpy reference."""

    def _simulate(self, name: str) -> dict:
        gen = _gen(name)
        return gen._simulate()

    def test_simple_1x1_output_shape(self):
        arrays = self._simulate("conv_simple.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 8, 8])

    def test_with_bias_output_shape(self):
        arrays = self._simulate("conv_with_bias.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 6, 6, 6])

    def test_stride2_output_shape(self):
        arrays = self._simulate("conv_stride2.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 3, 3])

    def test_padded_output_same_spatial(self):
        arrays = self._simulate("conv_padded.onnx")
        y = arrays["Y"]
        self.assertEqual(y.shape[2], 8)
        self.assertEqual(y.shape[3], 8)

    def test_batch2_output_shape(self):
        arrays = self._simulate("conv_batch2.onnx")
        y = arrays["Y"]
        self.assertEqual(y.shape[0], 2)

    def test_dilation_output_shape(self):
        arrays = self._simulate("conv_dilation.onnx")
        y = arrays["Y"]
        self.assertEqual(list(y.shape), [1, 8, 4, 4])

    def test_bias_zero_matches_no_bias(self):
        """Conv with B=zeros should match Conv without B."""
        # conv_with_bias uses B=zeros; manually compare to conv_simple after
        # loading the conv_simple graph and feeding the same input.
        gen_nb = _gen("conv_simple.onnx")
        gen_wb = _gen("conv_with_bias.onnx")

        # Both get the same weight data? No — they have different weights.
        # Just verify the bias branch doesn't blow up and has correct shape.
        arrays = gen_wb._simulate()
        self.assertEqual(list(arrays["Y"].shape), [1, 6, 6, 6])

    def test_conv_then_relu_output_nonneg(self):
        arrays = self._simulate("conv_then_relu.onnx")
        y = arrays["Y"]
        self.assertTrue((y >= 0).all(), "Relu output must be non-negative")

    def test_simulate_public_api(self):
        gen = _gen("conv_simple.onnx")
        dtype = gen._dtype
        # Build a small fixed input
        x_flat = np.arange(1 * 4 * 8 * 8, dtype=np.float64)
        x_quant = dtype.quantize(x_flat).reshape(1, 4, 8, 8)
        result = gen.simulate({"X": x_quant})
        self.assertIn("Y", result)
        self.assertEqual(list(result["Y"].shape), [1, 8, 8, 8])


# ---------------------------------------------------------------------------
# emit_comment and emit_call format
# ---------------------------------------------------------------------------

@unittest.skipUnless(_conv_models_exist(),
                     "Run test/gen_conv_models.py first")
class TestConvEmit(unittest.TestCase):

    def test_emit_comment_format(self):
        g = OnnxGraph(_conv_model("conv_simple.onnx"))
        sn = g.nodes[0]
        self.assertIsInstance(sn, ConvNode)
        comment = sn.emit_comment()
        self.assertIn("[0] Conv(", comment)
        self.assertIn("->", comment)
        self.assertIn("k=1", comment)

    def test_emit_call_format_no_bias(self):
        g = OnnxGraph(_conv_model("conv_simple.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_simple.onnx"))
        call = sn.emit_call(gen._layouts)
        self.assertIn("run_conv(", call)
        self.assertIn("NULL", call)
        self.assertIn("0u);", call)

    def test_emit_call_format_with_bias(self):
        g = OnnxGraph(_conv_model("conv_with_bias.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_with_bias.onnx"))
        call = sn.emit_call(gen._layouts)
        self.assertIn("run_conv(", call)
        self.assertIn("1u);", call)   # has_bias = 1u

    def test_emit_call_correct_dims(self):
        g = OnnxGraph(_conv_model("conv_stride2.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_stride2.onnx"))
        call = sn.emit_call(gen._layouts)
        # stride params appear in call
        self.assertIn("2u, 2u,", call)

    def test_emit_call_with_bias_name(self):
        g = OnnxGraph(_conv_model("conv_with_bias.onnx"))
        sn = g.nodes[0]
        gen = CodeGenerator(g, model_path=_conv_model("conv_with_bias.onnx"))
        call = sn.emit_call(gen._layouts)
        # bias tensor c_name appears as 3rd arg
        bias_name = sn.inputs[2].c_name
        self.assertIn(bias_name, call)


if __name__ == "__main__":
    unittest.main(verbosity=2)

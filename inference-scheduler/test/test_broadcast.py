"""Tests for broadcasting model handling (TestBroadcast)."""

import os
import sys
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import SchedulerError


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcast(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # ---------------------------------------------------------------- #
    # broadcast_aligned: X[8,16] + bias[16] -> Y[8,16]                 #
    # chunk=16 elems × 2 bytes = 32 B — already a multiple of 16 B     #
    # ---------------------------------------------------------------- #

    def test_aligned_outer_count(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.outer_count, 8)

    def test_aligned_chunk_size(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.chunk_size, 16)

    def test_aligned_no_padding(self):
        # chunk_size already a multiple of _ALIGN_ELEMS → no gap
        from src.nodes import _ALIGN_ELEMS
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, sn.chunk_size)
        self.assertEqual(sn.chunk_size % _ALIGN_ELEMS, 0)

    def test_aligned_x_advances(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertTrue(sn.a_advances)   # X is the advancing input
        self.assertFalse(sn.b_advances)  # bias repeats

    def test_aligned_chunk_macro_in_header(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_CHUNK", h)
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_chunk_stride_uses_align_up(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_ALIGN_UP(INFERENCE_Y_CHUNK)", h)

    def test_aligned_x_size_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        # X is an advancing buffer — SIZE must reference CHUNK_STRIDE
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_X_SIZE", h)
        # The SIZE expression must multiply outer_count by CHUNK_STRIDE
        self.assertIn("8u * INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_y_size_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("8u * INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_run_op_at_in_source(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("run_op_at(", s)

    def test_aligned_for_loop_in_source(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("for (unsigned _i = 0u; _i < 8u; _i++)", s)

    def test_aligned_loop_uses_chunk_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK,", s)

    def test_aligned_loop_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", s)

    def test_aligned_advancing_uses_stride_offset(self):
        # X (advancing) offset must be _i * CHUNK_STRIDE
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("_i * INFERENCE_Y_CHUNK_STRIDE", s)

    def test_aligned_repeating_offset_zero(self):
        # bias (repeating) offset must be 0u
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("bias, 0u", s)

    def test_aligned_explicit_syncs_in_source(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("inference_buf_sync_to_device(X)", s)
        self.assertIn("inference_buf_sync_from_device(Y)", s)

    def test_aligned_no_plain_run_op_for_broadcast_node(self):
        # The broadcast node must NOT emit the no-offset run_op()
        import re
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        plain_calls = re.findall(r'\brun_op\(X,', s)
        self.assertEqual(len(plain_calls), 0)

    def test_aligned_bias_alloc_uses_aligned_chunk(self):
        # bias (repeating) is allocated with aligned_chunk_size elements
        from src.nodes import _ALIGN_ELEMS
        g, cg = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        alloc_sizes = cg._alloc_sizes
        bias_name = sn.inputs[1].onnx_name  # bias is b input
        expected = sn.aligned_chunk_size
        self.assertEqual(alloc_sizes[bias_name], expected)

    def test_aligned_x_alloc_uses_total_padded(self):
        g, cg = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        x_name = sn.inputs[0].onnx_name
        expected = sn.outer_count * sn.aligned_chunk_size
        self.assertEqual(cg._alloc_sizes[x_name], expected)

    def test_aligned_alignment_macros_in_header(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_ALIGN_BYTES", h)
        self.assertIn("INFERENCE_ALIGN_ELEMS", h)
        self.assertIn("INFERENCE_ALIGN_UP", h)

    # ---------------------------------------------------------------- #
    # broadcast_unaligned: X[4,6] + bias[6] -> Y[4,6]                  #
    # chunk=6 elems × 2 bytes = 12 B — NOT a multiple of 16 B          #
    # aligned_chunk = 8 (gap of 2 elements per block)                   #
    # ---------------------------------------------------------------- #

    def test_unaligned_outer_count(self):
        g, _ = self._gen("broadcast_unaligned.onnx")
        self.assertEqual(g.nodes[0].outer_count, 4)

    def test_unaligned_chunk_size(self):
        g, _ = self._gen("broadcast_unaligned.onnx")
        self.assertEqual(g.nodes[0].chunk_size, 6)

    def test_unaligned_aligned_chunk_has_gap(self):
        from src.nodes import _ALIGN_ELEMS
        g, _ = self._gen("broadcast_unaligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, 8)
        self.assertGreater(sn.aligned_chunk_size, sn.chunk_size)
        self.assertEqual(sn.aligned_chunk_size % _ALIGN_ELEMS, 0)

    def test_unaligned_chunk_macro_value(self):
        # CHUNK macro must hold the raw chunk_size (6), not the padded stride
        _, cg = self._gen("broadcast_unaligned.onnx")
        h = cg.generate_header()
        import re
        m = re.search(r'#define\s+INFERENCE_Y_CHUNK\s+(\d+)u', h)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 6)

    def test_unaligned_x_size_larger_than_numel(self):
        # With padding, X needs more than its natural numel
        g, cg = self._gen("broadcast_unaligned.onnx")
        sn = g.nodes[0]
        x = g.input_tensors[0]
        alloc = cg._alloc_sizes[x.onnx_name]
        self.assertGreater(alloc, x.numel)      # 32 > 24
        self.assertEqual(alloc, sn.outer_count * sn.aligned_chunk_size)  # 4*8=32

    def test_unaligned_for_loop_in_source(self):
        _, cg = self._gen("broadcast_unaligned.onnx")
        s = cg.generate_source()
        self.assertIn("for (unsigned _i = 0u; _i < 4u; _i++)", s)

    def test_unaligned_run_op_size_is_chunk_not_stride(self):
        # The 'size' argument to run_op_at must be INFERENCE_Y_CHUNK (the raw
        # data count), not INFERENCE_Y_CHUNK_STRIDE (the padded stride).
        # In the call: run_op_at(..., INFERENCE_Y_CHUNK, VECTOROP_ADD)
        # so CHUNK is immediately followed by ", VECTOROP_".
        _, cg = self._gen("broadcast_unaligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK, VECTOROP_", s)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE, VECTOROP_", s)

    # ---------------------------------------------------------------- #
    # Alignment macros are emitted even for non-broadcast models        #
    # ---------------------------------------------------------------- #

    def test_alignment_macros_always_present(self):
        for model in ("single_add.onnx", "relu_chain.onnx", "two_input_add.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                h = cg.generate_header()
                self.assertIn("INFERENCE_ALIGN_BYTES", h)
                self.assertIn("INFERENCE_ALIGN_ELEMS", h)
                self.assertIn("INFERENCE_ALIGN_UP",    h)

    # ---------------------------------------------------------------- #
    # Non-broadcast models still use plain run_op() (no regression)    #
    # ---------------------------------------------------------------- #

    def test_no_broadcast_uses_run_op_not_run_op_at(self):
        for model in ("single_add.onnx", "relu_chain.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                s = cg.generate_source()
                self.assertIn("run_op(", s)
                # run_op_at must not appear at all in non-broadcast source
                # (the function is not emitted, eliminating the unused-function warning).
                self.assertNotIn("run_op_at", s)

    # ---------------------------------------------------------------- #
    # SchedulerError for invalid broadcast shapes                       #
    # ---------------------------------------------------------------- #

    def test_interleaved_broadcast_raises(self):
        """[N,1,H,W] against [N,C,H,W]: broadcast dim is flanked by matches."""
        import numpy as np
        import onnx.helper as oh
        import onnx.numpy_helper as nph
        from onnx import TensorProto

        N, C, H, W = 2, 3, 4, 4
        # bias has shape [N,1,H,W] — dim-1 is 1 (broadcasts to C), others match
        bias_data = np.ones([N, 1, H, W], dtype=np.float32)
        X_info    = oh.make_tensor_value_info("X",    TensorProto.FLOAT, [N, C, H, W])
        Y_info    = oh.make_tensor_value_info("Y",    TensorProto.FLOAT, [N, C, H, W])
        bias_init = nph.from_array(bias_data, name="bias")
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", "bias"], ["Y"])],
            "bad_broadcast",
            inputs=[X_info],
            outputs=[Y_info],
            initializer=[bias_init],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model.ir_version = 8
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fname = f.name
        try:
            import onnx
            onnx.save(model, fname)
            with self.assertRaises(SchedulerError):
                OnnxGraph(fname)
        finally:
            os.unlink(fname)

    def test_both_inputs_smaller_raises(self):
        """Both A and B smaller than the output — not supported."""
        import numpy as np
        import onnx.helper as oh
        import onnx.numpy_helper as nph
        from onnx import TensorProto

        # A[4] + B[4] -> Y[4] is fine (no broadcast).
        # To get both inputs smaller than output we need the output to be larger.
        # Use A[4] of shape [4] and B[4] of shape [4], but declare output [2,4].
        # ONNX broadcast: [4] + [4] -> [4], not [2,4], so we use a different trick:
        # A[1,4] + B[2,1] -> Y[2,4] — both A and B are smaller than output.
        A_info = oh.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        B_info = oh.make_tensor_value_info("B", TensorProto.FLOAT, [2, 1])
        Y_info = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        graph = oh.make_graph(
            [oh.make_node("Add", ["A", "B"], ["Y"])],
            "both_broadcast",
            inputs=[A_info, B_info],
            outputs=[Y_info],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model.ir_version = 8
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fname = f.name
        try:
            import onnx
            onnx.save(model, fname)
            with self.assertRaises(SchedulerError):
                OnnxGraph(fname)
        finally:
            os.unlink(fname)

    # ---------------------------------------------------------------- #
    # generate_test() — broadcast-aware fill and print                  #
    # ---------------------------------------------------------------- #

    def test_broadcast_fill_uses_chunk_loop(self):
        """Input fill uses a chunk-by-chunk loop, not a flat SIZE loop."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # chunk loop present
        self.assertIn("for (_chunk = 0u; _chunk < 4u; _chunk++)", t)
        self.assertIn("_off = _chunk * INFERENCE_Y_CHUNK_STRIDE", t)
        self.assertIn("for (i = 0u; i < INFERENCE_Y_CHUNK; i++)", t)
        # flat SIZE loop must NOT be used for the fill
        self.assertNotIn("for (i = 0u; i < INFERENCE_X_SIZE; i++)", t)

    def test_broadcast_fill_comment(self):
        """Fill block is annotated as broadcast."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)

    def test_broadcast_print_uses_chunk_loop(self):
        """Output print uses a chunk-by-chunk loop, not a flat SIZE loop."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # chunk print loop present
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)
        self.assertIn("_off = _chunk * INFERENCE_Y_CHUNK_STRIDE", t)
        self.assertIn("INFERENCE_Y_CHUNK && shown < lim", t)
        # flat SIZE loop must NOT be used for the print
        self.assertNotIn("for (i = 0u; i < lim; i++)", t)

    def test_broadcast_print_comment(self):
        """Print block is annotated as broadcast."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("broadcast output, print data chunks only", t)

    def test_broadcast_print_logical_numel(self):
        """printf reports outer_count * CHUNK (logical numel), not SIZE."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # numel computed as 4 * CHUNK, not from SIZE
        self.assertIn("numel = 4u * INFERENCE_Y_CHUNK", t)
        # SIZE must not appear in printf numel context
        self.assertNotIn("(unsigned)INFERENCE_Y_SIZE", t)

    def test_broadcast_print_shown_counter(self):
        """Print loop uses 'shown' to count valid elements printed."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("shown = 0u", t)
        self.assertIn("i++, shown++", t)
        self.assertIn("shown, (unsigned)p[_off + i]", t)

    def test_non_broadcast_test_unchanged(self):
        """Non-broadcast models still use the simple flat fill/print."""
        for model in ("single_add.onnx", "relu_chain.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                t = cg.generate_test()
                # flat fill
                self.assertIn("for (i = 0u; i < INFERENCE_X_SIZE; i++)", t)
                # flat print
                self.assertIn("for (i = 0u; i < lim; i++)", t)
                # no chunk machinery
                self.assertNotIn("_chunk", t)
                self.assertNotIn("_off", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

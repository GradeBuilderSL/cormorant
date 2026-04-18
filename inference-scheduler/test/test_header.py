"""Tests for header file generation (TestHeaderGen)."""

import os
import sys
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestHeaderGen(unittest.TestCase):

    def _header(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_header()

    def test_pragma_once(self):
        h = self._header("single_add.onnx")
        self.assertIn("#pragma once", h)

    def test_extern_c(self):
        h = self._header("single_add.onnx")
        self.assertIn('extern "C"', h)

    def test_data_t_typedef(self):
        h = self._header("single_add.onnx")
        self.assertIn("typedef uint16_t Data_t", h)

    def test_bytes_per_elem_macro(self):
        h = self._header("single_add.onnx")
        self.assertIn("INFERENCE_BYTES_PER_ELEM", h)

    def test_size_macros_present(self):
        h = self._header("single_add.onnx")
        # single_add model has input 'X' and output 'Y'
        self.assertIn("INFERENCE_X_SIZE", h)
        self.assertIn("INFERENCE_Y_SIZE", h)

    def test_init_declaration(self):
        h = self._header("single_add.onnx")
        # Two spaces between 'int' and name (aligned with 'void inference_deinit')
        self.assertIn("inference_init(", h)

    def test_run_declaration(self):
        h = self._header("single_add.onnx")
        self.assertIn("void inference_run(", h)

    def test_no_implementation_in_header(self):
        # Header must not contain function bodies
        h = self._header("relu_chain.onnx")
        self.assertNotIn("XVectoropkernel_Start", h)
        self.assertNotIn("static void run_op", h)
        self.assertNotIn("static const uint16_t", h)

    def test_inference_buf_type_in_header(self):
        h = self._header("single_add.onnx")
        self.assertIn("inference_buf_t", h)

    def test_inference_buf_alloc_in_header(self):
        h = self._header("single_add.onnx")
        self.assertIn("inference_buf_alloc", h)

    def test_pool_size_macro_in_header(self):
        h = self._header("single_add.onnx")
        self.assertIn("INFERENCE_BUF_POOL_SIZE_BYTES", h)

    def test_pool_size_is_positive(self):
        import re
        h = self._header("single_add.onnx")
        m = re.search(r"INFERENCE_BUF_POOL_SIZE_BYTES\s+(\d+)u", h)
        self.assertIsNotNone(m, "INFERENCE_BUF_POOL_SIZE_BYTES not found")
        self.assertGreater(int(m.group(1)), 0)

    def test_run_uses_buf_type(self):
        # inference_run must accept inference_buf_t* not raw Data_t*
        h = self._header("single_add.onnx")
        self.assertIn("inference_buf_t *", h)
        # Confirm the run declaration is present
        self.assertIn("void inference_run(", h)

    def test_deinit_declaration(self):
        h = self._header("single_add.onnx")
        self.assertIn("inference_deinit(void)", h)

    def test_fill_float_declaration_in_header(self):
        h = self._header("single_add.onnx")
        self.assertIn("inference_buf_fill_float(", h)

    def test_read_float_declaration_in_header(self):
        h = self._header("single_add.onnx")
        self.assertIn("inference_buf_read_float(", h)


if __name__ == "__main__":
    unittest.main(verbosity=2)

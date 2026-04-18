"""Tests for inference_buf.c buffer implementation generation (TestBufImpl)."""

import os
import sys
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBufImpl(unittest.TestCase):

    def _buf(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_buf_impl()

    def test_struct_definition(self):
        b = self._buf("single_add.onnx")
        self.assertIn("struct inference_buf", b)

    def test_linux_xrt_alloc(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclAllocBO(", b)

    def test_linux_xrt_map(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclMapBO(", b)

    def test_linux_xrt_sync(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclSyncBO(", b)
        self.assertIn("XCL_BO_SYNC_BO_TO_DEVICE", b)
        self.assertIn("XCL_BO_SYNC_BO_FROM_DEVICE", b)

    def test_linux_xrt_paddr(self):
        b = self._buf("single_add.onnx")
        self.assertIn("xclGetBOProperties(", b)
        self.assertIn("props.paddr", b)

    def test_alloc_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_alloc(", b)

    def test_phys_equals_virt_on_baremetal(self):
        # Bare-metal: physical address = cast of virtual pointer
        b = self._buf("single_add.onnx")
        self.assertIn("(uint64_t)(uintptr_t)", b)

    def test_pool_init_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_pool_init(", b)

    def test_pool_deinit_function(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_pool_deinit(", b)

    def test_accessors_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_ptr(", b)
        self.assertIn("inference_buf_phys(", b)
        self.assertIn("inference_buf_count(", b)

    def test_sync_functions_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_sync_to_device(", b)
        self.assertIn("inference_buf_sync_from_device(", b)

    def test_bare_metal_cache_api_in_buf(self):
        b = self._buf("single_add.onnx")
        self.assertIn("Xil_DCacheFlushRange", b)
        self.assertIn("Xil_DCacheInvalidateRange", b)

    def test_float_cast_fill_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_fill_float(", b)

    def test_float_cast_read_present(self):
        b = self._buf("single_add.onnx")
        self.assertIn("inference_buf_read_float(", b)

    def test_fill_float_uses_data_t_cast(self):
        # Must use (Data_t) cast, not a type-specific encoding constant
        b = self._buf("single_add.onnx")
        self.assertIn("(Data_t)src[i]", b)

    def test_read_float_uses_float_cast(self):
        b = self._buf("single_add.onnx")
        self.assertIn("(float)src[i]", b)

    def test_no_hardcoded_scale_in_buf(self):
        # Must not contain ap_fixed<16,8>-specific scaling constants
        b = self._buf("single_add.onnx")
        self.assertNotIn("256.0f", b)
        self.assertNotIn("127.99609375f", b)


if __name__ == "__main__":
    unittest.main(verbosity=2)

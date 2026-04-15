#!/usr/bin/env python3
"""
test_scheduler.py — unit tests for inference_scheduler.

Run from the inference-scheduler directory:
  python -m pytest test/test_scheduler.py -v
or directly:
  python test/test_scheduler.py
"""

import os
import sys
import shutil
import subprocess
import tempfile
import unittest

# Allow importing from inference-scheduler root
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import SchedulerError

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _models_exist() -> bool:
    required = ["single_add.onnx", "relu_chain.onnx", "mixed_ops.onnx", "unsupported.onnx"]
    return all(os.path.isfile(_model(m)) for m in required)


# ------------------------------------------------------------------ #
# Tensor encoding                                                     #
# ------------------------------------------------------------------ #

class TestTensorEncoding(unittest.TestCase):
    """ap_fixed<16,8> weight encoding."""

    def test_zero(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        self.assertEqual(_float_to_apfixed_hex(np.array([0.0])), ["0x0000"])

    def test_one(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # 1.0 * 256 = 256 = 0x0100
        self.assertEqual(_float_to_apfixed_hex(np.array([1.0])), ["0x0100"])

    def test_minus_one(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # -1.0 * 256 = -256 = 0xFF00 as uint16
        self.assertEqual(_float_to_apfixed_hex(np.array([-1.0])), ["0xFF00"])

    def test_half(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # 0.5 * 256 = 128 = 0x0080
        self.assertEqual(_float_to_apfixed_hex(np.array([0.5])), ["0x0080"])

    def test_saturation_positive(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex, _AP_FIXED_MAX
        hexvals  = _float_to_apfixed_hex(np.array([999.0]))
        expected = _float_to_apfixed_hex(np.array([_AP_FIXED_MAX]))
        self.assertEqual(hexvals, expected)

    def test_saturation_negative(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex, _AP_FIXED_MIN
        hexvals  = _float_to_apfixed_hex(np.array([-999.0]))
        expected = _float_to_apfixed_hex(np.array([_AP_FIXED_MIN]))
        self.assertEqual(hexvals, expected)


# ------------------------------------------------------------------ #
# Graph parsing                                                       #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestGraphParsing(unittest.TestCase):

    def test_single_add_nodes(self):
        g = OnnxGraph(_model("single_add.onnx"))
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(g.nodes[0].onnx_node.op_type, "Add")

    def test_single_add_weight(self):
        g = OnnxGraph(_model("single_add.onnx"))
        weights = g.weight_tensors
        self.assertEqual(len(weights), 1)
        self.assertTrue(weights[0].is_weight)

    def test_relu_chain_nodes(self):
        g = OnnxGraph(_model("relu_chain.onnx"))
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertEqual(ops, ["Add", "Relu"])

    def test_mixed_ops_nodes(self):
        g = OnnxGraph(_model("mixed_ops.onnx"))
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertEqual(ops, ["Add", "Mul", "Clip"])

    def test_mixed_ops_clip_is_relu6(self):
        from src.nodes import OP_RELU6
        g = OnnxGraph(_model("mixed_ops.onnx"))
        clip_node = g.nodes[2]
        self.assertEqual(clip_node.op_code, OP_RELU6)
        self.assertEqual(clip_node.arity, 1)

    def test_unsupported_raises(self):
        with self.assertRaises(SchedulerError):
            OnnxGraph(_model("unsupported.onnx"))


# ------------------------------------------------------------------ #
# Header generation                                                   #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Source generation                                                   #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSourceGen(unittest.TestCase):

    def _source(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_source()

    def test_includes_own_header(self):
        s = self._source("single_add.onnx")
        self.assertIn('#include "inference.h"', s)

    def test_includes_driver(self):
        s = self._source("single_add.onnx")
        self.assertIn("xvectoropkernel.h", s)
        self.assertIn("xil_cache.h", s)

    def test_no_data_t_typedef_in_source(self):
        # Data_t is defined in inference.h; source must not redefine it
        s = self._source("single_add.onnx")
        self.assertNotIn("typedef uint16_t Data_t", s)

    def test_bytes_per_elem_not_redefined(self):
        s = self._source("single_add.onnx")
        # BYTES_PER_ELEM in run_op body references the header's INFERENCE_BYTES_PER_ELEM
        self.assertNotIn("#define BYTES_PER_ELEM ", s)

    def test_op_defines_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("#define VECTOROP_ADD", s)

    def test_weight_array_present(self):
        s = self._source("single_add.onnx")
        self.assertIn("static const uint16_t", s)

    def test_run_op_helper(self):
        s = self._source("single_add.onnx")
        self.assertIn("static void run_op(", s)

    def test_platform_guard_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("#ifdef __linux__", s)
        # Both branches define these macros (Linux no-op, bare-metal Xil API)
        self.assertIn("INFERENCE_CACHE_FLUSH", s)
        self.assertIn("INFERENCE_CACHE_INVALIDATE", s)

    def test_cache_macros_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("INFERENCE_CACHE_FLUSH", s)
        self.assertIn("INFERENCE_CACHE_INVALIDATE", s)

    def test_bare_metal_cache_api_in_source(self):
        s = self._source("single_add.onnx")
        # Bare-metal branch must reference Xil API
        self.assertIn("Xil_DCacheFlushRange", s)
        self.assertIn("Xil_DCacheInvalidateRange", s)

    def test_linux_cache_noop_in_source(self):
        s = self._source("single_add.onnx")
        # Linux branch must have no-op macros
        self.assertIn("(void)(ptr)", s)

    def test_run_op_call_add(self):
        s = self._source("single_add.onnx")
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("run_op(", s)

    def test_relu_null_b(self):
        s = self._source("relu_chain.onnx")
        self.assertIn("VECTOROP_RELU", s)
        self.assertIn("NULL", s)

    def test_mixed_ops_relu6(self):
        s = self._source("mixed_ops.onnx")
        self.assertIn("VECTOROP_RELU6", s)

    def test_inference_run_body(self):
        s = self._source("single_add.onnx")
        self.assertIn("void inference_run(", s)

    def test_inference_init_body(self):
        s = self._source("single_add.onnx")
        self.assertIn("int inference_init(", s)
        self.assertIn("XVectoropkernel_Initialize", s)

    def test_weight_rom_prefix(self):
        # ROM arrays must use _rom_ prefix; buf pointer is separate
        s = self._source("single_add.onnx")
        self.assertIn("_rom_", s)

    def test_inference_buf_alloc_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_alloc(", s)

    def test_inference_buf_phys_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_phys(", s)

    def test_memcpy_in_source(self):
        # Weight ROM data must be copied into DMA buffers at init
        s = self._source("single_add.onnx")
        self.assertIn("memcpy(", s)

    def test_inference_deinit_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_deinit(", s)

    def test_pool_init_in_source(self):
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_pool_init(", s)

    def test_run_op_uses_physical_address(self):
        # run_op must use inference_buf_phys not a raw pointer cast
        s = self._source("single_add.onnx")
        self.assertIn("inference_buf_phys(a)", s)
        self.assertNotIn("(u64)(UINTPTR)", s)

    def test_source_includes_string_h(self):
        # memcpy requires <string.h>
        s = self._source("single_add.onnx")
        self.assertIn("<string.h>", s)


# ------------------------------------------------------------------ #
# Buffer implementation (inference_buf.c)                            #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBufImpl(unittest.TestCase):

    def _buf(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_buf_impl()

    def test_struct_definition(self):
        b = self._buf("single_add.onnx")
        self.assertIn("struct inference_buf", b)

    def test_linux_udmabuf_device(self):
        b = self._buf("single_add.onnx")
        self.assertIn("/dev/udmabuf0", b)

    def test_linux_mmap(self):
        b = self._buf("single_add.onnx")
        self.assertIn("mmap(", b)

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


# ------------------------------------------------------------------ #
# Setup script (scripts/setup_udmabuf.sh)                            #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSetupScript(unittest.TestCase):

    def _script(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_setup_script()

    def test_shebang(self):
        s = self._script("single_add.onnx")
        self.assertTrue(s.startswith("#!/bin/sh"))

    def test_modprobe_udmabuf(self):
        s = self._script("single_add.onnx")
        self.assertIn("modprobe u-dma-buf", s)

    def test_pool_size_in_script(self):
        import re
        from src.graph import OnnxGraph as G
        from src.codegen import CodeGenerator as CG
        g  = G(_model("single_add.onnx"))
        cg = CG(g, model_path=_model("single_add.onnx"))
        pool_bytes = cg._compute_pool_bytes()
        s = self._script("single_add.onnx")
        self.assertIn(str(pool_bytes), s)

    def test_dev_udmabuf0_referenced(self):
        s = self._script("single_add.onnx")
        self.assertIn("/dev/udmabuf0", s)


# ------------------------------------------------------------------ #
# Test source generation (test/test_inference.c)                     #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestTestGen(unittest.TestCase):

    def _test_src(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_test()

    def test_includes_stdio(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("#include <stdio.h>", t)

    def test_includes_inference_h(self):
        t = self._test_src("single_add.onnx")
        self.assertIn('#include "inference.h"', t)

    def test_main_signature(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("int main(int argc, char **argv)", t)

    def test_inference_init_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_init(", t)

    def test_inference_run_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_run(", t)

    def test_inference_deinit_called(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_deinit(", t)

    def test_input_buf_alloc(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_X_SIZE)", t)

    def test_output_buf_alloc(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_Y_SIZE)", t)

    def test_ramp_fill_pattern(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("i & 0xFFFF", t)

    def test_printf_output(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("printf(", t)

    def test_linux_default_uio(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("/dev/uio0", t)

    def test_baremetal_default_instance(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("VectorOPKernel", t)

    def test_passed_message(self):
        t = self._test_src("single_add.onnx")
        self.assertIn("test_inference PASSED", t)

    def test_multi_op_allocs(self):
        # relu_chain: input X, output relu_Y — both must be allocated
        t = self._test_src("relu_chain.onnx")
        self.assertIn("inference_buf_alloc(INFERENCE_X_SIZE)", t)
        # output is relu_Y → macro is INFERENCE_RELU_Y_SIZE
        self.assertIn("inference_buf_alloc(INFERENCE_RELU_Y_SIZE)", t)
        self.assertIn("inference_run(", t)


# ------------------------------------------------------------------ #
# CMake generation                                                    #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestCMakeGen(unittest.TestCase):

    def _cmake(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_cmake()

    def test_add_library_static(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("add_library(inference STATIC", cm)

    def test_inference_target_option(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("INFERENCE_TARGET", cm)
        self.assertIn("BARE_METAL", cm)
        self.assertIn("LINUX", cm)

    def test_bare_metal_driver_source(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("xvectoropkernel_sinit.c", cm)

    def test_linux_driver_source(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("xvectoropkernel_linux.c", cm)

    def test_inference_source_listed(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("src/inference.c", cm)

    def test_buf_impl_source_listed(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("src/inference_buf.c", cm)

    def test_test_executable_in_cmake(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("test_inference", cm)
        self.assertIn("INFERENCE_BUILD_TEST", cm)
        self.assertIn("test/test_inference.c", cm)

    def test_include_dirs(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("include", cm)
        self.assertIn("driver", cm)
        self.assertIn("BSP_INCLUDE_DIR", cm)

    def test_install_targets(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("install(TARGETS inference", cm)
        self.assertIn("install(FILES include/inference.h", cm)

    def test_driver_existence_check(self):
        cm = self._cmake("single_add.onnx")
        self.assertIn("FATAL_ERROR", cm)


# ------------------------------------------------------------------ #
# CLI — project directory output                                      #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestCLI(unittest.TestCase):

    def _run_cli(self, model_name: str, extra_args=None) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "inference_scheduler.py"),
            _model(model_name),
        ]
        if extra_args:
            cmd += extra_args
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_exit_zero_default_outdir(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("single_add.onnx", ["--out-dir", td])
            self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_unsupported_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("unsupported.onnx", ["--out-dir", td])
            self.assertNotEqual(r.returncode, 0)

    def test_project_files_created(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("relu_chain.onnx", ["--out-dir", td])
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(td, "CMakeLists.txt")))
            self.assertTrue(os.path.isfile(os.path.join(td, "include", "inference.h")))
            self.assertTrue(os.path.isfile(os.path.join(td, "src", "inference.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "src", "inference_buf.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "test", "test_inference.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "scripts", "setup_udmabuf.sh")))
            self.assertTrue(os.path.isdir(os.path.join(td, "driver")))

    def test_driver_readme_when_no_src_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            self.assertTrue(
                os.path.isfile(os.path.join(td, "driver", "README.md"))
            )

    def test_driver_copied_when_dir_given(self):
        driver_src = os.path.join(
            ROOT, "..", "build", "kv260",
            "vadd_kv260", "solution1", "impl", "ip",
            "drivers", "VectorOPKernel_v1_0", "src",
        )
        if not os.path.isdir(driver_src):
            self.skipTest("HLS synthesis output not present")
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli(
                "single_add.onnx",
                ["--out-dir", td, "--driver-dir", driver_src],
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            # Both bare-metal and Linux driver variants must be present
            for fname in ["xvectoropkernel.c", "xvectoropkernel.h",
                          "xvectoropkernel_hw.h",
                          "xvectoropkernel_sinit.c",
                          "xvectoropkernel_linux.c"]:
                self.assertTrue(
                    os.path.isfile(os.path.join(td, "driver", fname)),
                    msg=f"Missing driver file: {fname}",
                )

    def test_header_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "include", "inference.h")) as f:
                h = f.read()
            self.assertIn("#pragma once", h)
            self.assertIn("inference_run", h)

    def test_source_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("relu_chain.onnx", ["--out-dir", td])
            with open(os.path.join(td, "src", "inference.c")) as f:
                s = f.read()
            self.assertIn('#include "inference.h"', s)
            self.assertIn("VECTOROP_ADD", s)
            self.assertIn("VECTOROP_RELU", s)

    def test_buf_impl_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "src", "inference_buf.c")) as f:
                b = f.read()
            self.assertIn("inference_buf_alloc(", b)
            self.assertIn("/dev/udmabuf0", b)

    def test_setup_script_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "scripts", "setup_udmabuf.sh")) as f:
                s = f.read()
            self.assertIn("modprobe u-dma-buf", s)
            self.assertIn("POOL_BYTES=", s)

    def test_test_source_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "test", "test_inference.c")) as f:
                t = f.read()
            self.assertIn("inference_init(", t)
            self.assertIn("inference_run(", t)
            self.assertIn("inference_deinit(", t)
            self.assertIn("test_inference PASSED", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

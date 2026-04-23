"""Tests for multi-kernel graph support (MatmulKernel + VectorOPKernel)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.kernels  import KERNEL_REGISTRY


def _mixed_model(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "models", name)


def _mixed_models_exist() -> bool:
    return os.path.isfile(_mixed_model("mixed_matmul_relu.onnx"))


def _gen(name: str) -> CodeGenerator:
    path = _mixed_model(name)
    g    = OnnxGraph(path)
    return CodeGenerator(g, model_path=path)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelActive(unittest.TestCase):
    """_active_kernels returns the right KernelDesc objects."""

    def test_vectorop_only_active(self):
        path = _model("single_add.onnx") if _models_exist() else None
        if path is None:
            self.skipTest("VOP-only models not generated")
        gen = CodeGenerator(OnnxGraph(path), model_path=path)
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["VectorOPKernel"])

    def test_matmul_only_active(self):
        path = _model("mm_batch.onnx") if _models_exist() else None
        if path is None:
            self.skipTest("MatMul-only models not generated")
        gen = CodeGenerator(OnnxGraph(path), model_path=path)
        names = [kd.name for kd in gen._active_kernels]
        self.assertEqual(names, ["MatmulKernel"])

    def test_mixed_matmul_relu_both_active(self):
        gen = _gen("mixed_matmul_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertIn("VectorOPKernel", names)
        self.assertIn("MatmulKernel",   names)

    def test_active_kernel_order(self):
        """VectorOPKernel always precedes MatmulKernel (registry insertion order)."""
        gen = _gen("mixed_matmul_relu.onnx")
        names = [kd.name for kd in gen._active_kernels]
        self.assertLess(names.index("VectorOPKernel"), names.index("MatmulKernel"))


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelSource(unittest.TestCase):
    """Generated inference.c is correct for mixed-kernel models."""

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_both_headers_included(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn('#include "xvectoropkernel.h"', s)
        self.assertIn('#include "xmatmulkernel.h"',   s)

    def test_both_static_instances(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("XVectoropkernel s_vectoropkernel", s)
        self.assertIn("XMatmulkernel s_matmulkernel",     s)

    def test_no_single_s_kernel(self):
        """Old generic 's_kernel' variable must not appear in multi-kernel output."""
        s = self._src("mixed_matmul_relu.onnx")
        self.assertNotIn("s_kernel;", s)

    def test_init_has_both_params(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("vectoropkernel_instance", s)
        self.assertIn("matmulkernel_instance",   s)

    def test_init_calls_both_initialize(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("XVectoropkernel_Initialize(&s_vectoropkernel", s)
        self.assertIn("XMatmulkernel_Initialize(&s_matmulkernel",     s)

    def test_run_op_uses_named_var(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("s_vectoropkernel", s)

    def test_run_matmul_uses_named_var(self):
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("s_matmulkernel", s)

    def test_vectorop_only_uses_named_var(self):
        """VectorOP-only model also uses the named variable."""
        if not _models_exist():
            self.skipTest("VOP-only models not generated")
        path = _model("single_add.onnx")
        s = CodeGenerator(OnnxGraph(path), model_path=path).generate_source()
        self.assertIn("s_vectoropkernel", s)
        self.assertNotIn("s_kernel;", s)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelHeader(unittest.TestCase):
    """Generated inference.h is correct for mixed-kernel models."""

    def _hdr(self, name: str) -> str:
        return _gen(name).generate_header()

    def test_both_uio_macros(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", h)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   h)

    def test_uio_defaults_match_dtsi(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn('"VectorOPKernel_0"', h)
        self.assertIn('"MatmulKernel_0"',   h)

    def test_init_signature_has_both_params(self):
        h = self._hdr("mixed_matmul_relu.onnx")
        self.assertIn("vectoropkernel_instance", h)
        self.assertIn("matmulkernel_instance",   h)

    def test_single_kernel_init_single_param(self):
        """VectorOP-only model has exactly one init param."""
        if not _models_exist():
            self.skipTest("VOP-only models not generated")
        path = _model("single_add.onnx")
        h = CodeGenerator(OnnxGraph(path), model_path=path).generate_header()
        self.assertIn("vectoropkernel_instance", h)
        self.assertNotIn("matmulkernel_instance", h)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelCMake(unittest.TestCase):
    """Generated CMakeLists.txt handles both kernel driver sets."""

    def _cmake(self, name: str) -> str:
        return _gen(name).generate_cmake()

    def test_both_driver_common_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("driver/xvectoropkernel.c", cm)
        self.assertIn("driver/xmatmulkernel.c",   cm)

    def test_both_sinit_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("xvectoropkernel_sinit.c", cm)
        self.assertIn("xmatmulkernel_sinit.c",   cm)

    def test_both_linux_files(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("xvectoropkernel_linux.c", cm)
        self.assertIn("xmatmulkernel_linux.c",   cm)

    def test_axi_addresses_in_comment(self):
        cm = self._cmake("mixed_matmul_relu.onnx")
        self.assertIn("0xA0000000", cm)   # VectorOPKernel AXI base
        self.assertIn("0xA0010000", cm)   # MatmulKernel AXI base


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestMixedKernelTestGen(unittest.TestCase):
    """Generated test_inference.c calls inference_init() with per-kernel args."""

    def _test(self, name: str) -> str:
        return _gen(name).generate_test()

    def test_both_instance_macros(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", t)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   t)

    def test_both_uio_defaults(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn('"VectorOPKernel_0"', t)
        self.assertIn('"MatmulKernel_0"',   t)

    def test_inference_init_call_has_both_args(self):
        t = self._test("mixed_matmul_relu.onnx")
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", t)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   t)
        # Both must appear inside the inference_init() call
        init_pos  = t.index("inference_init(")
        paren_end = t.index(");", init_pos)
        call_text = t[init_pos:paren_end]
        self.assertIn("INFERENCE_VECTOROPKERNEL_INSTANCE", call_text)
        self.assertIn("INFERENCE_MATMULKERNEL_INSTANCE",   call_text)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestKernelRegistry(unittest.TestCase):
    """Sanity checks on KERNEL_REGISTRY contents."""

    def test_vectorop_entry(self):
        kd = KERNEL_REGISTRY["VectorOPKernel"]
        self.assertEqual(kd.driver_prefix, "xvectoropkernel")
        self.assertEqual(kd.c_type, "XVectoropkernel")
        self.assertEqual(kd.uio_default, "VectorOPKernel_0")
        self.assertEqual(kd.axi_base, 0xA000_0000)
        self.assertIn("xvectoropkernel.h", kd.driver_files)

    def test_matmul_entry(self):
        kd = KERNEL_REGISTRY["MatmulKernel"]
        self.assertEqual(kd.driver_prefix, "xmatmulkernel")
        self.assertEqual(kd.c_type, "XMatmulkernel")
        self.assertEqual(kd.uio_default, "MatmulKernel_0")
        self.assertEqual(kd.axi_base, 0xA001_0000)
        self.assertIn("xmatmulkernel.h", kd.driver_files)

    def test_c_var_names(self):
        self.assertEqual(KERNEL_REGISTRY["VectorOPKernel"].c_var, "s_vectoropkernel")
        self.assertEqual(KERNEL_REGISTRY["MatmulKernel"].c_var,   "s_matmulkernel")

    def test_init_param_names(self):
        self.assertEqual(KERNEL_REGISTRY["VectorOPKernel"].init_param, "vectoropkernel_instance")
        self.assertEqual(KERNEL_REGISTRY["MatmulKernel"].init_param,   "matmulkernel_instance")

    def test_instance_macros(self):
        self.assertEqual(
            KERNEL_REGISTRY["VectorOPKernel"].instance_macro,
            "INFERENCE_VECTOROPKERNEL_INSTANCE"
        )
        self.assertEqual(
            KERNEL_REGISTRY["MatmulKernel"].instance_macro,
            "INFERENCE_MATMULKERNEL_INSTANCE"
        )


# ======================================================================
# Tests for the three runtime-failure fixes:
#   Fix A — _broadcast_io_map pass 2 excludes MatmulNode outputs
#   Fix B — _compute_alloc_sizes pads MatmulNode output when it feeds
#             a downstream broadcast VectorOP
#   Fix C — _get_matmul_strides() returns correct (a_row_stride, c_row_stride)
#   Fix D — MatmulNode.emit_call() generates strided run_matmul() correctly
#   Fix E — _source.py uses _get_matmul_strides when emitting MatmulNode calls
# ======================================================================

@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestBroadcastIoMapExcludesMatmul(unittest.TestCase):
    """
    Fix A: _broadcast_io_map() pass 2 must not propagate a bcast entry
    from a strided input into a MatmulNode's output.

    mixed_add_matmul:     Add(X,bias)->Z  MatMul(Z,W)->Y
      Z is an advancing input of the broadcast Add → Z is in the map.
      Y is MatMul's output — must NOT inherit Z's entry (Fix A).

    mixed_matmul_add_relu: MatMul(X,W)->Z  Add(Z,bias)->A  Relu(A)->Y
      Z is a MatmulNode output — must NOT appear in the map (Fix A).
      A is the broadcast Add output → A is in the map.
      Y is the Relu output, which propagates from A → Y is in the map.
    """

    def _bmap(self, name: str) -> dict:
        return _gen(name)._broadcast_io_map()

    def test_add_matmul_z_in_map(self):
        """Z is an advancing input of the broadcast Add — must be in the map."""
        bmap = self._bmap("mixed_add_matmul.onnx")
        self.assertIn("Z", bmap)

    def test_add_matmul_y_not_in_map(self):
        """Y is a MatmulNode output — must NOT be in the map (Fix A)."""
        bmap = self._bmap("mixed_add_matmul.onnx")
        self.assertNotIn("Y", bmap)

    def test_matmul_add_relu_z_in_map(self):
        """Z is an advancing input of the broadcast Add(Z, bias) → A — pass 1
        correctly adds it to the map so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("Z", bmap)

    def test_matmul_add_relu_a_in_map(self):
        """A is the broadcast Add output — must be in the map."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("A", bmap)

    def test_matmul_add_relu_y_in_map(self):
        """Y is Relu(A) — stride propagates from A through the non-broadcast
        Relu, so Y must be in the map."""
        bmap = self._bmap("mixed_matmul_add_relu.onnx")
        self.assertIn("Y", bmap)

    def test_two_layer_mlp_z1_in_map(self):
        """z1 is an advancing input of the first broadcast Add(z1,b1) — pass 1
        correctly adds it so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("z1", bmap)

    def test_two_layer_mlp_z2_in_map(self):
        """z2 is an advancing input of the second broadcast Add(z2,b2) — pass 1
        correctly adds it so VectorOPKernel reads it at strided offsets."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("z2", bmap)

    def test_two_layer_mlp_a1_in_map(self):
        """a1 is the first broadcast Add output — must be in the map."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("a1", bmap)

    def test_two_layer_mlp_h1_in_map(self):
        """h1 is Relu(a1) — stride propagates from a1, so h1 must be in the map."""
        bmap = self._bmap("mixed_two_layer_mlp.onnx")
        self.assertIn("h1", bmap)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestAllocSizesForMixedModels(unittest.TestCase):
    """
    Fix B: _compute_alloc_sizes() must pad a MatmulNode output when it is
    used as an advancing input of a downstream broadcast VectorOP.

    All models use n=4, aligned_chunk_size=8.
    Padded alloc = outer_count(4) × aligned_chunk(8) = 32.

    mixed_add_matmul (n=4, k=8, m=4):
      Z[4,8] numel=32 — advancing input of broadcast Add, aligned_chunk=8.
      Z.alloc must equal numel (32) — no extra padding needed since K==aligned_chunk.
      Y[4,4] numel=16 — MatMul output with no downstream broadcast.
      Y.alloc must equal numel (16) — not inflated by Fix A.

    mixed_matmul_add_relu (n=4, k=8, m=4):
      Z[4,4] numel=16 — MatMul output that feeds broadcast Add as advancing input.
      Z.alloc must be 32 (padded to outer×aligned_chunk).
      Y[4,4] numel=16 — Relu output, inherits Z→A→Y padding.
      Y.alloc must be 32.

    mixed_two_layer_mlp (n=4, k=8, h=6, m=4):
      z1[4,6] numel=24 — first MatMul output, feeds broadcast Add (chunk=6, aligned=8).
      z1.alloc must be 32.
      h1[4,6] numel=24 — Relu output, inherits padding from z1→a1→h1.
      h1.alloc must be 32.
      z2[4,4] numel=16 — second MatMul output, feeds broadcast Add (chunk=4, aligned=8).
      z2.alloc must be 32.
    """

    def _alloc(self, name: str) -> dict:
        return _gen(name)._alloc_sizes

    # ---- mixed_add_matmul ------------------------------------------ #

    def test_add_matmul_z_alloc_not_inflated(self):
        """Z.numel==32 and aligned_chunk==8==K — no extra padding needed."""
        alloc = self._alloc("mixed_add_matmul.onnx")
        self.assertEqual(alloc["Z"], 32)

    def test_add_matmul_y_alloc_equals_numel(self):
        """Y is the MatMul output with no downstream broadcast — alloc must
        equal numel (16), not the inflated 32 that the old bug produced."""
        alloc = self._alloc("mixed_add_matmul.onnx")
        self.assertEqual(alloc["Y"], 16)

    # ---- mixed_matmul_add_relu ------------------------------------- #

    def test_matmul_add_relu_z_alloc_padded(self):
        """Z[4,4] numel=16 feeds broadcast Add — alloc must be 4×8=32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["Z"], 32)

    def test_matmul_add_relu_a_alloc_padded(self):
        """A is the broadcast Add output — alloc must be 32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["A"], 32)

    def test_matmul_add_relu_y_alloc_padded(self):
        """Y=Relu(A) inherits A's padded alloc — must be 32."""
        alloc = self._alloc("mixed_matmul_add_relu.onnx")
        self.assertEqual(alloc["Y"], 32)

    # ---- mixed_two_layer_mlp --------------------------------------- #

    def test_two_layer_mlp_z1_alloc_padded(self):
        """z1[4,6] numel=24 feeds broadcast Add (aligned_chunk=8) — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["z1"], 32)

    def test_two_layer_mlp_h1_alloc_padded(self):
        """h1 is Relu(a1), inherits padding — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["h1"], 32)

    def test_two_layer_mlp_z2_alloc_padded(self):
        """z2[4,4] numel=16 feeds broadcast Add — alloc=32."""
        alloc = self._alloc("mixed_two_layer_mlp.onnx")
        self.assertEqual(alloc["z2"], 32)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestGetMatmulStrides(unittest.TestCase):
    """
    Fix C: _get_matmul_strides() must return correct (a_row_stride, c_row_stride)
    for each MatmulNode that has alignment-padded inputs or outputs.

    mixed_matmul_relu:     MatMul(X,W)->Z, Relu(Z)->Y — no broadcast VectorOP,
      no padding → _get_matmul_strides must be empty.

    mixed_add_matmul:      Add(X,bias)->Z, MatMul(Z,W)->Y — Z.numel==32==Z.alloc
      (K==aligned_chunk), Y.alloc==Y.numel → no strides needed.

    mixed_matmul_add_relu: MatMul(X,W)->Z, Add(Z,bias)->A, Relu(A)->Y
      Z.alloc=32 > Z.numel=16 → c_row_stride=32/4=8.
      X.alloc==X.numel → a_row_stride=0.

    mixed_two_layer_mlp:   MatMul(X,W1)->z1, …, MatMul(h1,W2)->z2, …
      z1.alloc=32 > z1.numel=24 → first MatMul c_row_stride=8.
      h1.alloc=32 > h1.numel=24 → second MatMul a_row_stride=8.
      z2.alloc=32 > z2.numel=16 → second MatMul c_row_stride=8.
    """

    def _strides(self, name: str) -> dict:
        return _gen(name)._get_matmul_strides()

    def test_matmul_relu_no_strides_needed(self):
        """No broadcast VectorOP in graph → no MatmulNode needs strides."""
        strides = self._strides("mixed_matmul_relu.onnx")
        self.assertEqual(strides, {})

    def test_add_matmul_no_strides_needed(self):
        """Z.numel==Z.alloc (K equals aligned_chunk) and Y has no downstream
        broadcast → no strides needed for the MatMul node."""
        strides = self._strides("mixed_add_matmul.onnx")
        self.assertEqual(strides, {})

    def test_matmul_add_relu_a_row_stride_zero(self):
        """X is the graph input with natural layout — a_row_stride must be 0."""
        strides = self._strides("mixed_matmul_add_relu.onnx")
        a_row, _ = strides["Z"]
        self.assertEqual(a_row, 0)

    def test_matmul_add_relu_c_row_stride(self):
        """Z.alloc=32, N=4 → c_row_stride=8 (rows padded to aligned_chunk)."""
        strides = self._strides("mixed_matmul_add_relu.onnx")
        _, c_row = strides["Z"]
        self.assertEqual(c_row, 8)

    def test_two_layer_mlp_z1_a_row_stride_zero(self):
        """First MatMul reads X which has natural layout — a_row_stride=0."""
        strides = self._strides("mixed_two_layer_mlp.onnx")
        a_row, _ = strides["z1"]
        self.assertEqual(a_row, 0)

    def test_two_layer_mlp_z1_c_row_stride(self):
        """z1.alloc=32, N=4 → c_row_stride=8."""
        strides = self._strides("mixed_two_layer_mlp.onnx")
        _, c_row = strides["z1"]
        self.assertEqual(c_row, 8)

    def test_two_layer_mlp_z2_a_row_stride(self):
        """h1.alloc=32, N=4 → a_row_stride=8 (reading strided h1)."""
        strides = self._strides("mixed_two_layer_mlp.onnx")
        a_row, _ = strides["z2"]
        self.assertEqual(a_row, 8)

    def test_two_layer_mlp_z2_c_row_stride(self):
        """z2.alloc=32, N=4 → c_row_stride=8."""
        strides = self._strides("mixed_two_layer_mlp.onnx")
        _, c_row = strides["z2"]
        self.assertEqual(c_row, 8)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestRunMatmulCallStrided(unittest.TestCase):
    """
    Fixes D+E: generated run_matmul() calls must use the row-strided
    decomposition (batch=N, n=1) whenever a_row_stride or c_row_stride
    is non-zero, and the standard layout (batch=1, n=N) otherwise.

    All models use n=4.  The strided form is:
        run_matmul(a, b, c,
                   1u, <k>u, <m>u, 4u,
                   <eff_a>u, 0u, <eff_c>u);

    The natural form is:
        run_matmul(a, b, c,
                   <n>u, <k>u, <m>u, 1u,
                   0u, 0u, 0u);
    """

    def _src(self, name: str) -> str:
        return _gen(name).generate_source()

    def test_matmul_relu_natural_call(self):
        """No striding needed — MatMul(X,W)->Z followed by Relu."""
        s = self._src("mixed_matmul_relu.onnx")
        self.assertIn("4u, 8u, 4u, 1u", s)   # n=4, k=8, m=4, batch=1
        self.assertIn("0u, 0u, 0u", s)         # all strides zero

    def test_add_matmul_natural_call(self):
        """Z.numel==Z.alloc (K=aligned_chunk=8) — no striding needed."""
        s = self._src("mixed_add_matmul.onnx")
        self.assertIn("4u, 8u, 4u, 1u", s)   # n=4, k=8, m=4, batch=1
        self.assertIn("0u, 0u, 0u", s)

    def test_matmul_add_relu_strided_call_dimensions(self):
        """c_row_stride=8 → decomposed as batch=4, n=1."""
        s = self._src("mixed_matmul_add_relu.onnx")
        self.assertIn("1u, 8u, 4u, 4u", s)   # n=1, k=8, m=4, batch=4

    def test_matmul_add_relu_strided_call_strides(self):
        """eff_a=K=8 (natural), eff_c=8 (aligned_chunk)."""
        s = self._src("mixed_matmul_add_relu.onnx")
        self.assertIn("8u, 0u, 8u", s)         # a_stride=8, b_stride=0, c_stride=8

    def test_two_layer_mlp_first_matmul_dimensions(self):
        """MatMul(X[4,8], W1[8,6]): c_row_stride=8 → n=1, k=8, m=6, batch=4."""
        s = self._src("mixed_two_layer_mlp.onnx")
        self.assertIn("1u, 8u, 6u, 4u", s)

    def test_two_layer_mlp_second_matmul_dimensions(self):
        """MatMul(h1[4,6], W2[6,4]): a_row_stride=8, c_row_stride=8 → n=1, k=6, m=4, batch=4."""
        s = self._src("mixed_two_layer_mlp.onnx")
        self.assertIn("1u, 6u, 4u, 4u", s)

    def test_two_layer_mlp_both_matmul_strides(self):
        """Both MatMul calls use a_stride=8, b_stride=0, c_stride=8."""
        s = self._src("mixed_two_layer_mlp.onnx")
        # count occurrences: there must be exactly two
        self.assertEqual(s.count("8u, 0u, 8u"), 2)


@unittest.skipUnless(_mixed_models_exist(),
                     "Run test/gen_mixed_kernel_models.py first")
class TestHeaderSizeMacrosForMixedModels(unittest.TestCase):
    """
    Fix A (header side): when a MatmulNode output is no longer added to
    _broadcast_io_map, the header must not emit a CHUNK macro for it and
    must size it at numel, not outer_count*aligned_chunk.

    Before Fix A, mixed_add_matmul emitted:
        INFERENCE_Y_SIZE   (4u * INFERENCE_Y_CHUNK_STRIDE)  /* 32 instead of 16 */
    and the test binary verified 32 elements — positions 16..31 were DMA-zero,
    causing "FAIL Y[chunk=2,j=0]: got 0.0000 expected ..." on hardware.
    """

    def _hdr(self, name: str) -> str:
        return _gen(name).generate_header()

    def test_add_matmul_y_size_is_numel(self):
        """INFERENCE_Y_SIZE must be the bare numel (16), not 32."""
        h = self._hdr("mixed_add_matmul.onnx")
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("16u", h)
        # Must not reference the (non-existent) CHUNK_STRIDE macro for Y
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_add_matmul_no_y_chunk_macro(self):
        """No CHUNK macro should be emitted for Y (it is a MatmulNode output)."""
        h = self._hdr("mixed_add_matmul.onnx")
        self.assertNotIn("INFERENCE_Y_CHUNK", h)

    def test_matmul_add_relu_z_no_chunk_macro(self):
        """Z is a MatmulNode output — no CHUNK macro must be emitted for it."""
        h = self._hdr("mixed_matmul_add_relu.onnx")
        self.assertNotIn("INFERENCE_Z_CHUNK", h)

    def test_matmul_add_relu_y_size_uses_stride_macro(self):
        """Y is the final graph output and inherits broadcast stride from A.
        INFERENCE_Y_SIZE must reference INFERENCE_A_CHUNK_STRIDE so the
        caller allocates enough space for all aligned rows.  There is no
        INFERENCE_Y_CHUNK macro — Y borrows the broadcast Add output's macro."""
        h = self._hdr("mixed_matmul_add_relu.onnx")
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("INFERENCE_A_CHUNK_STRIDE", h)
        # The SIZE expression must multiply outer_count by the stride macro
        self.assertIn("4u * INFERENCE_A_CHUNK_STRIDE", h)

    def test_two_layer_mlp_z1_no_chunk_macro(self):
        """z1 is the first MatMul output — no CHUNK macro emitted."""
        h = self._hdr("mixed_two_layer_mlp.onnx")
        self.assertNotIn("INFERENCE_Z1_CHUNK", h)

    def test_two_layer_mlp_z2_no_chunk_macro(self):
        """z2 is the second MatMul output — no CHUNK macro emitted."""
        h = self._hdr("mixed_two_layer_mlp.onnx")
        self.assertNotIn("INFERENCE_Z2_CHUNK", h)


if __name__ == "__main__":
    unittest.main(verbosity=2)

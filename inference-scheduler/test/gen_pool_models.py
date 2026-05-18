"""Generate ONNX test models that use PoolingKernel nodes."""

import os
import onnx
from onnx import helper, TensorProto

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _vi(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _opset(v: int = 13):
    return [helper.make_opsetid("", v)]


# ---------------------------------------------------------------------------
# pool_maxpool_simple: 2x2 MaxPool, stride=2, no padding
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_maxpool_simple() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_maxpool_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_maxpool_simple.onnx")


# ---------------------------------------------------------------------------
# pool_avgpool_simple: 2x2 AveragePool, stride=2, no padding
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_avgpool_simple() -> None:
    node = helper.make_node(
        "AveragePool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_avgpool_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_avgpool_simple.onnx")


# ---------------------------------------------------------------------------
# pool_maxpool_padded: 3x3 MaxPool, stride=1, pad=1 (same-size output)
# X[1,4,8,8] → Y[1,4,8,8]
# ---------------------------------------------------------------------------
def gen_maxpool_padded() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_maxpool_padded",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 8, 8])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_maxpool_padded.onnx")


# ---------------------------------------------------------------------------
# pool_avgpool_count_pad: 2x2 AveragePool with count_include_pad=1
# X[1,2,6,6] → Y[1,2,3,3]
# ---------------------------------------------------------------------------
def gen_avgpool_count_pad() -> None:
    node = helper.make_node(
        "AveragePool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], count_include_pad=1,
    )
    graph = helper.make_graph(
        [node], "pool_avgpool_count_pad",
        inputs=[_vi("X", [1, 2, 6, 6])],
        outputs=[_vi("Y", [1, 2, 3, 3])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_avgpool_count_pad.onnx")


# ---------------------------------------------------------------------------
# pool_global_max: GlobalMaxPool
# X[1,8,4,4] → Y[1,8,1,1]
# ---------------------------------------------------------------------------
def gen_global_max() -> None:
    node = helper.make_node(
        "GlobalMaxPool", inputs=["X"], outputs=["Y"],
    )
    graph = helper.make_graph(
        [node], "pool_global_max",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 8, 1, 1])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_global_max.onnx")


# ---------------------------------------------------------------------------
# pool_global_avg: GlobalAveragePool
# X[1,8,4,4] → Y[1,8,1,1]
# ---------------------------------------------------------------------------
def gen_global_avg() -> None:
    node = helper.make_node(
        "GlobalAveragePool", inputs=["X"], outputs=["Y"],
    )
    graph = helper.make_graph(
        [node], "pool_global_avg",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 8, 1, 1])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_global_avg.onnx")


# ---------------------------------------------------------------------------
# pool_lp_p2: LpPool p=2 (default), 2x2, stride=2
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_lp_p2() -> None:
    node = helper.make_node(
        "LpPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], p=2,
    )
    graph = helper.make_graph(
        [node], "pool_lp_p2",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset(18)), "pool_lp_p2.onnx")


# ---------------------------------------------------------------------------
# pool_lp_p1: LpPool p=1, 2x2, stride=2
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_lp_p1() -> None:
    node = helper.make_node(
        "LpPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], p=1,
    )
    graph = helper.make_graph(
        [node], "pool_lp_p1",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset(18)), "pool_lp_p1.onnx")


# ---------------------------------------------------------------------------
# pool_then_relu: MaxPool followed by Relu (PoolingKernel + VectorOPKernel)
# X[1,4,8,8] → Z[1,4,4,4] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_pool_then_relu() -> None:
    pool = helper.make_node("MaxPool",  inputs=["X"], outputs=["Z"], kernel_shape=[2, 2], strides=[2, 2])
    relu = helper.make_node("Relu",     inputs=["Z"], outputs=["Y"])
    graph = helper.make_graph(
        [pool, relu], "pool_then_relu",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_then_relu.onnx")


# ---------------------------------------------------------------------------
# pool_batch2: MaxPool with batch=2
# X[2,4,8,8] → Y[2,4,4,4]
# ---------------------------------------------------------------------------
def gen_pool_batch2() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_batch2",
        inputs=[_vi("X", [2, 4, 8, 8])],
        outputs=[_vi("Y", [2, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_batch2.onnx")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    gen_maxpool_simple,
    gen_avgpool_simple,
    gen_maxpool_padded,
    gen_avgpool_count_pad,
    gen_global_max,
    gen_global_avg,
    gen_lp_p2,
    gen_lp_p1,
    gen_pool_then_relu,
    gen_pool_batch2,
]


# ---------------------------------------------------------------------------
# Hardware-bound violation models — must each raise SchedulerError when
# loaded by OnnxGraph.  The bounds come from kernels/pool/CMakeLists.txt
# (defaults: kMaxPoolH/W=7, kMaxLineBufRows=16, kMaxLineBufCols=64).
#
# Each model violates exactly one bound by exactly one unit so the test
# can identify which constraint fired.  Geometries are otherwise minimal
# to keep ONNX shape-inference fast and the model files tiny.
# ---------------------------------------------------------------------------

def gen_unsupported_pool_h_too_large() -> None:
    """pool_h = 8 violates kMaxPoolH=7 (window taller than the adder tree)."""
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[8, 3], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_pool_h",
        inputs=[_vi("X", [1, 4, 12, 8])],
        outputs=[_vi("Y", [1, 4, 5, 6])],   # out_h = 12-8+1 = 5
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_pool_h.onnx")


def gen_unsupported_pool_w_too_large() -> None:
    """pool_w = 8 violates kMaxPoolW=7."""
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, 8], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_pool_w",
        inputs=[_vi("X", [1, 4, 8, 12])],
        outputs=[_vi("Y", [1, 4, 6, 5])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_pool_w.onnx")


def gen_unsupported_dil_h_overflows_line_buf() -> None:
    """pool_h=4 with dil_h=6 → vertical span = 3*6 + 1 = 19 > kMaxLineBufRows=16.

    Uses pool_h within the kMaxPoolH=7 limit so the violation isolates the
    line-buffer-row constraint, not the pool-height constraint.
    """
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[4, 3], strides=[1, 1], dilations=[6, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_dil_h",
        inputs=[_vi("X", [1, 4, 24, 8])],
        outputs=[_vi("Y", [1, 4, 6, 6])],   # out_h = 24 - 19 + 1 = 6
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_dil_h.onnx")


def gen_unsupported_dil_w_overflows_line_buf() -> None:
    """pool_w=4 with dil_w=22 → horizontal span = 3*22 + 1 = 67 > kMaxLineBufCols=64."""
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, 4], strides=[1, 1], dilations=[1, 22],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_dil_w",
        inputs=[_vi("X", [1, 4, 8, 80])],
        outputs=[_vi("Y", [1, 4, 6, 14])],   # out_w = 80 - 67 + 1 = 14
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_dil_w.onnx")


def gen_pool_h_at_limit() -> None:
    """Boundary-case: pool_h=7 exactly equals kMaxPoolH; must parse OK."""
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[7, 3], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_pool_h_at_limit",
        inputs=[_vi("X", [1, 4, 12, 8])],
        outputs=[_vi("Y", [1, 4, 6, 6])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_pool_h_at_limit.onnx")


def gen_dil_h_at_line_buf_limit() -> None:
    """Boundary-case: pool_h=4 dil_h=5 → span=16 = kMaxLineBufRows; must parse OK."""
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[4, 3], strides=[1, 1], dilations=[5, 1],
    )
    graph = helper.make_graph(
        [node], "pool_dil_h_at_limit",
        inputs=[_vi("X", [1, 4, 20, 8])],
        outputs=[_vi("Y", [1, 4, 5, 6])],   # out_h = 20 - 16 + 1 = 5
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_dil_h_at_limit.onnx")


_UNSUPPORTED_POOL_GENERATORS = [
    gen_unsupported_pool_h_too_large,
    gen_unsupported_pool_w_too_large,
    gen_unsupported_dil_h_overflows_line_buf,
    gen_unsupported_dil_w_overflows_line_buf,
    gen_pool_h_at_limit,
    gen_dil_h_at_line_buf_limit,
]
ALL_GENERATORS += _UNSUPPORTED_POOL_GENERATORS


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating pool test models in {OUT_DIR}/")
    for gen in ALL_GENERATORS:
        gen()
    print("Done.")


if __name__ == "__main__":
    main()

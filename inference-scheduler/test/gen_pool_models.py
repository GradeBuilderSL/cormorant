"""Generate ONNX test models that use PoolingKernel nodes."""

import os
import numpy as np
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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating pool test models in {OUT_DIR}/")
    for gen in ALL_GENERATORS:
        gen()
    print("Done.")


if __name__ == "__main__":
    main()

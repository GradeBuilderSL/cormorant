"""Generate ONNX test models that combine MatmulKernel and VectorOPKernel nodes."""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def gen_matmul_then_relu(n=4, k=8, m=4) -> None:
    """MatMul(X, W) → Relu(Z) — uses both kernels in sequence."""
    w_data = (np.random.randn(k, m) * 0.5).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    relu = helper.make_node("Relu", inputs=["Z"], outputs=["Y"])

    graph = helper.make_graph(
        [mm, relu],
        "matmul_relu",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_matmul_relu.onnx")


def gen_add_then_matmul(n=4, k=8, m=4) -> None:
    """Add(X, bias) → MatMul(Z, W) — VectorOP feeds into MatMul."""
    bias_data = np.zeros(k, dtype=np.float32)
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_init = numpy_helper.from_array(bias_data, name="bias")
    w_init    = numpy_helper.from_array(w_data,    name="W")

    add = helper.make_node("Add",    inputs=["X", "bias"], outputs=["Z"])
    mm  = helper.make_node("MatMul", inputs=["Z", "W"],    outputs=["Y"])

    graph = helper.make_graph(
        [add, mm],
        "add_matmul",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[bias_init, w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_add_matmul.onnx")


def gen_matmul_add_relu(n=4, k=8, m=4) -> None:
    """MatMul(X, W) → Add(Z, bias) → Relu(A) — classic linear layer."""
    w_data    = (np.random.randn(k, m) * 0.5).astype(np.float32)
    bias_data = np.zeros(m, dtype=np.float32)
    w_init    = numpy_helper.from_array(w_data,    name="W")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    mm   = helper.make_node("MatMul", inputs=["X", "W"],      outputs=["Z"])
    add  = helper.make_node("Add",    inputs=["Z", "bias"],    outputs=["A"])
    relu = helper.make_node("Relu",   inputs=["A"],            outputs=["Y"])

    graph = helper.make_graph(
        [mm, add, relu],
        "matmul_add_relu",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=[w_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_matmul_add_relu.onnx")


def gen_two_layer_mlp(n=4, k=8, h=6, m=4) -> None:
    """Two linear layers: MatMul+Add+Relu → MatMul+Add — a minimal MLP."""
    w1_data  = (np.random.randn(k, h) * 0.5).astype(np.float32)
    b1_data  = np.zeros(h, dtype=np.float32)
    w2_data  = (np.random.randn(h, m) * 0.5).astype(np.float32)
    b2_data  = np.zeros(m, dtype=np.float32)

    inits = [
        numpy_helper.from_array(w1_data, name="W1"),
        numpy_helper.from_array(b1_data, name="b1"),
        numpy_helper.from_array(w2_data, name="W2"),
        numpy_helper.from_array(b2_data, name="b2"),
    ]

    nodes = [
        helper.make_node("MatMul", ["X",  "W1"], ["z1"]),
        helper.make_node("Add",    ["z1", "b1"], ["a1"]),
        helper.make_node("Relu",   ["a1"],        ["h1"]),
        helper.make_node("MatMul", ["h1", "W2"], ["z2"]),
        helper.make_node("Add",    ["z2", "b2"], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes, "two_layer_mlp",
        inputs=[helper.make_tensor_value_info("X",  TensorProto.FLOAT, [n, k])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "mixed_two_layer_mlp.onnx")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating mixed-kernel test models...")
    gen_matmul_then_relu()
    gen_add_then_matmul()
    gen_matmul_add_relu()
    gen_two_layer_mlp()
    print("Done.")

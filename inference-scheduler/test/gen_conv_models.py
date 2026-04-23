"""Generate ONNX test models that use ConvKernel nodes."""

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


def _vi(name: str, shape) -> onnx.TensorProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


# ---------------------------------------------------------------------------
# conv_simple: 1x1 conv, no bias, batch=1
# X[1,4,8,8] * W[8,4,1,1] → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_simple() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_simple.onnx")


# ---------------------------------------------------------------------------
# conv_with_bias: 3x3 conv with bias, stride=1, no padding
# X[1,4,8,8] * W[6,4,3,3] + B[6] → Y[1,6,6,6]
# ---------------------------------------------------------------------------
def gen_conv_with_bias() -> None:
    w_data = (np.random.randn(6, 4, 3, 3) * 0.25).astype(np.float32)
    b_data = np.zeros(6, dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_with_bias",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 6, 6, 6])],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_with_bias.onnx")


# ---------------------------------------------------------------------------
# conv_stride2: 3x3 conv, stride=2, no padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,3,3]
# ---------------------------------------------------------------------------
def gen_conv_stride2() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_stride2",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 3, 3])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_stride2.onnx")


# ---------------------------------------------------------------------------
# conv_padded: 3x3 conv, stride=1, SAME_UPPER padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_padded() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_padded",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_padded.onnx")


# ---------------------------------------------------------------------------
# conv_batch2: 1x1 conv, batch=2
# X[2,4,4,4] * W[8,4,1,1] → Y[2,8,4,4]
# ---------------------------------------------------------------------------
def gen_conv_batch2() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[1, 1],
    )
    graph = helper.make_graph(
        [conv], "conv_batch2",
        inputs=[_vi("X", [2, 4, 4, 4])],
        outputs=[_vi("Y", [2, 8, 4, 4])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_batch2.onnx")


# ---------------------------------------------------------------------------
# conv_then_relu: Conv → Relu (ConvKernel then VectorOPKernel)
# X[1,4,8,8] * W[8,4,1,1] → Z[1,8,8,8] → Relu → Y[1,8,8,8]
# ---------------------------------------------------------------------------
def gen_conv_then_relu() -> None:
    w_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node("Conv",  inputs=["X", "W"], outputs=["Z"], kernel_shape=[1, 1])
    relu = helper.make_node("Relu",  inputs=["Z"],      outputs=["Y"])

    graph = helper.make_graph(
        [conv, relu], "conv_then_relu",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_then_relu.onnx")


# ---------------------------------------------------------------------------
# conv_then_add_flat: Conv → Add with same-shape weight (non-broadcast)
# Z[1,8,8,8] + scale[1,8,8,8] → Y  (no broadcasting, flat layout)
# ---------------------------------------------------------------------------
def gen_conv_then_add_flat() -> None:
    w_data     = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    scale_data = np.ones((1, 8, 8, 8), dtype=np.float32)
    w_init     = numpy_helper.from_array(w_data,     name="W")
    scale_init = numpy_helper.from_array(scale_data, name="scale")

    conv = helper.make_node("Conv", inputs=["X", "W"],       outputs=["Z"], kernel_shape=[1, 1])
    add  = helper.make_node("Add",  inputs=["Z", "scale"],   outputs=["Y"])

    graph = helper.make_graph(
        [conv, add], "conv_then_add_flat",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 8, 8])],
        initializer=[w_init, scale_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_then_add_flat.onnx")


# ---------------------------------------------------------------------------
# conv_relu_chain: Conv (no bias) → Relu → Conv (with bias) → Relu
# Tests two-stage pipeline with both kernel types
# X[1,4,8,8] → [1,8,8,8] → [1,4,8,8]
# ---------------------------------------------------------------------------
def gen_conv_relu_chain() -> None:
    w1_data = (np.random.randn(8, 4, 1, 1) * 0.25).astype(np.float32)
    w2_data = (np.random.randn(4, 8, 1, 1) * 0.25).astype(np.float32)
    b2_data = np.zeros(4, dtype=np.float32)
    w1_init = numpy_helper.from_array(w1_data, name="W1")
    w2_init = numpy_helper.from_array(w2_data, name="W2")
    b2_init = numpy_helper.from_array(b2_data, name="B2")

    conv1 = helper.make_node("Conv", inputs=["X", "W1"],      outputs=["Z1"], kernel_shape=[1, 1])
    relu1 = helper.make_node("Relu", inputs=["Z1"],            outputs=["Z2"])
    conv2 = helper.make_node("Conv", inputs=["Z2", "W2", "B2"], outputs=["Z3"], kernel_shape=[1, 1])
    relu2 = helper.make_node("Relu", inputs=["Z3"],            outputs=["Y"])

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2], "conv_relu_chain",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 8, 8])],
        initializer=[w1_init, w2_init, b2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_relu_chain.onnx")


# ---------------------------------------------------------------------------
# conv_depthwise: depthwise convolution (groups == in_channels)
# Not supported by ConvKernel (groups > 1) — must raise SchedulerError
# This model is intentionally invalid for the scheduler and is used only in
# test_conv.py::TestConvNodeValidation to verify the error is raised.
# X[1,4,8,8] * W[4,1,3,3] → Y[1,4,6,6]  groups=4
# ---------------------------------------------------------------------------
def gen_conv_depthwise() -> None:
    w_data = (np.random.randn(4, 1, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        group=4,
    )
    graph = helper.make_graph(
        [conv], "conv_depthwise",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 6, 6])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_depthwise.onnx")


# ---------------------------------------------------------------------------
# conv_dilation: 3x3 conv with dilation=2, no padding
# X[1,4,8,8] * W[8,4,3,3] → Y[1,8,4,4]  (eff. kernel 5x5 → 4 output pixels)
# ---------------------------------------------------------------------------
def gen_conv_dilation() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        dilations=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv], "conv_dilation",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 4, 4])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_dilation.onnx")


# ---------------------------------------------------------------------------
# conv_auto_pad_valid: 3x3 conv with auto_pad=VALID
# Equivalent to pads=[0,0,0,0]
# X[1,4,8,8] → Y[1,8,6,6]
# ---------------------------------------------------------------------------
def gen_conv_auto_pad_valid() -> None:
    w_data = (np.random.randn(8, 4, 3, 3) * 0.25).astype(np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        auto_pad="VALID",
    )
    graph = helper.make_graph(
        [conv], "conv_auto_pad_valid",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 8, 6, 6])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_auto_pad_valid.onnx")


# ---------------------------------------------------------------------------
# conv_two_layer_vgg: VGG-style two-layer conv block, 224×224 input.
#
# Mirrors the first two convolutional layers of VGG-16:
#   Layer 1: X[1,3,224,224]   * W1[64,3,3,3]   → Z[1,64,222,222]  (no pad, stride=1)
#   Layer 2: Z[1,64,222,222]  * W2[64,64,3,3]  → Y[1,64,220,220]  (no pad, stride=1)
#
# The 64×64×3×3 second-layer weight shape is what motivates the first layer:
# the input has in_ch=3, so a 64→64 conv requires a preceding 3→64 lift.
#
# Weight sizes:
#   W1: 64×3×3×3   = 1 728 elements  (<  4096 threshold → embedded as C array)
#   W2: 64×64×3×3  = 36 864 elements (>  4096 threshold → external weights/W2.dat)
# ---------------------------------------------------------------------------
def gen_conv_two_layer_vgg() -> None:
    w1_data = (np.random.randn(64, 3,  3, 3) * 0.1).astype(np.float32)
    w2_data = (np.random.randn(64, 64, 3, 3) * 0.1).astype(np.float32)
    w1_init = numpy_helper.from_array(w1_data, name="W1")
    w2_init = numpy_helper.from_array(w2_data, name="W2")

    conv1 = helper.make_node(
        "Conv",
        inputs=["X", "W1"],
        outputs=["Z"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    conv2 = helper.make_node(
        "Conv",
        inputs=["Z", "W2"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [conv1, conv2], "conv_two_layer_vgg",
        inputs=[_vi("X", [1, 3, 224, 224])],
        outputs=[_vi("Y", [1, 64, 220, 220])],
        initializer=[w1_init, w2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    _save(model, "conv_two_layer_vgg.onnx")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating Conv test models...")
    gen_conv_simple()
    gen_conv_with_bias()
    gen_conv_stride2()
    gen_conv_padded()
    gen_conv_batch2()
    gen_conv_then_relu()
    gen_conv_then_add_flat()
    gen_conv_relu_chain()
    gen_conv_depthwise()
    gen_conv_dilation()
    gen_conv_auto_pad_valid()
    gen_conv_two_layer_vgg()
    print("Done.")

#!/usr/bin/env python3
"""
gen_test_models.py — create minimal ONNX models for testing inference_scheduler.

Models produced:
  single_add.onnx           one Add node with a constant bias (one runtime input)
  relu_chain.onnx           Add followed by Relu (one runtime input)
  mixed_ops.onnx            Add -> Mul -> Relu6 (Clip 0,6) chain (one runtime input)
  two_input_add.onnx        Add with two runtime inputs: A + B -> Y
  two_input_chain.onnx      Two runtime inputs: A + B -> Relu -> Y
  two_output_tap.onnx       One input, two outputs: X+bias -> add_Y[out1] -> Relu -> relu_Y[out2]
  two_output_chain.onnx     One input, two outputs: X+b1 -> add_Y[out1] -> *scale -> Relu6 -> clip_Y[out2]
  batch_relu.onnx           2D input [4, 64]: X+bias -> Relu -> Y  (batch of 4 rows)
  matrix_ops.onnx           2D input [8, 32]: A*B -> Relu6 -> Y  (two runtime inputs, matrix shape)
  large_tensor.onnx         4D input [64, 64, 16, 16] (1M elements): X+bias -> add_Y[out1] -> Relu -> Y[out2]
  residual_add.onnx         minimal skip connection: Relu(X) -> relu_X, then X + relu_X -> Y
  residual_chain.onnx       full residual block: X+bias -> Relu -> *scale -> result, then X+result -> Y
  sub_bias.onnx             single Sub with a constant offset: X - mean -> Y
  div_scale.onnx            single Div with a constant scale: X / std -> Y
  normalize.onnx            Sub then Div: (X - mean) / std -> Y  (mean/std normalisation)
  broadcast_aligned.onnx    X[8,16] + bias[16] -> Y[8,16]  (chunk=16 elems, naturally aligned)
  broadcast_unaligned.onnx  X[4,6]  + bias[6]  -> Y[4,6]   (chunk=6 elems, needs padding to 8)
  unsupported.onnx          Conv node — must trigger a SchedulerError

Run from the inference-scheduler directory:
  python test/gen_test_models.py [--out-dir /path/to/models]
"""

import argparse
import os
import sys
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _save(model: onnx.ModelProto, path: str) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  saved: {path}")


def _float32(name: str, shape: list[int]) -> onnx.TypeProto:
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _initializer(name: str, data: np.ndarray) -> onnx.TensorProto:
    return nph.from_array(data.astype(np.float32), name=name)


# ------------------------------------------------------------------ #
# Model 1: single Add with a constant bias                           #
# ------------------------------------------------------------------ #

def make_single_add(out_dir: str) -> None:
    """input + bias  (both [1, 256])"""
    N = 256
    bias = np.ones((1, N), dtype=np.float32) * 0.5

    X    = _float32("X",    [1, N])
    Y    = _float32("Y",    [1, N])
    bias_init = _initializer("bias", bias)

    add_node = oh.make_node("Add", inputs=["X", "bias"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node],
        name="single_add",
        inputs=[X],
        outputs=[Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "single_add.onnx"))


# ------------------------------------------------------------------ #
# Model 2: Add -> Relu                                               #
# ------------------------------------------------------------------ #

def make_relu_chain(out_dir: str) -> None:
    """input + bias  then  Relu"""
    N = 128
    bias = np.random.randn(1, N).astype(np.float32) * 0.5

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="relu_chain",
        inputs=[X],
        outputs=[relu_Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "relu_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 3: Add -> Mul -> Clip(0,6)                                   #
# ------------------------------------------------------------------ #

def make_mixed_ops(out_dir: str) -> None:
    """
    input + bias1  →  * scale  →  Clip(0, 6)

    Clip uses input-based bounds (opset >= 11 style).
    """
    N     = 64
    bias1 = np.random.randn(1, N).astype(np.float32)
    scale = np.ones((1, N), dtype=np.float32) * 2.0
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    X      = _float32("X",       [1, N])
    add_Y  = _float32("add_Y",   [1, N])
    mul_Y  = _float32("mul_Y",   [1, N])
    clip_Y = _float32("clip_Y",  [1, N])

    inits = [
        _initializer("bias1", bias1),
        _initializer("scale", scale),
        _initializer("clip_min", mn),
        _initializer("clip_max", mx),
    ]

    add_node  = oh.make_node("Add",  ["X",     "bias1"], ["add_Y"])
    mul_node  = oh.make_node("Mul",  ["add_Y", "scale"], ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["clip_Y"])

    graph = oh.make_graph(
        nodes=[add_node, mul_node, clip_node],
        name="mixed_ops",
        inputs=[X],
        outputs=[clip_Y],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "mixed_ops.onnx"))


# ------------------------------------------------------------------ #
# Model 4: two runtime inputs — Add(A, B) -> Y                       #
# ------------------------------------------------------------------ #

def make_two_input_add(out_dir: str) -> None:
    """A + B  (both [1, 256], both runtime inputs — no initializers)"""
    N = 256

    A = _float32("A", [1, N])
    B = _float32("B", [1, N])
    Y = _float32("Y", [1, N])

    add_node = oh.make_node("Add", inputs=["A", "B"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node],
        name="two_input_add",
        inputs=[A, B],
        outputs=[Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_input_add.onnx"))


# ------------------------------------------------------------------ #
# Model 5: two runtime inputs — Add(A, B) -> Relu -> Y               #
# ------------------------------------------------------------------ #

def make_two_input_chain(out_dir: str) -> None:
    """A + B  then  Relu  (both inputs [1, 128], no initializers)"""
    N = 128

    A      = _float32("A",      [1, N])
    B      = _float32("B",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])

    add_node  = oh.make_node("Add",  inputs=["A", "B"],    outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="two_input_chain",
        inputs=[A, B],
        outputs=[relu_Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_input_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 6: one input, two outputs — tap an intermediate result       #
#   X + bias -> add_Y [output 1] -> Relu -> relu_Y [output 2]        #
# ------------------------------------------------------------------ #

def make_two_output_tap(out_dir: str) -> None:
    """
    add_Y is the output of the Add node AND a graph output, so the
    caller receives both the pre-activation (add_Y) and the rectified
    (relu_Y) result in separate caller-supplied buffers.
    """
    N    = 128
    bias = np.ones((1, N), dtype=np.float32) * 0.5

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    relu_Y = _float32("relu_Y", [1, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["relu_Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="two_output_tap",
        inputs=[X],
        outputs=[add_Y, relu_Y],   # intermediate add_Y is also exposed
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_output_tap.onnx"))


# ------------------------------------------------------------------ #
# Model 7: one input, two outputs — longer chain with tapped middle  #
#   X + b1 -> add_Y [output 1] -> * scale -> Relu6 -> clip_Y [out2] #
# ------------------------------------------------------------------ #

def make_two_output_chain(out_dir: str) -> None:
    """
    add_Y (post-Add) is tapped as output 1; the full chain result
    clip_Y (post-Relu6) is output 2.  scale is a constant weight.
    """
    N     = 64
    b1    = np.random.randn(1, N).astype(np.float32) * 0.5
    scale = np.ones((1, N), dtype=np.float32) * 2.0
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    X      = _float32("X",      [1, N])
    add_Y  = _float32("add_Y",  [1, N])
    mul_Y  = _float32("mul_Y",  [1, N])
    clip_Y = _float32("clip_Y", [1, N])

    inits = [
        _initializer("b1",       b1),
        _initializer("scale",    scale),
        _initializer("clip_min", mn),
        _initializer("clip_max", mx),
    ]

    add_node  = oh.make_node("Add",  ["X",     "b1"],              ["add_Y"])
    mul_node  = oh.make_node("Mul",  ["add_Y", "scale"],            ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["clip_Y"])

    graph = oh.make_graph(
        nodes=[add_node, mul_node, clip_node],
        name="two_output_chain",
        inputs=[X],
        outputs=[add_Y, clip_Y],   # tap after Add; final after Relu6
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "two_output_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 8: 2D input [4, 64] — batch of rows, one runtime input       #
#   X[4,64] + bias[4,64] -> add_Y -> Relu -> Y[4,64]                 #
# ------------------------------------------------------------------ #

def make_batch_relu(out_dir: str) -> None:
    """
    X has shape [4, 64] — 4 independent rows processed as a flat
    vector of 256 elements by VectorOPKernel.
    """
    B, N = 4, 64
    bias = np.ones((B, N), dtype=np.float32) * 0.25

    X     = _float32("X",     [B, N])
    add_Y = _float32("add_Y", [B, N])
    Y     = _float32("Y",     [B, N])
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="batch_relu",
        inputs=[X],
        outputs=[Y],
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "batch_relu.onnx"))


# ------------------------------------------------------------------ #
# Model 9: 2D input [8, 32] — two runtime inputs, matrix shape       #
#   A[8,32] * B[8,32] -> mul_Y -> Relu6 -> Y[8,32]                   #
# ------------------------------------------------------------------ #

def make_matrix_ops(out_dir: str) -> None:
    """
    A and B both have shape [8, 32] — a matrix of 256 elements each.
    Both are runtime inputs (no initializers).
    """
    M, N = 8, 32

    A     = _float32("A",     [M, N])
    B     = _float32("B",     [M, N])
    mul_Y = _float32("mul_Y", [M, N])
    Y     = _float32("Y",     [M, N])
    mn    = np.array([0.0], dtype=np.float32)
    mx    = np.array([6.0], dtype=np.float32)

    mul_node  = oh.make_node("Mul",  ["A", "B"],              ["mul_Y"])
    clip_node = oh.make_node("Clip", ["mul_Y", "clip_min", "clip_max"], ["Y"])

    graph = oh.make_graph(
        nodes=[mul_node, clip_node],
        name="matrix_ops",
        inputs=[A, B],
        outputs=[Y],
        initializer=[
            _initializer("clip_min", mn),
            _initializer("clip_max", mx),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "matrix_ops.onnx"))


# ------------------------------------------------------------------ #
# Model 10: large 4-D tensor [64, 64, 16, 16] — 1 048 576 elements  #
#   X[64,64,16,16] + bias -> add_Y[out1], Relu(add_Y) -> Y[out2]    #
# ------------------------------------------------------------------ #

def make_large_tensor(out_dir: str) -> None:
    """
    Feature-map-shaped tensor typical of a mid-network conv layer:
      batch=64, channels=64, height=16, width=16  => 1 048 576 elements.

    The intermediate add_Y is exposed as a second output so the caller
    can inspect pre-activation values.  VectorOPKernel sees a flat
    vector of 1 048 576 elements for each op.
    """
    shape = [64, 64, 16, 16]
    numel = 64 * 64 * 16 * 16   # 1 048 576
    bias  = np.full(shape, -0.01, dtype=np.float32)

    X      = _float32("X",     shape)
    add_Y  = _float32("add_Y", shape)
    Y      = _float32("Y",     shape)
    bias_init = _initializer("bias", bias)

    add_node  = oh.make_node("Add",  inputs=["X", "bias"], outputs=["add_Y"])
    relu_node = oh.make_node("Relu", inputs=["add_Y"],     outputs=["Y"])

    graph = oh.make_graph(
        nodes=[add_node, relu_node],
        name="large_tensor",
        inputs=[X],
        outputs=[add_Y, Y],     # tap pre-activation as first output
        initializer=[bias_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "large_tensor.onnx"))


# ------------------------------------------------------------------ #
# Model 11: minimal residual connection                              #
#   Relu(X) -> relu_X                                                #
#   X + relu_X -> Y          (skip: X bypasses the Relu branch)     #
# ------------------------------------------------------------------ #

def make_residual_add(out_dir: str) -> None:
    """
    Simplest residual block: one unary transform on the skip path,
    then the input is added back.  X is used as input to two nodes:
    the Relu and the final Add.
    """
    N = 128

    X      = _float32("X",      [1, N])
    relu_X = _float32("relu_X", [1, N])
    Y      = _float32("Y",      [1, N])

    relu_node = oh.make_node("Relu", inputs=["X"],          outputs=["relu_X"])
    add_node  = oh.make_node("Add",  inputs=["relu_X", "X"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[relu_node, add_node],
        name="residual_add",
        inputs=[X],
        outputs=[Y],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "residual_add.onnx"))


# ------------------------------------------------------------------ #
# Model 12: full residual block                                      #
#   X + bias  -> add_X                                               #
#   Relu(add_X) -> relu_X                                            #
#   relu_X * scale -> mul_X                                          #
#   X + mul_X -> Y           (skip: X bypasses the whole block)     #
# ------------------------------------------------------------------ #

def make_residual_chain(out_dir: str) -> None:
    """
    Residual block with bias shift, activation, and scaling on the
    transform branch; the original X is added back at the end.
    X is used as input to two nodes: the first Add and the last Add.
    """
    N     = 64
    bias  = np.random.randn(1, N).astype(np.float32) * 0.1
    scale = np.ones((1, N), dtype=np.float32) * 0.5

    X     = _float32("X",     [1, N])
    add_X = _float32("add_X", [1, N])
    relu_X= _float32("relu_X",[1, N])
    mul_X = _float32("mul_X", [1, N])
    Y     = _float32("Y",     [1, N])

    inits = [
        _initializer("bias",  bias),
        _initializer("scale", scale),
    ]

    nodes = [
        oh.make_node("Add",  ["X",      "bias"],  ["add_X"]),
        oh.make_node("Relu", ["add_X"],            ["relu_X"]),
        oh.make_node("Mul",  ["relu_X", "scale"],  ["mul_X"]),
        oh.make_node("Add",  ["X",      "mul_X"],  ["Y"]),
    ]

    graph = oh.make_graph(
        nodes=nodes,
        name="residual_chain",
        inputs=[X],
        outputs=[Y],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "residual_chain.onnx"))


# ------------------------------------------------------------------ #
# Model 13: single Sub — X - mean -> Y                               #
# ------------------------------------------------------------------ #

def make_sub_bias(out_dir: str) -> None:
    """Element-wise subtraction of a constant mean vector."""
    N    = 256
    mean = np.random.randn(1, N).astype(np.float32) * 0.1

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Sub", ["X", "mean"], ["Y"])],
        name="sub_bias",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("mean", mean)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "sub_bias.onnx"))


# ------------------------------------------------------------------ #
# Model 14: single Div — X / std -> Y                                #
# ------------------------------------------------------------------ #

def make_div_scale(out_dir: str) -> None:
    """Element-wise division by a constant standard-deviation vector."""
    N   = 256
    std = np.abs(np.random.randn(1, N).astype(np.float32)) + 0.1  # keep > 0

    X = _float32("X", [1, N])
    Y = _float32("Y", [1, N])

    graph = oh.make_graph(
        nodes=[oh.make_node("Div", ["X", "std"], ["Y"])],
        name="div_scale",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("std", std)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "div_scale.onnx"))


# ------------------------------------------------------------------ #
# Model 15: Sub then Div — (X - mean) / std -> Y                    #
#   Classic mean/std normalisation as a two-op chain.                #
# ------------------------------------------------------------------ #

def make_normalize(out_dir: str) -> None:
    """Mean-centre then scale: two VectorOPKernel calls."""
    N    = 256
    mean = np.random.randn(1, N).astype(np.float32) * 0.1
    std  = np.abs(np.random.randn(1, N).astype(np.float32)) + 0.1

    X     = _float32("X",     [1, N])
    sub_Y = _float32("sub_Y", [1, N])
    Y     = _float32("Y",     [1, N])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Sub", ["X",     "mean"], ["sub_Y"]),
            oh.make_node("Div", ["sub_Y", "std"],  ["Y"]),
        ],
        name="normalize",
        inputs=[X],
        outputs=[Y],
        initializer=[
            _initializer("mean", mean),
            _initializer("std",  std),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "normalize.onnx"))


# ------------------------------------------------------------------ #
# Model 16: broadcasting — chunk naturally 16-byte aligned           #
#   X[8, 16] + bias[16] -> Y[8, 16]                                  #
#   chunk = 16 elements × 2 bytes = 32 bytes (multiple of 16 → no gap)
# ------------------------------------------------------------------ #

def make_broadcast_aligned(out_dir: str) -> None:
    """
    Bias [16] broadcasts over the leading dimension of X [8, 16].
    chunk_size = 16 elements; with 2-byte elements chunk_bytes = 32,
    which is a multiple of INFERENCE_ALIGN_BYTES (16) so aligned_chunk == chunk.
    outer_count = 8.
    """
    ROWS, COLS = 8, 16
    bias = np.ones(COLS, dtype=np.float32) * 0.5

    X = _float32("X", [ROWS, COLS])
    Y = _float32("Y", [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="broadcast_aligned",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_aligned.onnx"))


# ------------------------------------------------------------------ #
# Model 17: broadcasting — chunk NOT 16-byte aligned (needs padding) #
#   X[4, 6] + bias[6] -> Y[4, 6]                                     #
#   chunk = 6 elements × 2 bytes = 12 bytes → padded to 8 elems (16 B)
# ------------------------------------------------------------------ #

def make_broadcast_unaligned(out_dir: str) -> None:
    """
    Bias [6] broadcasts over the leading dimension of X [4, 6].
    chunk_size = 6 elements; with 2-byte elements chunk_bytes = 12,
    NOT a multiple of INFERENCE_ALIGN_BYTES (16), so aligned_chunk = 8
    (gap of 2 unused elements per block).
    outer_count = 4.
    """
    ROWS, COLS = 4, 6
    bias = np.ones(COLS, dtype=np.float32) * 0.25

    X = _float32("X", [ROWS, COLS])
    Y = _float32("Y", [ROWS, COLS])

    graph = oh.make_graph(
        nodes=[oh.make_node("Add", ["X", "bias"], ["Y"])],
        name="broadcast_unaligned",
        inputs=[X],
        outputs=[Y],
        initializer=[_initializer("bias", bias)],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "broadcast_unaligned.onnx"))


# ------------------------------------------------------------------ #
# Model 18: unsupported op (Conv)                                    #
# ------------------------------------------------------------------ #

def make_unsupported(out_dir: str) -> None:
    """A single Conv node — must cause SchedulerError."""
    # Small 1x1 conv so the model is valid ONNX
    W = np.ones((4, 1, 1, 1), dtype=np.float32)

    X = _float32("X", [1, 1, 4, 4])
    Y = _float32("Y", [1, 4, 4, 4])
    w_init = _initializer("W", W)

    conv_node = oh.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[1, 1],
    )
    graph = oh.make_graph(
        nodes=[conv_node],
        name="unsupported",
        inputs=[X],
        outputs=[Y],
        initializer=[w_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    _save(model, os.path.join(out_dir, "unsupported.onnx"))


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Generate test ONNX models")
    p.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory to write .onnx files (created if absent)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating test ONNX models …")
    make_single_add(args.out_dir)
    make_relu_chain(args.out_dir)
    make_mixed_ops(args.out_dir)
    make_two_input_add(args.out_dir)
    make_two_input_chain(args.out_dir)
    make_two_output_tap(args.out_dir)
    make_two_output_chain(args.out_dir)
    make_batch_relu(args.out_dir)
    make_matrix_ops(args.out_dir)
    make_large_tensor(args.out_dir)
    make_residual_add(args.out_dir)
    make_residual_chain(args.out_dir)
    make_sub_bias(args.out_dir)
    make_div_scale(args.out_dir)
    make_normalize(args.out_dir)
    make_broadcast_aligned(args.out_dir)
    make_broadcast_unaligned(args.out_dir)
    make_unsupported(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()

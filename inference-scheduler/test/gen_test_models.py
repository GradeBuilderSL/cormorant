#!/usr/bin/env python3
"""
gen_test_models.py — create minimal ONNX models for testing inference_scheduler.

Models produced:
  single_add.onnx      one Add node with a constant bias
  relu_chain.onnx      Add followed by Relu
  mixed_ops.onnx       Add -> Mul -> Relu6 (Clip 0,6) chain
  unsupported.onnx     Conv node — must trigger a SchedulerError

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
# Model 4: unsupported op (Conv)                                     #
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
    make_unsupported(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()

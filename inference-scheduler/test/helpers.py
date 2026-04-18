import os
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
def _model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)
def _models_exist() -> bool:
    required = ["single_add.onnx", "relu_chain.onnx", "mixed_ops.onnx", "unsupported.onnx"]
    return all(os.path.isfile(_model(m)) for m in required)

"""
preprocessing.py — board-side camera-frame preprocessing for the KV260
camera demo.  Runs ON the KV260 (imported by camera_loop.py).

Converts an OpenCV BGR camera frame into the exact NCHW ap_fixed<16,8>
int16 blob the generated inference project expects on its input buffer.

The encoding MUST stay identical to demo/image_classification's
download_assets.py::_encode_pixels so the camera demo's predictions match
the static-image demo bit for bit:

  1. resize to input_size x input_size,
  2. BGR -> RGB  (OpenCV frames are BGR; the model was trained on RGB —
     skipping this swap silently corrupts every prediction),
  3. normalize  (tf | unit | none),
  4. HWC -> CHW,
  5. scale by 256 and pack as little-endian int16 (the ap_fixed<16,8>
     bit pattern; 1.0 -> 0x0100).

Pure numpy + OpenCV — deliberately no Pillow dependency, since the board
image may not ship it.
"""

import cv2
import numpy as np


def preprocess_frame(bgr_frame: np.ndarray, input_size: int,
                     normalize: str) -> bytes:
    """Encode one BGR camera frame into the FPGA input blob.

    Returns ``3 * input_size * input_size * 2`` bytes (int16, NCHW).
    """
    resized = cv2.resize(bgr_frame, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]                       # BGR -> RGB

    a = rgb.astype(np.float32)
    if normalize == "tf":
        a = (a - 127.5) / 127.5                     # [-1, 1]  (TF MobileNet)
    elif normalize == "unit":
        a = a / 255.0                               # [0, 1]   (Keras-style)
    elif normalize == "none":
        a = a / 256.0                               # ~ raw byte / 256
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r} "
                         f"(expected 'tf', 'unit', 'none')")

    a = np.transpose(a, (2, 0, 1))                  # HWC -> CHW
    bits = np.rint(a * 256.0)                       # ap_fixed<16,8>
    bits = np.clip(bits, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(bits.reshape(-1)).tobytes()

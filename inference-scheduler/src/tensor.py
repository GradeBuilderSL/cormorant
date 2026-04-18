"""
Tensor metadata and code-emission helpers.

TensorInfo wraps an ONNX tensor (weight, input, intermediate, or output)
and knows how to emit:
  - a static weight array (for constants/initializers)
  - a buffer declaration (for mutable intermediate tensors)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ap_fixed<16,8> encoding: value v is stored as round(v * 2^8) in a
# 16-bit two's-complement word.  Representable range: [-128, 127.99609375].
_AP_FIXED_SCALE = 256.0          # 2^8
_AP_FIXED_MIN   = -128.0
_AP_FIXED_MAX   =  127.99609375  # 127 + 255/256


def _sanitize_c_name(name: str) -> str:
    """Turn any ONNX tensor name into a valid C identifier."""
    # Replace anything that isn't alphanumeric or underscore
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Identifiers must not start with a digit
    if s and s[0].isdigit():
        s = "t_" + s
    # Collapse runs of underscores for readability
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tensor"


def _float_to_apfixed_hex(arr: np.ndarray) -> List[str]:
    """
    Convert a numpy float array to a list of ap_fixed<16,8> hex literals.

    The encoding is identical to how Xilinx HLS stores ap_fixed<16,8>
    values in memory: each element is a 16-bit two's-complement integer
    equal to round(v * 256), stored in little-endian order.
    """
    clipped = np.clip(arr.astype(np.float64), _AP_FIXED_MIN, _AP_FIXED_MAX)
    scaled  = np.round(clipped * _AP_FIXED_SCALE).astype(np.int16)
    # Reinterpret as uint16 for the hex display
    bits    = scaled.view(np.uint16).flatten()
    return [f"0x{v:04X}" for v in bits]


# Weight tensors with more elements than this are written to external .dat
# files and loaded at runtime via fread(), rather than embedded as C arrays.
# 4096 elements = 8 KB in ap_fixed<16,8>; keeps generated C files small.
LARGE_WEIGHT_THRESHOLD = 4096


@dataclass
class TensorInfo:
    onnx_name: str
    shape:     List[int]       # [] = scalar
    dtype:     str             # 'float32', 'int8', …  (ONNX dtype string)
    data:      Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return max(n, 1)

    @property
    def c_name(self) -> str:
        return _sanitize_c_name(self.onnx_name)

    @property
    def is_weight(self) -> bool:
        return self.data is not None

    @property
    def is_large_weight(self) -> bool:
        """True when the weight should be stored in an external .dat file."""
        return self.is_weight and self.numel > LARGE_WEIGHT_THRESHOLD

    # ------------------------------------------------------------------ #
    # Code emission                                                        #
    # ------------------------------------------------------------------ #

    def emit_weight_decl(self) -> str:
        """
        Emit a ROM array (prefixed _rom_) plus a DMA buffer pointer initialised
        to NULL.  inference_init() allocates the DMA buffer and copies the ROM
        data into it so the kernel can access the weights via physical addresses.

        Example:
            /* ROM data for weight 'bias' ... */
            static const uint16_t _rom_bias[64] = { 0x0100, ... };
            /* DMA buffer pointer — allocated at inference_init() */
            static inference_buf_t *bias = NULL;
        """
        if not self.is_weight:
            raise ValueError(f"Tensor '{self.onnx_name}' has no data")

        hexvals = _float_to_apfixed_hex(self.data)
        n       = len(hexvals)

        # Format 8 values per row for readability
        rows = []
        for i in range(0, n, 8):
            chunk = hexvals[i:i+8]
            rows.append("    " + ", ".join(chunk))
        inner = ",\n".join(rows)

        return (
            f"/* ROM data for weight '{self.onnx_name}'"
            f"  shape={self.shape}  dtype={self.dtype}\n"
            f" * Copied into a DMA-capable buffer at inference_init(). */\n"
            f"static const uint16_t _rom_{self.c_name}[{n}] = {{\n"
            f"{inner}\n"
            f"}};\n"
            f"/* DMA buffer pointer for '{self.onnx_name}' */\n"
            f"static inference_buf_t *{self.c_name} = NULL;"
        )

    def emit_large_weight_ptr_decl(self) -> str:
        """
        For large weights: emit only the DMA buffer pointer (no ROM array).
        The data is loaded at runtime from a .dat file.

        Example:
            /* External weight 'bias'  shape=[64,64,16,16]  numel=1048576
             * Loaded at inference_init() from weights/bias.dat */
            static inference_buf_t *bias = NULL;
        """
        return (
            f"/* External weight '{self.onnx_name}'"
            f"  shape={self.shape}  numel={self.numel}\n"
            f" * Loaded at inference_init() from weights/{self.c_name}.dat */\n"
            f"static inference_buf_t *{self.c_name} = NULL;"
        )

    def to_dat_bytes(self) -> bytes:
        """
        Serialise the weight tensor to raw little-endian uint16 bytes, identical
        to the in-memory layout used by the C ROM arrays.  Written to
        weights/<c_name>.dat by the scheduler and loaded at runtime by fread().
        """
        if not self.is_weight:
            raise ValueError(f"Tensor '{self.onnx_name}' has no data")
        clipped = np.clip(self.data.astype(np.float64), _AP_FIXED_MIN, _AP_FIXED_MAX)
        scaled  = np.round(clipped * _AP_FIXED_SCALE).astype(np.int16)
        return scaled.view(np.uint16).flatten().astype('<u2').tobytes()

    def emit_buffer_decl(self) -> str:
        """
        Emit a DMA buffer pointer for an intermediate tensor.
        Allocated at inference_init(); NULL until then.

        Example:
            static inference_buf_t *add_Y = NULL;  /* 'add_Y' shape=[1,64] */
        """
        return (
            f"static inference_buf_t *{self.c_name} = NULL;"
            f"  /* '{self.onnx_name}'  shape={self.shape} */"
        )

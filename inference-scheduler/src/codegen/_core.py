"""
C code generator for the inference scheduler.

CodeGenerator produces three files for an inference project:

  generate_header()  →  include/inference.h
      Public API: Data_t typedef, array-size macros,
      inference_init() and inference_run() declarations.

  generate_source()  →  src/inference.c
      Implementation: weight arrays, intermediate buffers,
      static run_op() helper, inference_init(), inference_run().
      Includes "inference.h" for the shared type and declarations.

  generate_cmake()   →  CMakeLists.txt
      Builds a static library 'inference' from src/inference.c
      and the XVectoropkernel driver sources in driver/.

The generated code targets the Xilinx KV260 (bare-metal and Linux):
  - XVectoropkernel driver API  (driver/xvectoropkernel.h)
  - Xil cache maintenance API   (xil_cache.h, from BSP; no-op on Linux)
  - DMA-capable buffers via inference_buf_t (src/inference_buf.c)
  - ap_fixed<16,8> element type

Physical-address model
----------------------
VectorOPKernel's AXI master ports read/write DDR using physical addresses
stored in AXI-Lite registers (Set_a / Set_b / Set_c).  On Linux, virtual
pointers from malloc/stack are NOT valid DDR addresses.

inference_buf_t abstracts this:
  - Linux:      dma-proxy pool; physical address from /proc/self/pagemap
                (requires root; dma_alloc_coherent memory is contiguous).
  - Bare-metal: malloc (virtual == physical on Xilinx standalone).

inference_buf_phys() returns the physical address to program into the kernel
registers; inference_buf_ptr() returns the virtual pointer for CPU access.
"""

from __future__ import annotations
import os
from typing import List

from ..graph  import OnnxGraph
from ..tensor import TensorInfo, LARGE_WEIGHT_THRESHOLD
from ..nodes  import _BYTES_PER_ELEM
from ._banners import _file_banner


class _CoreMixin:
    """Core state and shared helpers for CodeGenerator."""

    def __init__(self, graph: OnnxGraph, model_path: str,
                 embed_large_weights: bool = False) -> None:
        self._graph               = graph
        self._model_path          = model_path
        self._embed_large_weights = embed_large_weights
        # Precompute padded allocation sizes for all tensors (accounts for
        # broadcast alignment gaps).  Computed once; used by header, init,
        # and pool-size calculations.
        self._alloc_sizes         = self._compute_alloc_sizes()

    def _compute_alloc_sizes(self) -> dict:
        """
        Return {onnx_name: alloc_size_in_elements} for every tensor that
        needs a DMA buffer, accounting for broadcast alignment padding.

        For non-broadcast tensors: alloc_size == t.numel.
        For advancing buffers in a broadcast node: outer_count * aligned_chunk.
        For repeating buffers in a broadcast node: aligned_chunk (one block).
        When a tensor appears in multiple nodes the maximum size is used.
        """
        sizes: dict = {}

        # Seed with natural sizes
        all_tensors = (
            self._graph.weight_tensors
            + self._graph.intermediate_tensors
            + self._graph.input_tensors
            + self._graph.output_tensors
        )
        for t in all_tensors:
            sizes[t.onnx_name] = t.numel

        # Override for nodes that use broadcasting
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            total = sn.outer_count * sn.aligned_chunk_size

            # Output always advances through the full padded range
            sizes[sn.output.onnx_name] = max(
                sizes.get(sn.output.onnx_name, 0), total
            )
            # Input a
            a_size = total if sn.a_advances else sn.aligned_chunk_size
            sizes[sn.inputs[0].onnx_name] = max(
                sizes.get(sn.inputs[0].onnx_name, 0), a_size
            )
            # Input b (binary ops only)
            if sn.arity == 2:
                b_size = total if sn.b_advances else sn.aligned_chunk_size
                sizes[sn.inputs[1].onnx_name] = max(
                    sizes.get(sn.inputs[1].onnx_name, 0), b_size
                )

        return sizes

    def _compute_pool_bytes(self) -> int:
        """
        Total DMA memory needed for all model buffers, with 64-byte alignment
        per buffer, rounded up to a 4 KiB page boundary.

        Covers: weight tensors + intermediate tensors + model inputs + outputs.
        Uses _alloc_sizes (which accounts for broadcast alignment padding) and
        _BYTES_PER_ELEM so the result is correct regardless of the data type.
        Used to size the u-dma-buf pool and advertised as
        INFERENCE_BUF_POOL_SIZE_BYTES in the generated header.
        """
        def _align64(n: int) -> int:
            return (n + 63) & ~63

        total = sum(
            _align64(size * _BYTES_PER_ELEM)
            for size in self._alloc_sizes.values()
        )
        page = 4096
        return (total + page - 1) & ~(page - 1)

    @property
    def large_weight_tensors(self) -> List[TensorInfo]:
        """Weight tensors stored as external .dat files.
        Empty when embed_large_weights=True (all weights inlined)."""
        if self._embed_large_weights:
            return []
        return [t for t in self._graph.weight_tensors if t.is_large_weight]

    def generate_weight_dat(self, tensor: TensorInfo) -> bytes:
        """
        Return the raw binary content for weights/<c_name>.dat.
        Little-endian uint16 values in the same ap_fixed<16,8> encoding
        used by the C ROM arrays, suitable for fread() at runtime.
        """
        return tensor.to_dat_bytes()

    def _broadcast_io_map(self) -> dict:
        """
        Returns {onnx_name → (outer_count, chunk_macro, stride_macro)} for
        every I/O tensor (input or output) that advances through a broadcast
        node.  Repeating tensors (e.g. a bias that stays at offset 0 each
        iteration) are not included — they don't need strided fill/print logic.
        """
        result = {}
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            c_up         = sn.output.c_name.upper()
            chunk_macro  = f"INFERENCE_{c_up}_CHUNK"
            stride_macro = f"INFERENCE_{c_up}_CHUNK_STRIDE"
            n = sn.outer_count
            result[sn.output.onnx_name] = (n, chunk_macro, stride_macro)
            if sn.a_advances:
                result[sn.inputs[0].onnx_name] = (n, chunk_macro, stride_macro)
            if sn.arity == 2 and sn.b_advances:
                result[sn.inputs[1].onnx_name] = (n, chunk_macro, stride_macro)
        return result

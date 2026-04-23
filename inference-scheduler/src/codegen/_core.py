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
from typing import List, Optional

from ..graph   import OnnxGraph
from ..nodes   import ScheduledNode, MatmulNode
from ..kernels import KernelDesc, KERNEL_REGISTRY
from ..tensor  import TensorInfo, LARGE_WEIGHT_THRESHOLD
from ..dtype   import DataType, AP_FIXED_16_8
from ._banners import _file_banner


class _CoreMixin:
    """Core state and shared helpers for CodeGenerator."""

    def __init__(self, graph: OnnxGraph, model_path: str,
                 embed_large_weights: bool = False,
                 embed_large_expected: bool = False,
                 dtype: Optional[DataType] = None) -> None:
        self._graph                = graph
        self._model_path           = model_path
        self._embed_large_weights  = embed_large_weights
        self._embed_large_expected = embed_large_expected
        # Data type descriptor — controls encoding, quantization, and C type
        # declarations.  Defaults to ap_fixed<16,8> (the standard KV260 type).
        # Pass dtype=FLOAT32 (or any DataType subclass) for other types.
        self._dtype               = dtype if dtype is not None else AP_FIXED_16_8
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

        # Override for VectorOP nodes that use chunk-based broadcasting.
        # MatmulNode is excluded even when outer_count > 1: its outer loop
        # uses physical-address offsets rather than alignment-padded chunks,
        # so buffer sizes are just numel (set above) with no padding needed.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, MatmulNode):
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

        # Propagate padded sizes through non-broadcast VectorOP nodes.
        # When a node's input was padded by an upstream broadcast op, its
        # output buffer must be at least as large so run_op() can process
        # the full padded range (including alignment gap elements) in one
        # call.  Additionally, any co-input (e.g. a weight) whose numel is
        # smaller than the padded size is raised to that size so run_op()'s
        # uniform size argument is valid for every buffer in the operation.
        # Nodes are in topological order, so one forward pass is sufficient.
        #
        # MatmulNode is explicitly excluded: its inputs (A[N,K], B[K,M]) and
        # output (Y[N,M]) have inherently different element counts, so the
        # "uniform size" invariant does not apply.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, MatmulNode):
                continue
            max_input = max(sizes.get(inp.onnx_name, inp.numel)
                            for inp in sn.inputs)
            out = sn.output.onnx_name
            if max_input > sizes[out]:
                sizes[out] = max_input
            # Raise every co-input to the same padded alloc so the kernel
            # never reads past the end of any buffer.
            padded = sizes[out]
            for inp in sn.inputs:
                if sizes.get(inp.onnx_name, inp.numel) < padded:
                    sizes[inp.onnx_name] = padded

        return sizes

    def _strided_weight_params(self) -> dict:
        """
        Return {onnx_name: (outer_count, aligned_chunk_size)} for weight
        tensors whose alloc_size was raised above numel by backward
        propagation in _compute_alloc_sizes().

        These weights are used as co-inputs to non-broadcast nodes alongside
        strided (padded) intermediate buffers.  Their ROM arrays must be
        emitted in strided layout — data elements interleaved with zero-
        padding in the gap slots — so that a single run_op() call with
        size=alloc_size is valid for every buffer in the operation.

        outer_count and aligned_chunk_size are derived from the strided
        co-input's entry in _broadcast_io_map().
        """
        io_map = self._broadcast_io_map()
        result: dict = {}

        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            # Find outer_count from any strided non-weight co-input.
            outer_count = None
            for inp in sn.inputs:
                if inp.onnx_name in io_map:
                    outer_count = io_map[inp.onnx_name][0]
                    break
            if outer_count is None:
                continue
            # Record stride params for each weight input that was padded.
            for inp in sn.inputs:
                if not inp.is_weight:
                    continue
                alloc = self._alloc_sizes.get(inp.onnx_name, inp.numel)
                if alloc <= inp.numel:
                    continue
                result[inp.onnx_name] = (outer_count, alloc // outer_count)

        return result

    def _compute_pool_bytes(self) -> int:
        """
        Total DMA memory needed for all model buffers, with 64-byte alignment
        per buffer, rounded up to a 4 KiB page boundary.

        Covers: weight tensors + intermediate tensors + model inputs + outputs.
        Uses _alloc_sizes (which accounts for broadcast alignment padding) and
        dtype.bytes_per_elem so the result is correct regardless of the data type.
        Used to size the u-dma-buf pool and advertised as
        INFERENCE_BUF_POOL_SIZE_BYTES in the generated header.
        """
        def _align64(n: int) -> int:
            return (n + 63) & ~63

        bpe   = self._dtype.bytes_per_elem
        total = sum(
            _align64(size * bpe)
            for size in self._alloc_sizes.values()
        )
        page = 4096
        return (total + page - 1) & ~(page - 1)

    @property
    def _active_kernels(self) -> List[KernelDesc]:
        """Ordered list of KernelDesc for each kernel type present in the graph.

        Order follows KERNEL_REGISTRY insertion order (VectorOPKernel before
        MatmulKernel before any future kernels), not the graph's node order.
        This makes the generated inference_init() parameter order deterministic
        regardless of which node appears first in the model.
        """
        present = {sn.kernel_name for sn in self._graph.nodes}
        return [kd for kd in KERNEL_REGISTRY.values() if kd.name in present]

    # ------------------------------------------------------------------
    # Backward-compat helpers (kept for existing code and tests)
    # ------------------------------------------------------------------

    @property
    def _has_matmul_nodes(self) -> bool:
        """True when the graph contains at least one MatmulNode."""
        return any(isinstance(sn, MatmulNode) for sn in self._graph.nodes)

    @property
    def _has_vectorop_nodes(self) -> bool:
        """True when the graph contains at least one VectorOP ScheduledNode."""
        return any(isinstance(sn, ScheduledNode) for sn in self._graph.nodes)

    @property
    def _driver_prefix(self) -> str:
        """Driver prefix for the first active kernel — kept for backward compat."""
        ak = self._active_kernels
        return ak[0].driver_prefix if ak else "xvectoropkernel"

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
        Encoded using the active dtype (little-endian), suitable for
        fread() at runtime.
        """
        return tensor.to_dat_bytes(self._dtype)

    def _broadcast_io_map(self) -> dict:
        """
        Returns {onnx_name → (outer_count, chunk_macro, stride_macro)} for
        every tensor whose DMA buffer has alignment-gap padding and therefore
        needs strided fill/print logic.

        Two categories:
          1. Direct participants of a broadcast node: the advancing inputs and
             the output of any node with outer_count > 1.
          2. Outputs of non-broadcast nodes whose inputs are strided (stride
             propagation).  E.g. a Relu applied to a broadcast intermediate
             buffer produces a strided output that must be iterated chunk-by-
             chunk even though Relu itself is not a broadcast op.

        Repeating tensors (bias at offset 0 every iteration) are not included.
        Nodes are processed in topological order so one forward pass suffices.
        """
        result = {}

        # Pass 1 — VectorOP broadcast nodes: collect advancing inputs and outputs.
        # MatmulNode is excluded: its outer_count > 1 means a run_matmul_at()
        # loop with physical offsets, not chunk-stride alignment macros.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, MatmulNode):
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

        # Pass 2 — non-broadcast VectorOP nodes: propagate stride to their outputs.
        # MatmulNode is excluded: it reads/writes its own row layout and does not
        # inherit a broadcast chunk-stride descriptor from its inputs.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, MatmulNode):
                continue
            if sn.output.onnx_name in result:
                continue  # already classified in pass 1
            for inp in sn.inputs:
                if inp.onnx_name in result:
                    result[sn.output.onnx_name] = result[inp.onnx_name]
                    break

        return result

    def _get_matmul_strides(self) -> dict:
        """
        {output_onnx_name: (a_row_stride, c_row_stride)} for MatmulNodes
        whose A input or Y output DMA buffer has alignment-gap padding.

        When A is a strided intermediate (produced by a broadcast VectorOP),
        its rows in the DMA buffer are separated by aligned_chunk gaps.
        MatmulKernel must skip those gaps by using a_row_stride = alloc//N
        instead of the natural K.

        When Y feeds a downstream broadcast VectorOP as an advancing input,
        its alloc is padded to N*aligned_M.  MatmulKernel must write each row
        at that padded stride using c_row_stride = alloc//N instead of M.

        Only applies to outer_count==1 MatmulNodes (the outer-loop case
        handles its own strides via a/c_outer_stride).
        """
        result: dict = {}
        for sn in self._graph.nodes:
            if not isinstance(sn, MatmulNode):
                continue
            if sn.outer_count > 1:
                continue
            a = sn.inputs[0]
            n = sn.n
            a_alloc = self._alloc_sizes.get(a.onnx_name, a.numel)
            a_row_stride = (a_alloc // n) if a_alloc > a.numel else 0
            y = sn.output
            y_alloc = self._alloc_sizes.get(y.onnx_name, y.numel)
            c_row_stride = (y_alloc // n) if y_alloc > y.numel else 0
            if a_row_stride != 0 or c_row_stride != 0:
                result[y.onnx_name] = (a_row_stride, c_row_stride)
        return result

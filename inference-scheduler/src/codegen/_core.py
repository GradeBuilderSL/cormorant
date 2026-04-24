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
from ..nodes   import ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode, SchedulerError
from ..kernels import KernelDesc, KERNEL_REGISTRY
from ..tensor  import TensorInfo, LARGE_WEIGHT_THRESHOLD
from ..dtype   import DataType, AP_FIXED_16_8
from ..layout  import TensorLayout
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
        # Unified layout map: {onnx_name: TensorLayout} for every tensor.
        # Accounts for broadcast alignment gaps and propagated padding.
        # _alloc_sizes is derived for backward-compatibility with callers that
        # only need the flat {name: alloc_elements} view.
        self._layouts             = self._compute_tensor_layouts()
        self._alloc_sizes         = {k: v.alloc for k, v in self._layouts.items()}

    def _compute_tensor_layouts(self) -> dict:
        """
        Return {onnx_name: TensorLayout} for every tensor in the graph.

        Three-phase computation:

          Phase 1 — Seed all tensors as flat (alloc == numel).

          Phase 2 — VectorOP broadcast nodes (outer_count > 1, not MatmulNode):
            advancing inputs and the output get advancing-strided layouts
            (n_chunks == outer_count, stride == aligned_chunk_size).
            Repeating inputs (bias at offset 0 each iteration) get a single
            padded block (n_chunks == 1, alloc == aligned_chunk_size).
            MatmulNode is excluded: its outer loop uses physical-address offsets
            rather than alignment-padded chunks.

          Phase 3 — Non-broadcast non-MatmulNode propagation (topological order):
            When an input has a larger alloc than the output, the output
            inherits the alloc and (if the dominant input is strided) also
            inherits n_chunks and stride.  Co-inputs whose alloc is smaller
            than the output are raised to match, acquiring the same layout
            structure so that a single run_op() call with size=alloc is valid
            for every buffer in the operation.
            MatmulNode is excluded: it reads A/Y with per-row strides derived
            from its inputs' layouts at emit time, not via alloc propagation.
        """
        all_tensors = (
            self._graph.weight_tensors
            + self._graph.intermediate_tensors
            + self._graph.input_tensors
            + self._graph.output_tensors
        )

        # Phase 1: seed all tensors as flat.
        layouts: dict = {t.onnx_name: TensorLayout.flat(t.numel)
                         for t in all_tensors}

        # Phase 2: broadcast VectorOP nodes.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode)):
                continue

            n      = sn.outer_count          # number of loop iterations
            chunk  = sn.chunk_size           # data elements per iteration
            stride = sn.aligned_chunk_size   # buffer elements per iteration (>= chunk)
            alloc  = n * stride

            def _should_update_advancing(cur: TensorLayout) -> bool:
                # Update if switching from flat to advancing (n_chunks==1 → >1),
                # or if a later broadcast node needs a larger allocation.
                # When stride == chunk (no gap), alloc == numel so the pure-alloc
                # comparison would wrongly skip the flat→advancing transition.
                return cur.n_chunks == 1 or alloc > cur.alloc

            # Output: advancing-strided.
            out_name = sn.output.onnx_name
            if _should_update_advancing(layouts[out_name]):
                layouts[out_name] = TensorLayout.advancing(
                    sn.output.numel, n, stride
                )

            # Input a: advancing or repeating.
            a_name = sn.inputs[0].onnx_name
            if sn.a_advances:
                if _should_update_advancing(layouts[a_name]):
                    layouts[a_name] = TensorLayout.advancing(
                        sn.inputs[0].numel, n, stride
                    )
            else:
                new_alloc = max(layouts[a_name].alloc, stride)
                if new_alloc > layouts[a_name].alloc:
                    layouts[a_name] = TensorLayout.repeating(
                        sn.inputs[0].numel, new_alloc
                    )

            # Input b: advancing or repeating (binary ops only).
            if sn.arity == 2:
                b_name = sn.inputs[1].onnx_name
                if sn.b_advances:
                    if _should_update_advancing(layouts[b_name]):
                        layouts[b_name] = TensorLayout.advancing(
                            sn.inputs[1].numel, n, stride
                        )
                else:
                    new_alloc = max(layouts[b_name].alloc, stride)
                    if new_alloc > layouts[b_name].alloc:
                        layouts[b_name] = TensorLayout.repeating(
                            sn.inputs[1].numel, new_alloc
                        )

        # Phase 3: propagate layout through non-broadcast non-MatmulNode chains.
        # Nodes are in topological order; one forward pass is sufficient.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode)):
                continue

            input_layouts = [layouts[inp.onnx_name] for inp in sn.inputs]
            max_input_alloc = max(l.alloc for l in input_layouts)

            out_name = sn.output.onnx_name
            out_lay  = layouts[out_name]

            # Find the dominant strided input (highest alloc with n_chunks > 1).
            dominant = None
            for l in input_layouts:
                if l.n_chunks > 1:
                    if dominant is None or l.alloc > dominant.alloc:
                        dominant = l

            # Update output layout if any input has a larger alloc.
            if max_input_alloc > out_lay.alloc:
                if dominant is not None:
                    layouts[out_name] = TensorLayout.advancing(
                        out_lay.numel, dominant.n_chunks, dominant.stride
                    )
                else:
                    layouts[out_name] = TensorLayout(
                        numel=out_lay.numel, alloc=max_input_alloc,
                        n_chunks=1, chunk=out_lay.numel, stride=max_input_alloc
                    )

            # Raise every co-input whose alloc is smaller than the output's.
            # The dominant layout for inheritance is the (possibly updated) output.
            out_lay  = layouts[out_name]
            out_dom  = out_lay if out_lay.n_chunks > 1 else None

            for inp in sn.inputs:
                cur = layouts[inp.onnx_name]
                if cur.alloc >= out_lay.alloc:
                    continue
                if out_dom is not None:
                    layouts[inp.onnx_name] = TensorLayout.advancing(
                        cur.numel, out_dom.n_chunks, out_dom.stride
                    )
                else:
                    layouts[inp.onnx_name] = TensorLayout(
                        numel=cur.numel, alloc=out_lay.alloc,
                        n_chunks=1, chunk=cur.numel, stride=out_lay.alloc
                    )

        # Post-computation validation: ConvKernel / PoolingKernel write flat NCHW
        # output.  If the output ended up with n_chunks > 1 a broadcast VectorOP
        # downstream tried to assign an advancing layout — the kernel only populates
        # the first numel elements, so the gaps would be uninitialised.
        for sn in self._graph.nodes:
            if isinstance(sn, ConvNode):
                lay = layouts.get(sn.output.onnx_name)
                if lay is not None and lay.n_chunks > 1:
                    raise SchedulerError(
                        f"Conv node [{sn.index}] output '{sn.output.onnx_name}' "
                        f"(shape={sn.output.shape}) feeds a broadcast VectorOP node, "
                        f"which requires an advancing-strided buffer layout "
                        f"(n_chunks={lay.n_chunks}). "
                        f"ConvKernel writes a flat NCHW output, so adding a "
                        f"per-channel bias via a separate broadcast Add after Conv "
                        f"is not supported. Use the Conv operator's built-in bias "
                        f"input (3rd Conv input) instead."
                    )
            elif isinstance(sn, PoolNode):
                lay = layouts.get(sn.output.onnx_name)
                if lay is not None and lay.n_chunks > 1:
                    raise SchedulerError(
                        f"Pool node [{sn.index}] output '{sn.output.onnx_name}' "
                        f"(shape={sn.output.shape}) feeds a broadcast VectorOP node, "
                        f"which requires an advancing-strided buffer layout "
                        f"(n_chunks={lay.n_chunks}). "
                        f"PoolingKernel writes a flat NCHW output; insert a "
                        f"non-broadcast node between the pool and the broadcast op."
                    )

        return layouts

    def _strided_weight_params(self) -> dict:
        """
        Return {onnx_name: (n_chunks, stride)} for weight tensors whose DMA
        buffer has alignment gaps (n_chunks > 1 and stride > chunk).

        These weights are used as co-inputs to non-broadcast nodes alongside
        advancing-strided intermediates.  Their ROM arrays are emitted in
        strided layout — data elements interleaved with zero padding in the
        gap slots — so that a single run_op() call with size=alloc is valid
        for every buffer in the operation.
        """
        result: dict = {}
        for t in self._graph.weight_tensors:
            lay = self._layouts.get(t.onnx_name)
            if lay is None or lay.n_chunks <= 1 or not lay.is_strided:
                continue
            result[t.onnx_name] = (lay.n_chunks, lay.stride)
        return result

    def _compute_pool_layout(self):
        """
        Compute per-buffer offsets for the single contiguous pool allocation.

        Returns (layout, total_elems) where:
          layout     = list of (onnx_name, offset_in_elems, alloc_in_elems)
                       for all weights + non-reshape-alias intermediates, in
                       declaration order (weights first, then intermediates)
          total_elems = total pool size in elements (sum of aligned alloc sizes)

        Each slot is padded to a 64-byte boundary to ensure DMA cache-line
        alignment between sub-buffers.
        """
        bpe       = self._dtype.bytes_per_elem
        align_to  = 64 // bpe   # elements per 64-byte boundary
        def align_up(n):
            return (n + align_to - 1) & ~(align_to - 1)

        reshape_aliases = self._reshape_aliases
        layout  = []
        offset  = 0

        for t in self._graph.weight_tensors:
            alloc = self._alloc_sizes[t.onnx_name]
            layout.append((t.onnx_name, offset, alloc))
            offset += align_up(alloc)

        for t in self._graph.intermediate_tensors:
            if t.onnx_name in reshape_aliases:
                continue
            alloc = self._alloc_sizes[t.onnx_name]
            layout.append((t.onnx_name, offset, alloc))
            offset += align_up(alloc)

        return layout, offset

    def _compute_pool_bytes(self) -> int:
        """
        Total DMA memory needed for all model buffers, with 64-byte alignment
        per buffer, rounded up to a 4 KiB page boundary.

        Covers: weight tensors + intermediate tensors + model inputs + outputs.
        Uses _layouts (which accounts for broadcast alignment padding) and
        dtype.bytes_per_elem so the result is correct regardless of the data type.
        Used to size the u-dma-buf pool and advertised as
        INFERENCE_BUF_POOL_SIZE_BYTES in the generated header.
        """
        def _align64(n: int) -> int:
            return (n + 63) & ~63

        bpe   = self._dtype.bytes_per_elem
        total = sum(
            _align64(lay.alloc * bpe)
            for lay in self._layouts.values()
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
    def _has_conv_nodes(self) -> bool:
        """True when the graph contains at least one ConvNode."""
        return any(isinstance(sn, ConvNode) for sn in self._graph.nodes)

    @property
    def _has_pool_nodes(self) -> bool:
        """True when the graph contains at least one PoolNode."""
        return any(isinstance(sn, PoolNode) for sn in self._graph.nodes)

    @property
    def _reshape_aliases(self) -> dict:
        """Return {output_onnx_name: source_c_name} for all ReshapeNode outputs.

        The reshape output shares the same DMA buffer as its source — no
        allocation or free is needed; inference_init() just assigns the pointer.
        """
        result: dict = {}
        for sn in self._graph.nodes:
            if isinstance(sn, ReshapeNode):
                result[sn.output.onnx_name] = sn.inputs[0].c_name
        return result

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

        The integer values (outer_count) come from self._layouts.  The C macro
        name strings are derived from the broadcast node whose output sets the
        stride (pass 1), propagated through non-broadcast non-MatmulNode chains
        (pass 2 — Fix A: MatmulNode excluded to prevent its output inheriting
        a stride descriptor it doesn't use).
        """
        canonical: dict = {}   # onnx_name → c_up prefix string for C macros

        # Pass 1 — advancing inputs + output of each broadcast VectorOP node.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode)):
                continue
            c_up = sn.output.c_name.upper()
            canonical[sn.output.onnx_name] = c_up
            if sn.a_advances:
                canonical[sn.inputs[0].onnx_name] = c_up
            if sn.arity == 2 and sn.b_advances:
                canonical[sn.inputs[1].onnx_name] = c_up

        # Pass 2 — propagate through non-broadcast non-special-node chains.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode)):
                continue
            if sn.output.onnx_name in canonical:
                continue
            for inp in sn.inputs:
                if inp.onnx_name in canonical:
                    canonical[sn.output.onnx_name] = canonical[inp.onnx_name]
                    break

        # Build result: combine canonical string prefixes with integer values
        # from _layouts.  Only include tensors that actually have n_chunks > 1
        # (advancing-strided buffers) — repeating/flat tensors are excluded.
        result: dict = {}
        for name, c_up in canonical.items():
            lay = self._layouts.get(name)
            if lay and lay.n_chunks > 1:
                result[name] = (
                    lay.n_chunks,
                    f"INFERENCE_{c_up}_CHUNK",
                    f"INFERENCE_{c_up}_CHUNK_STRIDE",
                )
        return result

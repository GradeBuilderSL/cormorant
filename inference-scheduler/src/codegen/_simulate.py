"""
Mixin that forward-simulates the ONNX graph with fixed-point arithmetic.

Every operation is computed in float64 and then truncated/clipped to the
representable grid of ``self._dtype`` after each node — identical to what
the hardware kernel does after each element-wise op.

Quantization model
------------------
Two distinct rounding behaviours are used:

* **dtype.quantize()** — round-to-nearest (used when seeding weight values).
  Weights are written to the .dat / C-array via float_to_storage(), which
  also rounds to nearest; quantize() must match so the simulation uses the
  same value the hardware reads from the DMA buffer.

* **dtype.truncate()** — floor toward −∞ (used after each arithmetic node).
  HLS ap_fixed defaults to AP_TRN (truncation toward −∞) when narrowing an
  arithmetic result back to the element type.  For ADD/SUB/RELU/RELU6 this
  is a no-op because the inputs are already on the representable grid and
  the result inherits the same precision.  For MUL/DIV the intermediate
  result has more fractional bits and truncation matters.

The ramp input used by ``_simulate()`` mirrors the pattern written by the
generated test harness, so the expected output can be embedded verbatim in
test_inference.c for on-device verification.

Extending to a new data type
-----------------------------
Pass a different ``dtype`` to ``CodeGenerator.__init__``.  No changes to this
file are needed: all type-specific logic is encapsulated in ``DataType``.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

from ..nodes  import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_RELU, OP_RELU6,
    MatmulNode, ConvNode, PoolNode, ReshapeNode,
    POOL_MAX, POOL_AVG, POOL_LP,
)
from ..tensor import TensorInfo

# Expected GT arrays larger than this threshold are written to external


def _conv2d_ref(
    x: np.ndarray,        # float64, shape [N, C, H, W]
    w: np.ndarray,        # float64, shape [M, C, kH, kW]
    bias,                 # float64 shape [M] or None
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dilation_h: int,
    dilation_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reference 2-D standard convolution (group=1) in float64."""
    N, C, H, W = x.shape
    M, _, kH, kW = w.shape
    y = np.zeros((N, M, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            for khi in range(kH):
                ih = oh * stride_h + khi * dilation_h - pad_top
                if ih < 0 or ih >= H:
                    continue
                for kwi in range(kW):
                    iw = ow * stride_w + kwi * dilation_w - pad_left
                    if iw < 0 or iw >= W:
                        continue
                    # x[:, :, ih, iw]  shape [N, C]
                    # w[:, :, khi, kwi] shape [M, C]
                    # einsum('nc,mc->nm') → [N, M]
                    y[:, :, oh, ow] += np.einsum(
                        "nc,mc->nm", x[:, :, ih, iw], w[:, :, khi, kwi]
                    )

    if bias is not None:
        y += bias.reshape(1, M, 1, 1)

    return y


def _depthwise_conv2d_ref(
    x: np.ndarray,        # float64, shape [N, C, H, W]
    w: np.ndarray,        # float64, shape [C, 1, kH, kW]  (depthwise layout)
    bias,                 # float64 shape [C] or None
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dilation_h: int,
    dilation_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reference depthwise 2-D convolution (group=in_ch) in float64."""
    N, C, H, W = x.shape
    kH, kW = w.shape[2], w.shape[3]
    y = np.zeros((N, C, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            for khi in range(kH):
                ih = oh * stride_h + khi * dilation_h - pad_top
                if ih < 0 or ih >= H:
                    continue
                for kwi in range(kW):
                    iw = ow * stride_w + kwi * dilation_w - pad_left
                    if iw < 0 or iw >= W:
                        continue
                    # x[:, :, ih, iw]  shape [N, C]
                    # w[:, 0, khi, kwi] shape [C]  — one filter per channel
                    y[:, :, oh, ow] += x[:, :, ih, iw] * w[:, 0, khi, kwi].reshape(1, C)

    if bias is not None:
        y += bias.reshape(1, C, 1, 1)

    return y
def _pool2d_ref(
    x: np.ndarray,       # float64, shape [N, C, H, W]
    pool_h: int,
    pool_w: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    dil_h: int,
    dil_w: int,
    out_h: int,
    out_w: int,
    pool_type: int,
    lp_order: int,
    count_include_pad: int,
) -> np.ndarray:
    """Reference 2-D pooling in float64 matching PoolingKernel's NCHW output."""
    N, C, H, W = x.shape
    y = np.zeros((N, C, out_h, out_w), dtype=np.float64)

    for oh in range(out_h):
        for ow in range(out_w):
            valid_count = 0
            for khi in range(pool_h):
                for kwi in range(pool_w):
                    ih = oh * stride_h + khi * dil_h - pad_top
                    iw = ow * stride_w + kwi * dil_w - pad_left
                    if 0 <= ih < H and 0 <= iw < W:
                        valid_count += 1
            denom = (pool_h * pool_w) if count_include_pad else valid_count
            denom = max(denom, 1)

            acc = (
                np.full((N, C), -np.inf, dtype=np.float64)
                if pool_type == POOL_MAX
                else np.zeros((N, C), dtype=np.float64)
            )

            for khi in range(pool_h):
                for kwi in range(pool_w):
                    ih = oh * stride_h + khi * dil_h - pad_top
                    iw = ow * stride_w + kwi * dil_w - pad_left
                    if ih < 0 or ih >= H or iw < 0 or iw >= W:
                        continue
                    v = x[:, :, ih, iw]
                    if pool_type == POOL_MAX:
                        acc = np.maximum(acc, v)
                    elif pool_type == POOL_AVG:
                        acc += v
                    else:
                        acc += np.abs(v) if lp_order == 1 else v * v

            if pool_type == POOL_MAX:
                y[:, :, oh, ow] = acc
            elif pool_type == POOL_AVG:
                y[:, :, oh, ow] = acc / denom
            else:
                y[:, :, oh, ow] = acc if lp_order == 1 else np.sqrt(np.maximum(acc, 0.0))

    return y


# expected/<c_name>.dat files and loaded at runtime by fread(), matching
# the same approach used for large weight tensors.
LARGE_EXPECTED_THRESHOLD = 4096


class _SimulateMixin:
    """
    Fixed-point graph simulation and expected-output helpers.

    Public API
    ----------
    simulate(inputs)   — user-supplied float64 arrays → output dict
    _simulate()        — ramp inputs (matching test harness) → all arrays
    _expected_storage  — convert simulated logical array to raw storage array
    _emit_expected_c   — produce the static C array declaration for embedding
    """

    # ------------------------------------------------------------------ #
    # Public: simulate with caller-supplied inputs                        #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Forward-pass the graph with the given float inputs.

        Parameters
        ----------
        inputs : {onnx_name: ndarray}
            Float64 arrays in each tensor's logical shape.  Values should be
            representable by the active data type; the simulation quantizes at
            each node boundary exactly as the hardware does, but assumes
            inputs are already on the representable grid.

        Returns
        -------
        {onnx_name: ndarray}  for each graph output (float64, logical shape).
        """
        arrays = self._forward_pass(inputs)
        return {t.onnx_name: arrays[t.onnx_name]
                for t in self._graph.output_tensors}

    # ------------------------------------------------------------------ #
    # Internal: ramp-input simulation (used by generate_test)             #
    # ------------------------------------------------------------------ #

    def _simulate(self) -> Dict[str, np.ndarray]:
        """
        Forward-pass with the ramp inputs that generate_test() writes into
        the DMA buffers:

            p[pos] = (Data_t)(pos & mask)     (C test harness fill)

        The DataType converts these bit patterns to float64 values that match
        what the hardware reads from the buffer.  For broadcast tensors, the
        logical element at (chunk c, offset j) occupies buffer position
        ``c * aligned_chunk + j``.

        Returns
        -------
        {onnx_name: ndarray}  for every tensor visited (inputs, weights,
        intermediates, outputs), float64 in logical shape.
        """
        dtype = self._dtype
        ramp_inputs: Dict[str, np.ndarray] = {}

        for t in self._graph.input_tensors:
            idx = np.arange(t.numel, dtype=np.int64)
            lay = self._layouts.get(t.onnx_name)
            if lay and lay.n_chunks > 1:
                positions = (idx // lay.chunk) * lay.stride + (idx % lay.chunk)
            else:
                positions = idx
            ramp_inputs[t.onnx_name] = dtype.ramp_to_float(positions).reshape(t.shape)

        return self._forward_pass(ramp_inputs)

    # ------------------------------------------------------------------ #
    # Core: topological forward pass                                      #
    # ------------------------------------------------------------------ #

    def _forward_pass(
        self,
        input_arrays: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run every ScheduledNode in topological order, quantizing outputs with
        ``self._dtype.quantize()`` at each step.  Numpy broadcasting handles
        the same trailing-contiguous rules as the hardware broadcast loop.

        Returns {onnx_name: float64 ndarray} for every tensor visited.
        """
        dtype   = self._dtype
        arrays: Dict[str, np.ndarray] = {}

        # Seed with quantized weights (mirrors the ROM encoding written to C)
        for t in self._graph.weight_tensors:
            if t.data is not None:
                arrays[t.onnx_name] = dtype.quantize(
                    t.data.reshape(t.shape).astype(np.float64)
                )

        # Seed with caller-supplied inputs (already quantized by convention)
        arrays.update(input_arrays)

        # Node-by-node forward pass
        for sn in self._graph.nodes:
            if isinstance(sn, MatmulNode):
                a = arrays[sn.inputs[0].onnx_name]
                b = arrays[sn.inputs[1].onnx_name]
                result = np.matmul(a, b)
                arrays[sn.output.onnx_name] = dtype.truncate(result).reshape(
                    sn.output.shape
                )
                continue

            if isinstance(sn, ConvNode):
                conv_fn = _depthwise_conv2d_ref if sn.is_depthwise else _conv2d_ref
                arrays[sn.output.onnx_name] = dtype.truncate(
                    conv_fn(
                        x=arrays[sn.inputs[0].onnx_name],
                        w=arrays[sn.inputs[1].onnx_name],
                        bias=(arrays[sn.inputs[2].onnx_name]
                              if sn.has_bias else None),
                        stride_h=sn.stride_h,
                        stride_w=sn.stride_w,
                        pad_top=sn.pad_top,
                        pad_left=sn.pad_left,
                        dilation_h=sn.dilation_h,
                        dilation_w=sn.dilation_w,
                        out_h=sn.out_h,
                        out_w=sn.out_w,
                    )
                ).reshape(sn.output.shape)
                continue

            if isinstance(sn, ReshapeNode):
                src = arrays[sn.inputs[0].onnx_name]
                arrays[sn.output.onnx_name] = src.reshape(sn.output.shape)
                continue

            if isinstance(sn, PoolNode):
                arrays[sn.output.onnx_name] = dtype.truncate(
                    _pool2d_ref(
                        x=arrays[sn.inputs[0].onnx_name],
                        pool_h=sn.pool_h,
                        pool_w=sn.pool_w,
                        stride_h=sn.stride_h,
                        stride_w=sn.stride_w,
                        pad_top=sn.pad_top,
                        pad_left=sn.pad_left,
                        dil_h=sn.dil_h,
                        dil_w=sn.dil_w,
                        out_h=sn.out_h,
                        out_w=sn.out_w,
                        pool_type=sn.pool_type,
                        lp_order=sn.lp_order,
                        count_include_pad=sn.count_include_pad,
                    )
                ).reshape(sn.output.shape)
                continue

            a = arrays[sn.inputs[0].onnx_name]

            if sn.op_code == OP_ADD:
                result = a + arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_SUB:
                result = a - arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_MUL:
                result = a * arrays[sn.inputs[1].onnx_name]
            elif sn.op_code == OP_DIV:
                result = a / arrays[sn.inputs[1].onnx_name]
                # DIV: HLS computes a_int / b_int (C integer division =
                # truncation toward zero), so use truncate_div instead of
                # the default truncate (floor toward −∞).
                arrays[sn.output.onnx_name] = dtype.truncate_div(result).reshape(
                    sn.output.shape
                )
                continue
            elif sn.op_code == OP_RELU:
                result = np.maximum(a, 0.0)
            elif sn.op_code == OP_RELU6:
                result = np.minimum(np.maximum(a, 0.0), 6.0)
            else:
                raise ValueError(
                    f"_forward_pass: unknown op_code {sn.op_code} "
                    f"in node '{sn.onnx_node.name or sn.onnx_node.op_type}'"
                )

            arrays[sn.output.onnx_name] = dtype.truncate(result).reshape(
                sn.output.shape
            )

        return arrays

    # ------------------------------------------------------------------ #
    # Helpers: convert simulated output → C array                         #
    # ------------------------------------------------------------------ #

    def _expected_storage(
        self,
        name: str,
        logical: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a logical float64 array to the strided DMA-buffer layout that
        the hardware writes, encoded as the active data type's storage dtype.

        For broadcast tensors data elements occupy strided positions; gap slots
        (alignment padding) are zero.  For flat tensors the array is contiguous.

        Parameters
        ----------
        name    : onnx_name of the tensor
        logical : float64 ndarray in logical (non-strided) shape

        Returns
        -------
        ndarray with dtype == self._dtype.np_storage, length _alloc_sizes[name]
        """
        dtype = self._dtype
        lay   = self._layouts.get(name)
        alloc = lay.alloc if lay is not None else len(logical.flatten())
        buf   = np.zeros(alloc, dtype=dtype.np_storage)
        flat  = logical.flatten()
        encoded = dtype.float_to_storage(flat.astype(np.float64))

        if lay and lay.n_chunks > 1:
            idx      = np.arange(len(flat), dtype=np.int64)
            pos      = (idx // lay.chunk) * lay.stride + (idx % lay.chunk)
            buf[pos] = encoded
        else:
            buf[:len(flat)] = encoded

        return buf

    def _emit_expected_c(
        self,
        c_name: str,
        storage_buf: np.ndarray,
    ) -> str:
        """
        Emit a ``static const <type> expected_<c_name>[N]`` array suitable
        for embedding in test_inference.c.

        Eight elements per row, same style as the weight ROM arrays.
        The array type and literal format are determined by the active DataType.
        """
        dtype = self._dtype
        n     = len(storage_buf)
        rows  = []
        for i in range(0, n, 8):
            chunk = storage_buf[i : i + 8]
            rows.append("    " + ", ".join(dtype.format_literal(v) for v in chunk))
        inner = ",\n".join(rows)
        return (
            f"static const {dtype.c_array_type} expected_{c_name}[{n}] = {{\n"
            f"{inner}\n"
            f"}};\n"
        )

    # ------------------------------------------------------------------ #
    # Large expected: external .dat files                                  #
    # ------------------------------------------------------------------ #

    @property
    def large_expected_tensors(self) -> List[TensorInfo]:
        """
        Output tensors whose GT storage buffer exceeds LARGE_EXPECTED_THRESHOLD
        elements.  These are loaded from expected/<c_name>.dat at runtime
        instead of being embedded as static C arrays.

        Returns an empty list when embed_large_expected=True (all arrays inlined).
        """
        if self._embed_large_expected:
            return []
        result = []
        for t in self._graph.output_tensors:
            if self._alloc_sizes[t.onnx_name] > LARGE_EXPECTED_THRESHOLD:
                result.append(t)
        return result

    def generate_expected_dat(self, tensor: TensorInfo) -> bytes:
        """
        Return raw little-endian bytes for expected/<c_name>.dat.

        Serialises the storage buffer produced by ``_expected_storage()``
        using the same byte order as the weight .dat files, so the C
        ``_load_expected()`` helper can fread() it directly into a Data_t array.
        """
        sim_arrays  = self._simulate()
        storage_buf = self._expected_storage(
            tensor.onnx_name, sim_arrays[tensor.onnx_name]
        )
        return storage_buf.flatten().astype(
            storage_buf.dtype.newbyteorder("<")
        ).tobytes()

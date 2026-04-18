"""Mixin that generates src/inference.c."""

from __future__ import annotations
from typing import List

from ..nodes  import OP_NAMES, _ALIGN_BYTES, _BYTES_PER_ELEM, _ALIGN_ELEMS
from ..tensor import LARGE_WEIGHT_THRESHOLD
from ._banners import _banner, _file_banner


class _SourceMixin:
    """Generates src/inference.c."""

    def generate_source(self) -> str:
        """Content for src/inference.c."""
        parts = [
            _file_banner("inference.c", self._graph, self._model_path),
            self._source_includes(),
            self._source_op_defines(),
            self._weight_arrays(),
            self._buffer_declarations(),
            self._kernel_instance(),
            self._run_op_helper(),
            self._init_function(),
            self._inference_function(),
        ]
        return "\n".join(parts) + "\n"

    def _source_includes(self) -> str:
        stdio = '#include <stdio.h>    /* fopen, fread, snprintf */\n' \
                if self.large_weight_tensors else ''
        return (
            _banner("Includes") +
            '#include "inference.h"\n'
            '#include "xvectoropkernel.h"\n'
            '#include <string.h>    /* memcpy */\n'
            f'{stdio}'
            '\n'
            '/*\n'
            ' * Cache coherency between the CPU and the AXI DMA master is handled by\n'
            ' * inference_buf_sync_to_device() and inference_buf_sync_from_device(),\n'
            ' * both implemented in inference_buf.c:\n'
            ' *\n'
            ' *   Linux:      xclSyncBO (XRT) — XCL_BO_SYNC_BO_TO/FROM_DEVICE\n'
            ' *   Bare-metal: Xil_DCacheFlushRange / Xil_DCacheInvalidateRange\n'
            ' */\n'
            '\n'
            '/* Forward declarations — defined in inference_buf.c */\n'
            'void inference_buf_sync_to_device(inference_buf_t *buf);\n'
            'void inference_buf_sync_from_device(inference_buf_t *buf);\n'
        )

    def _source_op_defines(self) -> str:
        lines = [_banner("VectorOPKernel operation codes (must match VectorOP.h)")]
        for code, name in sorted(OP_NAMES.items()):
            lines.append(f"#define {name:<20} {code}u")
        return "\n".join(lines)

    def _weight_arrays(self) -> str:
        weights = self._graph.weight_tensors
        if not weights:
            return ""
        external = {t.onnx_name for t in self.large_weight_tensors}
        strided  = self._strided_weight_params()
        parts = [_banner("Constant weight arrays (ONNX initializers)")]
        for t in weights:
            if t.onnx_name in external:
                parts.append(t.emit_large_weight_ptr_decl())
            elif t.onnx_name in strided:
                outer_count, aligned_chunk = strided[t.onnx_name]
                parts.append(t.emit_weight_decl_strided(outer_count, aligned_chunk))
            else:
                parts.append(t.emit_weight_decl())
            parts.append("")
        return "\n".join(parts)

    def _buffer_declarations(self) -> str:
        lines = [_banner("Mutable intermediate buffers")]

        for t in self._graph.input_tensors:
            lines.append(f"/* INPUT  '{t.onnx_name}'  shape={t.shape} — caller-supplied */")
        for t in self._graph.output_tensors:
            lines.append(f"/* OUTPUT '{t.onnx_name}'  shape={t.shape} — caller-supplied */")
        lines.append("")

        intermediates = self._graph.intermediate_tensors
        if intermediates:
            for t in intermediates:
                lines.append(t.emit_buffer_decl())
        else:
            lines.append("/* (no intermediate buffers) */")

        return "\n".join(lines)

    def _kernel_instance(self) -> str:
        return (
            _banner("Kernel driver instance") +
            "static XVectoropkernel s_kernel;\n"
        )

    def _run_op_helper(self) -> str:
        nodes          = self._graph.nodes
        need_run_op    = any(sn.outer_count == 1 for sn in nodes)
        need_run_op_at = any(sn.outer_count >  1 for sn in nodes)

        if need_run_op and need_run_op_at:
            title = "run_op() / run_op_at() — VectorOPKernel dispatch helpers"
        elif need_run_op_at:
            title = "run_op_at() — VectorOPKernel dispatch helper"
        else:
            title = "run_op() — VectorOPKernel dispatch helper"

        parts = [_banner(title)]

        if need_run_op:
            parts.append(
                "/*\n"
                " * run_op() — program AXI-Lite registers with physical addresses,\n"
                " *             maintain cache coherence, start the kernel, and poll.\n"
                " *\n"
                " * The kernel's AXI master reads/writes DDR using PHYSICAL addresses;\n"
                " * inference_buf_phys() provides the address to program into the registers.\n"
                " * inference_buf_ptr() provides the virtual address for CPU cache ops.\n"
                " *\n"
                " *   a     input buffer A  (always required)\n"
                " *   b     input buffer B  (NULL for unary ops: RELU, RELU6)\n"
                " *   c     output buffer C (must not alias a or b)\n"
                " *   size  number of elements\n"
                " *   op    VECTOROP_* constant\n"
                " */\n"
                "static void run_op(\n"
                "    inference_buf_t *a,\n"
                "    inference_buf_t *b,\n"
                "    inference_buf_t *c,\n"
                "    unsigned         size,\n"
                "    unsigned         op)\n"
                "{\n"
                "    XVectoropkernel_Set_a(&s_kernel, inference_buf_phys(a));\n"
                "    XVectoropkernel_Set_b(&s_kernel, b ? inference_buf_phys(b) : (u64)0);\n"
                "    XVectoropkernel_Set_c(&s_kernel, inference_buf_phys(c));\n"
                "    XVectoropkernel_Set_size(&s_kernel, size);\n"
                "    XVectoropkernel_Set_op(&s_kernel, op);\n"
                "\n"
                "    /* Sync inputs: flush CPU cache / XRT BO to device before AXI master reads */\n"
                "    inference_buf_sync_to_device(a);\n"
                "    if (b != NULL)\n"
                "        inference_buf_sync_to_device(b);\n"
                "\n"
                "    XVectoropkernel_Start(&s_kernel);\n"
                "    while (!XVectoropkernel_IsDone(&s_kernel)) {}\n"
                "\n"
                "    /* Sync output: invalidate CPU cache / XRT BO from device after AXI master writes */\n"
                "    inference_buf_sync_from_device(c);\n"
                "}\n"
            )

        if need_run_op_at:
            parts.append(
                "/*\n"
                " * run_op_at() — offset-based dispatch for broadcasting loops.\n"
                " *\n"
                " * Same as run_op() but adds element offsets to the physical addresses\n"
                " * so each loop iteration targets a different chunk of the buffers.\n"
                " * Cache sync is the CALLER'S responsibility (done once per loop, not\n"
                " * per iteration).\n"
                " *\n"
                " * Offsets are in elements; converted to bytes using\n"
                " * INFERENCE_BYTES_PER_ELEM so the arithmetic is correct regardless\n"
                " * of the element data type.\n"
                " *\n"
                " *   a / b / c   buffer pointers (b may be NULL for unary ops)\n"
                " *   a_off / b_off / c_off   element offset into each buffer\n"
                " *   size        elements to process in this invocation\n"
                " *   op          VECTOROP_* constant\n"
                " */\n"
                "static void run_op_at(\n"
                "    inference_buf_t *a, unsigned a_off,\n"
                "    inference_buf_t *b, unsigned b_off,\n"
                "    inference_buf_t *c, unsigned c_off,\n"
                "    unsigned size, unsigned op)\n"
                "{\n"
                "    XVectoropkernel_Set_a(&s_kernel,\n"
                "        inference_buf_phys(a)"
                " + (uint64_t)a_off * INFERENCE_BYTES_PER_ELEM);\n"
                "    XVectoropkernel_Set_b(&s_kernel,\n"
                "        b ? inference_buf_phys(b)"
                " + (uint64_t)b_off * INFERENCE_BYTES_PER_ELEM\n"
                "          : (uint64_t)0);\n"
                "    XVectoropkernel_Set_c(&s_kernel,\n"
                "        inference_buf_phys(c)"
                " + (uint64_t)c_off * INFERENCE_BYTES_PER_ELEM);\n"
                "    XVectoropkernel_Set_size(&s_kernel, size);\n"
                "    XVectoropkernel_Set_op(&s_kernel, op);\n"
                "    XVectoropkernel_Start(&s_kernel);\n"
                "    while (!XVectoropkernel_IsDone(&s_kernel)) {}\n"
                "}\n"
            )

        return "".join(parts)

    def _load_weight_helper(self) -> str:
        """
        Emit the _load_weight() helper used by inference_init() to read a .dat
        file into a DMA buffer.  Only included when the model has large weights.
        """
        return (
            _banner("_load_weight() — read an external weight .dat file") +
            "/*\n"
            " * _load_weight() — open weights/<name>.dat and fread() its contents\n"
            " * directly into the DMA buffer.  The .dat file contains raw little-\n"
            " * endian uint16 values in the same ap_fixed<16,8> encoding used by\n"
            " * the inline ROM arrays.\n"
            " *\n"
            " * INFERENCE_WEIGHTS_DIR is set by CMake (default \".\").\n"
            " * Override at configure time: cmake -DINFERENCE_WEIGHTS_DIR=/path/to/weights\n"
            " */\n"
            "#ifndef INFERENCE_WEIGHTS_DIR\n"
            "#  define INFERENCE_WEIGHTS_DIR  \".\"\n"
            "#endif\n"
            "\n"
            "static int _load_weight(inference_buf_t *buf,\n"
            "                        const char *name, unsigned n_elem)\n"
            "{\n"
            "    char path[512];\n"
            "    FILE *f;\n"
            "    size_t n_read;\n"
            "\n"
            "    snprintf(path, sizeof(path),\n"
            "             INFERENCE_WEIGHTS_DIR \"/weights/%s.dat\", name);\n"
            "    f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            '        fprintf(stderr,\n'
            '                "inference: cannot open weight file \'%s\'\\n", path);\n'
            "        return -1;\n"
            "    }\n"
            "    n_read = fread(inference_buf_ptr(buf),\n"
            "                  INFERENCE_BYTES_PER_ELEM, n_elem, f);\n"
            "    fclose(f);\n"
            "    if (n_read != (size_t)n_elem) {\n"
            '        fprintf(stderr,\n'
            '                "inference: short read from \'%s\':"\n'
            '                " got %zu, expected %u\\n",\n'
            '                path, n_read, n_elem);\n'
            "        return -1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    def _init_function(self) -> str:
        graph         = self._graph
        weights       = graph.weight_tensors
        intermediates = graph.intermediate_tensors

        alloc_lines: List[str] = []

        if weights:
            external = {t.onnx_name for t in self.large_weight_tensors}
            alloc_lines.append(
                "    /* Allocate DMA buffers for weights */"
            )
            for t in weights:
                alloc_size = self._alloc_sizes[t.onnx_name]
                alloc_lines.append(
                    f"    {t.c_name} = inference_buf_alloc({alloc_size}u);"
                )
                alloc_lines.append(
                    f"    if (!{t.c_name}) {{ rc = -1; goto fail; }}"
                )
                if t.onnx_name in external:
                    # Large weight: load data (t.numel elements) into the
                    # buffer (alloc_size elements, may be larger due to padding)
                    alloc_lines.append(
                        f"    if (_load_weight({t.c_name},"
                        f" \"{t.c_name}\", {t.numel}u) != 0) {{ rc = -1; goto fail; }}"
                    )
                else:
                    alloc_lines.append(
                        f"    memcpy(inference_buf_ptr({t.c_name}),"
                        f" _rom_{t.c_name}, sizeof(_rom_{t.c_name}));"
                    )
            alloc_lines.append("")

        if intermediates:
            alloc_lines.append("    /* Allocate intermediate buffers */")
            for t in intermediates:
                alloc_size = self._alloc_sizes[t.onnx_name]
                alloc_lines.append(
                    f"    {t.c_name} = inference_buf_alloc({alloc_size}u);"
                )
                alloc_lines.append(
                    f"    if (!{t.c_name}) {{ rc = -1; goto fail; }}"
                )
            alloc_lines.append("")

        alloc_str = ("\n".join(alloc_lines) + "\n") if alloc_lines else ""

        # inference_deinit(): free owned buffers (safe on NULL), then close pool.
        # Buffers must be freed before inference_buf_pool_deinit() because the
        # Linux/XRT path needs the device handle open to call xclFreeBO().
        deinit_free: List[str] = []
        for t in weights + intermediates:
            deinit_free.append(
                f"    inference_buf_free({t.c_name}); {t.c_name} = NULL;"
            )
        deinit_body = ("\n".join(deinit_free) + "\n") if deinit_free else ""

        load_helper = self._load_weight_helper() if self.large_weight_tensors else ""

        return (
            load_helper +
            _banner("inference_init() / inference_deinit()") +
            "/* Internal: pool lifecycle — defined in inference_buf.c */\n"
            "int  inference_buf_pool_init(void);\n"
            "void inference_buf_pool_deinit(void);\n"
            "\n"
            "int inference_init(const char *instance_name)\n"
            "{\n"
            "    int rc;\n"
            "\n"
            "    /* Initialise DMA buffer pool */\n"
            "    rc = inference_buf_pool_init();\n"
            "    if (rc != 0) return rc;\n"
            "\n"
            f"{alloc_str}"
            "    /* Initialise kernel driver */\n"
            "    rc = XVectoropkernel_Initialize(&s_kernel, instance_name);\n"
            "    if (rc != 0) goto fail;\n"
            "    return 0;\n"
            "\n"
            "fail:\n"
            "    inference_deinit();\n"
            "    return rc;\n"
            "}\n"
            "\n"
            "void inference_deinit(void)\n"
            "{\n"
            f"{deinit_body}"
            "    inference_buf_pool_deinit();\n"
            "}\n"
        )

    def _inference_function(self) -> str:
        graph   = self._graph
        inputs  = graph.input_tensors
        outputs = graph.output_tensors

        params = []
        for t in inputs:
            params.append(f"    inference_buf_t *{t.c_name}")
        for t in outputs:
            params.append(f"    inference_buf_t *{t.c_name}")
        param_str = ",\n".join(params)

        body_lines = []
        for sn in graph.nodes:
            body_lines.append(sn.emit_comment())
            if sn.outer_count == 1:
                # Pass the alloc_size so run_op() covers the full padded
                # buffer when the output inherited stride from upstream.
                body_lines.append(
                    sn.emit_call(op_size=self._alloc_sizes[sn.output.onnx_name])
                )
            else:
                body_lines.append(sn.emit_call())
            body_lines.append("")
        if body_lines and body_lines[-1] == "":
            body_lines.pop()
        body = "\n".join(body_lines) if body_lines else "    /* (empty graph) */"

        return (
            _banner("inference_run()") +
            "void inference_run(\n"
            f"{param_str})\n"
            "{\n"
            f"{body}\n"
            "}\n"
        )

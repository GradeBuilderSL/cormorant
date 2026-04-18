"""Mixin that generates test/test_inference.c."""

from __future__ import annotations

from ._banners import _file_banner


class _TestMixin:
    """Generates test/test_inference.c."""

    def generate_test(self) -> str:
        """Content for test/test_inference.c — one-shot smoke test.

        Allocates input/output DMA buffers, fills inputs with a ramp
        (element i = i/256.0 in ap_fixed<16,8> encoding), runs inference,
        and prints the first 8 output values.  Exit code 0 = success.

        For broadcast models, inputs are filled chunk-by-chunk (skipping
        alignment gap elements), and outputs are printed chunk-by-chunk so
        that gap elements — which VectorOPKernel never writes — are not
        included in the output.
        """
        graph     = self._graph
        inputs    = graph.input_tensors
        outputs   = graph.output_tensors
        bcast_map = self._broadcast_io_map()  # onnx_name → (n, chunk_macro, stride_macro)

        # Variable declarations at top of main()
        decl_lines = []
        for t in inputs + outputs:
            decl_lines.append(f"    inference_buf_t *{t.c_name} = NULL;")

        # Buffer allocations with goto-cleanup error handling.
        # Each failure jumps to 'cleanup' which frees all I/O buffers
        # (NULL-safe) and calls inference_deinit(), so no buffer leaks.
        alloc_lines = []
        for t in inputs + outputs:
            macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
            alloc_lines.append(f"    {t.c_name} = inference_buf_alloc({macro});")
            alloc_lines.append(f"    if (!{t.c_name}) {{")
            alloc_lines.append(
                f"        fprintf(stderr, "
                f"\"inference_buf_alloc failed for '{t.c_name}'\\n\");"
            )
            alloc_lines.append("        rc = 1; goto cleanup;")
            alloc_lines.append("    }")

        # Fill each input with a ramp: raw uint16 value i = i/256.0
        # For broadcasting inputs, iterate chunk-by-chunk so gap elements
        # (alignment padding between data blocks) are left untouched.
        fill_lines = []
        for t in inputs:
            if t.onnx_name in bcast_map:
                n, chunk_macro, stride_macro = bcast_map[t.onnx_name]
                fill_lines += [
                    f"    {{  /* '{t.c_name}' — broadcast input, fill data chunks only */",
                    f"        Data_t   *p = inference_buf_ptr({t.c_name});",
                    f"        unsigned  _chunk, _off;",
                    f"        for (_chunk = 0u; _chunk < {n}u; _chunk++) {{",
                    f"            _off = _chunk * {stride_macro};",
                    f"            for (i = 0u; i < {chunk_macro}; i++)",
                    f"                p[_off + i] = (Data_t)((_off + i) & 0xFFFFu); /* i/256.0 */",
                    f"        }}",
                    f"    }}",
                ]
            else:
                macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
                fill_lines += [
                    f"    {{",
                    f"        Data_t *p = inference_buf_ptr({t.c_name});",
                    f"        for (i = 0u; i < {macro}; i++)",
                    f"            p[i] = (Data_t)(i & 0xFFFFu); /* i/256.0 */",
                    f"    }}",
                ]

        # inference_run() call
        run_args = ", ".join(t.c_name for t in inputs + outputs)

        # Print first ≤8 elements of each output.
        # For broadcasting outputs, iterate chunk-by-chunk so alignment gap
        # elements — never written by VectorOPKernel — are skipped.
        print_lines = []
        for t in outputs:
            if t.onnx_name in bcast_map:
                n, chunk_macro, stride_macro = bcast_map[t.onnx_name]
                print_lines += [
                    f"    {{  /* '{t.c_name}' — broadcast output, print data chunks only */",
                    f"        Data_t   *p     = inference_buf_ptr({t.c_name});",
                    f"        unsigned  numel = {n}u * {chunk_macro};",
                    f"        unsigned  lim   = (numel < 8u) ? numel : 8u;",
                    f"        unsigned  shown = 0u, _chunk, _off;",
                    f"        printf(\"Output '{t.onnx_name}' (%u elem, first %u):\\n\","
                    f" numel, lim);",
                    f"        for (_chunk = 0u; _chunk < {n}u && shown < lim; _chunk++) {{",
                    f"            _off = _chunk * {stride_macro};",
                    f"            for (i = 0u; i < {chunk_macro} && shown < lim;"
                    f" i++, shown++) {{",
                    f"                int16_t s = (int16_t)p[_off + i];",
                    f"                printf(\"  [%u] 0x%04X  (%.4f)\\n\",",
                    f"                       shown, (unsigned)p[_off + i],"
                    f" (double)s / 256.0);",
                    f"            }}",
                    f"        }}",
                    f"    }}",
                ]
            else:
                macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
                print_lines += [
                    f"    {{",
                    f"        Data_t   *p   = inference_buf_ptr({t.c_name});",
                    f"        unsigned  lim = ({macro} < 8u) ? {macro} : 8u;",
                    f"        printf(\"Output '{t.onnx_name}'"
                    f" (%u elem, first %u):\\n\", (unsigned){macro}, lim);",
                    f"        for (i = 0u; i < lim; i++) {{",
                    f"            int16_t s = (int16_t)p[i];",
                    f"            printf(\"  [%u] 0x%04X  (%.4f)\\n\",",
                    f"                   i, (unsigned)p[i], (double)s / 256.0);",
                    f"        }}",
                    f"    }}",
                ]

        # Cleanup label: free all I/O buffers (NULL-safe), then deinit.
        cleanup_lines = []
        for t in inputs + outputs:
            cleanup_lines.append(f"    inference_buf_free({t.c_name});")

        decl_str    = "\n".join(decl_lines)
        alloc_str   = "\n".join(alloc_lines)
        fill_str    = "\n".join(fill_lines) if fill_lines else "    /* (no inputs) */"
        print_str   = "\n".join(print_lines)
        cleanup_str = "\n".join(cleanup_lines)

        return (
            _file_banner("test_inference.c", graph, self._model_path) +
            "\n"
            "#include <stdio.h>\n"
            "#include \"inference.h\"\n"
            "\n"
            "/*\n"
            " * Default hardware instance passed to inference_init().\n"
            " *   Linux:      UIO device (see /sys/class/uio/ for the right index)\n"
            " *   Bare-metal: device name string from xparameters.h\n"
            " * Override: cmake -DINFERENCE_TEST_INSTANCE=\\\"/dev/uio1\\\"\n"
            " *       or: ./test_inference /dev/uio1\n"
            " */\n"
            "#ifndef INFERENCE_TEST_INSTANCE\n"
            "#  ifdef __linux__\n"
            "#    define INFERENCE_TEST_INSTANCE  \"/dev/uio0\"\n"
            "#  else\n"
            "#    define INFERENCE_TEST_INSTANCE  \"VectorOPKernel\"\n"
            "#  endif\n"
            "#endif\n"
            "\n"
            "int main(int argc, char **argv)\n"
            "{\n"
            "    const char *instance = (argc > 1) ? argv[1]\n"
            "                                      : INFERENCE_TEST_INSTANCE;\n"
            f"{decl_str}\n"
            "    unsigned i;\n"
            "    int      rc = 0;\n"
            "\n"
            "    /* 1. Initialise: open UIO device, map DMA pool */\n"
            "    rc = inference_init(instance);\n"
            "    if (rc != 0) {\n"
            "        fprintf(stderr, \"inference_init(\\\"%s\\\") failed: %d\\n\",\n"
            "                instance, rc);\n"
            "        return 1;\n"
            "    }\n"
            "    printf(\"inference_init OK (\\\"%s\\\")\\n\", instance);\n"
            "\n"
            "    /* 2. Allocate I/O buffers from the DMA pool */\n"
            f"{alloc_str}\n"
            "\n"
            "    /* 3. Fill inputs: ramp pattern (element i = i/256.0) */\n"
            f"{fill_str}\n"
            "\n"
            "    /* 4. Run inference */\n"
            f"    inference_run({run_args});\n"
            "\n"
            "    /* 5. Print first 8 output elements */\n"
            f"{print_str}\n"
            "\n"
            "    if (rc == 0)\n"
            "        printf(\"test_inference PASSED\\n\");\n"
            "\n"
            "    /* 6. Teardown — reached on both success and alloc failure */\n"
            "cleanup:\n"
            f"{cleanup_str}\n"
            "    inference_deinit();\n"
            "    return rc;\n"
            "}\n"
        )

cmake_minimum_required(VERSION 3.19)
project(axi_kernels)

# ---------------------------------------------------------------------------
# All four HLS kernels.  Each subdirectory is a self-contained CMake project.
#
# Simulation test targets (no hardware needed):
#   make TestSimulation   — VectorOPKernel
#   make TestConvRef      — ConvKernel
#   make TestMatmulRef    — MatmulKernel   (TestMatmulBlas if BLAS available)
#   make TestPoolingSim   — PoolingKernel
#
# HLS synthesis + IP export targets (requires Vitis):
#   make synthesize_<platform>              — all four kernels
#   make synthesize_<kernel>_<platform>     — individual kernel
#
# Platform definitions live in platforms/*.json (shared by all kernels).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global hardware parameters — shared by all kernels.
# ---------------------------------------------------------------------------

# AXI master bus data width.  Valid values: 32, 64, 128, 256, 512.
# Wider buses amortise burst overhead; must match the interconnect configured
# in the Vivado block design (MAXI_DATA_WIDTH for each kernel IP).
set(AXI_BUS_WIDTH 32 CACHE STRING
    "AXI master bus data width in bits (32, 64, 128, 256, 512)")
set_property(CACHE AXI_BUS_WIDTH PROPERTY STRINGS 32 64 128 256 512)

# Collect platform files once; kernels iterate over this list.
file(GLOB AXI_PLATFORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/*.json")
if(NOT AXI_PLATFORM_FILES)
    message(WARNING "axi_kernels: no platform JSON files found in platforms/")
endif()

add_subdirectory(kernels/vectorop)
add_subdirectory(kernels/conv)
add_subdirectory(kernels/matmul)
add_subdirectory(kernels/pool)

# ---------------------------------------------------------------------------
# dtc — device tree compiler (required for dtbo targets)
# ---------------------------------------------------------------------------
find_program(DTC_EXECUTABLE dtc)
if(DTC_EXECUTABLE)
    message(STATUS "axi_kernels: dtc found at ${DTC_EXECUTABLE}")
else()
    message(STATUS "axi_kernels: dtc not found; dtbo targets will fail at build time")
    message(STATUS "             Install: sudo apt install device-tree-compiler")
endif()

# ---------------------------------------------------------------------------
# Convenience aggregate synthesis targets — one per platform file.
# Device tree overlay targets — one per .dts file under dts/<platform>/.
# ---------------------------------------------------------------------------
foreach(_PF IN LISTS AXI_PLATFORM_FILES)
    get_filename_component(_pname "${_PF}" NAME_WE)

    add_custom_target("synthesize_${_pname}"
        DEPENDS
            "synthesize_vectorop_${_pname}"
            "synthesize_conv_${_pname}"
            "synthesize_matmul_${_pname}"
            "synthesize_pool_${_pname}"
        COMMENT "HLS synthesis: all kernels for platform '${_pname}'"
    )
    message(STATUS "axi_kernels: aggregate target 'synthesize_${_pname}'")

    file(GLOB _DTS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/dts/${_pname}/*.dts")
    foreach(_DTS IN LISTS _DTS_FILES)
        get_filename_component(_stem "${_DTS}" NAME_WE)
        set(_DTBO_DIR "${CMAKE_CURRENT_BINARY_DIR}/dts/${_pname}")
        set(_DTBO     "${_DTBO_DIR}/design_${_stem}.dtbo")

        if(DTC_EXECUTABLE)
            add_custom_command(
                OUTPUT  "${_DTBO}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DTBO_DIR}"
                COMMAND "${DTC_EXECUTABLE}" -@ -O dtb -o "${_DTBO}" "${_DTS}"
                DEPENDS "${_DTS}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dts/${_pname}"
                COMMENT "DTC: ${_stem}.dts → design_${_stem}.dtbo"
                VERBATIM
            )
        else()
            add_custom_command(
                OUTPUT  "${_DTBO}"
                COMMAND "${CMAKE_COMMAND}" -E echo
                        "ERROR: dtc not found -- install device-tree-compiler"
                COMMAND "${CMAKE_COMMAND}" -E false
                DEPENDS "${_DTS}"
                VERBATIM
            )
        endif()

        add_custom_target("dtbo_${_pname}_${_stem}"
            DEPENDS "${_DTBO}"
            COMMENT "Device tree overlay: ${_stem} for platform '${_pname}'"
        )
        message(STATUS "axi_kernels: target 'dtbo_${_pname}_${_stem}'")
    endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Hardware build target — Vivado synthesis + implementation + bitstream.
# Requires the cormorant_hw_128 submodule and Vivado 2025.2.
# Must configure with -DAXI_BUS_WIDTH=128 to match the 128-bit block design.
#
# Usage:
#   cmake .. -DAXI_BUS_WIDTH=128
#   make synthesize_kv260      # build all four HLS kernel IPs first
#   make build_hw_kv260        # then run Vivado on the block design
#
# Or in one shot (synthesize_kv260 is listed as a dependency):
#   make build_hw_kv260
# ---------------------------------------------------------------------------
set(_HW_REPO "${CMAKE_SOURCE_DIR}/hw/cormorant_hw_128")
if(EXISTS "${_HW_REPO}/build.sh")
    add_custom_target(build_hw_kv260
        COMMAND bash "${_HW_REPO}/build.sh" all
                -ip-repo "${CMAKE_BINARY_DIR}/kernels"
        WORKING_DIRECTORY "${_HW_REPO}"
        DEPENDS synthesize_kv260
        COMMENT "Vivado: synthesis + implementation + bitstream (KV260 128-bit)"
    )
    message(STATUS "axi_kernels: target 'build_hw_kv260' (depends on synthesize_kv260)")
else()
    message(STATUS "axi_kernels: 'build_hw_kv260' not available — submodule not initialised")
    message(STATUS "             Run: git submodule update --init hw/cormorant_hw_128")
endif()

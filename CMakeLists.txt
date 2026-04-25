cmake_minimum_required(VERSION 3.19)
project(axi_kernels)

# ---------------------------------------------------------------------------
# All four HLS kernels.  Each subdirectory is a self-contained CMake project
# that can also be built standalone (cmake kernels/<name>/).
#
# Simulation test targets (no hardware needed):
#   make TestSimulation   — VectorOPKernel
#   make TestConvRef      — ConvKernel
#   make TestMatmulRef    — MatmulKernel   (TestMatmulBlas if BLAS available)
#   make TestPoolingSim   — PoolingKernel
#
# HLS synthesis + IP export targets (requires Vitis):
#   make synthesize_vectorop_<platform>
#   make synthesize_conv_<platform>
#   make synthesize_matmul_<platform>
#   make synthesize_pool_<platform>
#
# Platform names come from JSON files in each kernel's platforms/ directory.
# ---------------------------------------------------------------------------

add_subdirectory(kernels/vectorop)
add_subdirectory(kernels/conv)
add_subdirectory(kernels/matmul)
add_subdirectory(kernels/pool)

# ---------------------------------------------------------------------------
# Convenience aggregate synthesis targets.
# Each synthesize_<kernel>_<platform> target is defined in the kernel's own
# CMakeLists.txt; the targets below just pull them all together.
# ---------------------------------------------------------------------------
file(GLOB _PLATFORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vectorop/platforms/*.json")
foreach(_PF IN LISTS _PLATFORM_FILES)
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
endforeach()

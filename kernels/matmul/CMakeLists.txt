cmake_minimum_required(VERSION 3.19)
project(MatmulKernel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
#
# The headers are header-only and work with standard GCC — no HLS synthesis
# flow is needed.  If not found, Data_t falls back to float and tests still
# verify correctness, but saturation behaviour won't be tested.
# ---------------------------------------------------------------------------
find_path(HLS_INCLUDE_DIR NAMES ap_fixed.h
    HINTS
        $ENV{XILINX_HLS}/include
        $ENV{XILINX_VITIS}/include
        /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(HLS_INCLUDE_DIR)
    message(STATUS "MatmulKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_MM_HAVE_HLS TRUE)
else()
    message(STATUS "MatmulKernel: Vitis HLS headers not found — "
                   "using float/double fallback (no saturation tests)")
    set(_MM_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_MM_HAVE_HLS)
    set(MM_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(MM_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(MM_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(MM_HAVE_APFIXED_DEFINE "#define MATMUL_HAVE_APFIXED")
else()
    set(MM_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(MM_ACC_DATA_TYPE "double" CACHE STRING "Accumulator type")
    set(MM_DATA_TYPE_INCLUDE  "")
    set(MM_HAVE_APFIXED_DEFINE "")
endif()

# ---------------------------------------------------------------------------
# Tile sizes
# ---------------------------------------------------------------------------
set(MM_TILE_N  4    CACHE STRING
    "Row interleave depth (must be power-of-two; >= MAC latency for II=1)")
set(MM_TILE_M  16   CACHE STRING
    "Parallel output columns per cycle (must be power-of-two)")
set(MM_TILE_K  256  CACHE STRING
    "On-chip B-buffer row count (must be power-of-two)")
set(MM_MAX_K   2048 CACHE STRING
    "Compile-time a_buf row length limit; K > kMaxK is rejected at scheduling")

# ---------------------------------------------------------------------------
# Generate Config.h into the build directory
# ---------------------------------------------------------------------------
configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

# ---------------------------------------------------------------------------
# Kernel library — MatmulKernel.cpp compiled as a static lib so that both
# the C++ simulation target and the future HLS synthesis target can link it.
# ---------------------------------------------------------------------------
add_library(matmul_kernel STATIC
    kernel/MatmulKernel.cpp
)
target_include_directories(matmul_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # MatmulKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_MM_HAVE_HLS)
    target_include_directories(matmul_kernel SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
    # MATMUL_HAVE_APFIXED is already defined in Config.h via MM_HAVE_APFIXED_DEFINE;
    # no need to pass it again via -D (that would cause a redefinition warning).
endif()
target_compile_options(matmul_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# Float kernel library — same MatmulKernel.cpp but always compiled with
# Data_t=float / AccData_t=float regardless of HLS availability.
#
# Used by TestMatmulBlas so that cblas_sgemm (float) can serve as the oracle.
# A separate float_config/Config.h is generated into the build directory so
# the include path override works without touching the primary Config.h.
# ---------------------------------------------------------------------------

# Generate float_config/Config.h by temporarily swapping the type variables.
set(_SAVE_DATA_TYPE          "${MM_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE      "${MM_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE  "${MM_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${MM_HAVE_APFIXED_DEFINE}")

set(MM_DATA_TYPE          "float")
set(MM_ACC_DATA_TYPE      "float")
set(MM_DATA_TYPE_INCLUDE  "")
set(MM_HAVE_APFIXED_DEFINE "")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(MM_DATA_TYPE          "${_SAVE_DATA_TYPE}")
set(MM_ACC_DATA_TYPE      "${_SAVE_ACC_DATA_TYPE}")
set(MM_DATA_TYPE_INCLUDE  "${_SAVE_DATA_TYPE_INCLUDE}")
set(MM_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")

add_library(matmul_kernel_float STATIC
    kernel/MatmulKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(matmul_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # MatmulKernel.h
)
target_compile_options(matmul_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestMatmulRef — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  The name "Ref" reflects that
# this is the Phase 1 reference; Phase 2 will rename/extend to TestMatmulSim
# once the HLS kernel function is wired in.
# ---------------------------------------------------------------------------
add_executable(TestMatmulRef
    test/TestMatmulSim.cpp
)
target_link_libraries(TestMatmulRef PRIVATE matmul_kernel)
target_compile_options(TestMatmulRef PRIVATE -Wall -Wextra -O2)

# Register with CTest so "ctest" runs it.
enable_testing()
add_test(NAME TestMatmulRef COMMAND TestMatmulRef)

# ---------------------------------------------------------------------------
# TestMatmulBlas — validates the float kernel against cblas_sgemm.
#
# Requires libblas-dev (provides cblas.h and libblas.so).  Skipped silently
# if BLAS is not found so the build remains self-contained on bare systems.
# ---------------------------------------------------------------------------
find_package(BLAS)
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
    HINTS
        /usr/include/x86_64-linux-gnu
        /usr/include
    DOC "Directory containing cblas.h"
)

if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
    message(STATUS "MatmulKernel: BLAS found — TestMatmulBlas enabled "
                   "(cblas.h at ${CBLAS_INCLUDE_DIR})")
    add_executable(TestMatmulBlas
        test/TestMatmulBlas.cpp
    )
    target_link_libraries(TestMatmulBlas PRIVATE matmul_kernel_float ${BLAS_LIBRARIES})
    target_include_directories(TestMatmulBlas PRIVATE ${CBLAS_INCLUDE_DIR})
    target_compile_options(TestMatmulBlas PRIVATE -Wall -Wextra -O2)
    add_test(NAME TestMatmulBlas COMMAND TestMatmulBlas)
else()
    message(STATUS "MatmulKernel: BLAS not found — TestMatmulBlas skipped")
endif()

message(STATUS "MatmulKernel: targets — matmul_kernel, matmul_kernel_float, TestMatmulRef")
message(STATUS "MatmulKernel: Data_t=${MM_DATA_TYPE}  "
               "AccData_t=${MM_ACC_DATA_TYPE}  "
               "TILE_N=${MM_TILE_N}  TILE_M=${MM_TILE_M}  "
               "TILE_K=${MM_TILE_K}  MAX_K=${MM_MAX_K}")

# ---------------------------------------------------------------------------
# HLS synthesis targets — one per platform JSON file in platforms/.
#
# Requires Vitis HLS (vitis-run for 2024.x+, vitis_hls for older versions).
# Each JSON must have a "part" field; "board" and "clock" are optional.
#
# Generated targets:
#   make synthesize_<name>   — C synthesis + ip_catalog export for that part
# ---------------------------------------------------------------------------
find_program(MM_VITIS_HLS NAMES vitis-run vitis_hls
    HINTS
        $ENV{XILINX_HLS}/bin
        $ENV{XILINX_VITIS}/bin
        /mnt/data/xilinx/2025.2/Vitis/bin
    DOC "Vitis HLS executable (vitis-run for 2024.x+, vitis_hls for older)")

if(MM_VITIS_HLS)
    # vitis-run (2024.x+) uses --tcl; the old vitis_hls used -f.
    if(MM_VITIS_HLS MATCHES "vitis-run")
        set(MM_HLS_TCL_FLAG "--tcl")
    else()
        set(MM_HLS_TCL_FLAG "-f")
    endif()
    message(STATUS "MatmulKernel: HLS synthesis enabled via ${MM_VITIS_HLS} ${MM_HLS_TCL_FLAG}")
else()
    message(STATUS "MatmulKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

if(MM_VITIS_HLS)
    set(MM_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatmulKernel.cpp")

    set(_MM_PLATFORM_FILES "${AXI_PLATFORM_FILES}")
    foreach(_MM_PLATFORM_FILE IN LISTS _MM_PLATFORM_FILES)
        get_filename_component(_pname "${_MM_PLATFORM_FILE}" NAME_WE)
        file(READ "${_MM_PLATFORM_FILE}" _pjson)

        # Required: FPGA part string
        string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
        if(_e1)
            message(WARNING "MatmulKernel platform '${_pname}': missing 'part' — skipping.")
            continue()
        endif()

        # Optional: board identifier
        string(JSON _p_board ERROR_VARIABLE _e GET "${_pjson}" board)
        if(_e)
            set(_p_board "")
        endif()

        # Optional: target clock in MHz (default 300)
        string(JSON _p_clock ERROR_VARIABLE _e GET "${_pjson}" clock)
        if(_e)
            set(_p_clock "300")
        endif()

        # Optional: AXI master bus width in bits (default: global AXI_BUS_WIDTH)
        string(JSON _p_axi_bus_width ERROR_VARIABLE _e GET "${_pjson}" axi_bus_width)
        if(_e OR NOT _p_axi_bus_width)
            set(_p_axi_bus_width "$CACHE{AXI_BUS_WIDTH}")
        elseif(NOT _p_axi_bus_width MATCHES "^(32|64|128|256|512)$")
            message(WARNING "MatmulKernel platform '${_pname}': invalid axi_bus_width "
                            "'${_p_axi_bus_width}' — must be 32, 64, 128, 256, or 512. "
                            "Falling back to global AXI_BUS_WIDTH=$CACHE{AXI_BUS_WIDTH}.")
            set(_p_axi_bus_width "$CACHE{AXI_BUS_WIDTH}")
        endif()

        # Per-platform build directory
        set(_p_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_pname}")
        file(MAKE_DIRECTORY "${_p_build_dir}")

        # Generate a platform-specific Config.h using the main type settings.
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h.in"
            "${_p_build_dir}/Config.h"
            @ONLY)

        # set_part TCL command (with optional -board)
        if(_p_board)
            set(MM_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
        else()
            set(MM_BOARD_OPTION "set_part {${_p_part}}")
        endif()

        # Variables consumed by Synthesis.tcl.in (@ONLY substitution)
        set(MM_HLS_PROJECT_NAME  "matmul_${_pname}")
        set(MM_SRC_SYNTHESIS     "${MM_KERNEL_SRC}")
        set(MM_SYNTHESIS_FLAGS
            "-O3 -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
        set(MM_TARGET_CLOCK_TCL  "${_p_clock}")
        set(MM_IP_OUTPUT_DIR     "${_p_build_dir}/ip_catalog")
        # Shadow the global AXI_BUS_WIDTH so configure_file picks up the per-platform value.
        set(AXI_BUS_WIDTH        "${_p_axi_bus_width}")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Synthesis.tcl.in"
            "${_p_build_dir}/synthesize_${_pname}.tcl"
            @ONLY)

        add_custom_target(
            "synthesize_matmul_${_pname}"
            COMMAND ${MM_VITIS_HLS} ${MM_HLS_TCL_FLAG}
                    "${_p_build_dir}/synthesize_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synthesis: MatmulKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz, ${_p_axi_bus_width}-bit AXI)"
            DEPENDS
                "${MM_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/MatmulKernel.h"
                "${_p_build_dir}/Config.h")

        message(STATUS "MatmulKernel: synthesis target 'synthesize_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz, ${_p_axi_bus_width}-bit AXI)")
    endforeach()
endif()

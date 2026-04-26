cmake_minimum_required(VERSION 3.19)
project(ConvKernel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate Vitis HLS headers (ap_fixed.h) for fixed-point C++ simulation.
# The headers are header-only — no HLS synthesis flow is needed for C-sim.
# If not found, Data_t falls back to float and tests still verify correctness,
# but saturation behaviour won't be tested.
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
    message(STATUS "ConvKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_CONV_HAVE_HLS TRUE)
else()
    message(STATUS "ConvKernel: Vitis HLS headers not found — "
                   "using float fallback (no saturation tests)")
    set(_CONV_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_CONV_HAVE_HLS)
    set(CONV_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(CONV_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(CONV_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(CONV_HAVE_APFIXED_DEFINE "#define CONV_HAVE_APFIXED")
else()
    set(CONV_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(CONV_ACC_DATA_TYPE "float"  CACHE STRING "Accumulator type")
    set(CONV_DATA_TYPE_INCLUDE  "")
    set(CONV_HAVE_APFIXED_DEFINE "")
endif()

# ---------------------------------------------------------------------------
# Tile sizes
# ---------------------------------------------------------------------------
set(CONV_TILE_M  8  CACHE STRING
    "Output channel tile (must be power-of-two; controls II=1 lane rotation depth)")
set(CONV_TILE_IC 16 CACHE STRING
    "Input channel tile (must be power-of-two; sets weight buffer IC depth)")
set(CONV_MAX_KH  7  CACHE STRING
    "Compile-time kernel height limit; Conv nodes with kH > MAX_KH are rejected at scheduling")
set(CONV_MAX_KW  7  CACHE STRING
    "Compile-time kernel width limit; Conv nodes with kW > MAX_KW are rejected at scheduling")

# ---------------------------------------------------------------------------
# Generate Config.h into the build directory
# ---------------------------------------------------------------------------
configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

# ---------------------------------------------------------------------------
# Kernel library — ConvKernel.cpp compiled as a static lib so that both
# the C++ simulation target and the future HLS synthesis target can link it.
# ---------------------------------------------------------------------------
add_library(conv_kernel STATIC
    kernel/ConvKernel.cpp
)
target_include_directories(conv_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # ConvKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_CONV_HAVE_HLS)
    target_include_directories(conv_kernel SYSTEM PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(conv_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# Float kernel library — same ConvKernel.cpp compiled with Data_t=float.
# Used by any future float-reference tests without the ap_fixed dependency.
#
# A separate float_config/Config.h is generated so the include path override
# works without touching the primary Config.h.
# ---------------------------------------------------------------------------
set(_SAVE_DATA_TYPE          "${CONV_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE      "${CONV_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE  "${CONV_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${CONV_HAVE_APFIXED_DEFINE}")

set(CONV_DATA_TYPE          "float")
set(CONV_ACC_DATA_TYPE      "float")
set(CONV_DATA_TYPE_INCLUDE  "")
set(CONV_HAVE_APFIXED_DEFINE "")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(CONV_DATA_TYPE          "${_SAVE_DATA_TYPE}")
set(CONV_ACC_DATA_TYPE      "${_SAVE_ACC_DATA_TYPE}")
set(CONV_DATA_TYPE_INCLUDE  "${_SAVE_DATA_TYPE_INCLUDE}")
set(CONV_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")

add_library(conv_kernel_float STATIC
    kernel/ConvKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(conv_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # ConvKernel.h
)
target_compile_options(conv_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestConvRef — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  Links against conv_kernel
# (ap_fixed<16,8> when HLS headers are available, float otherwise).
# ---------------------------------------------------------------------------
add_executable(TestConvRef
    test/TestConvSim.cpp
)
target_link_libraries(TestConvRef PRIVATE conv_kernel)
target_compile_options(TestConvRef PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME TestConvRef COMMAND TestConvRef)

message(STATUS "ConvKernel: targets — conv_kernel, conv_kernel_float, TestConvRef")
message(STATUS "ConvKernel: Data_t=${CONV_DATA_TYPE}  "
               "AccData_t=${CONV_ACC_DATA_TYPE}  "
               "TILE_M=${CONV_TILE_M}  TILE_IC=${CONV_TILE_IC}  "
               "MAX_KH=${CONV_MAX_KH}  MAX_KW=${CONV_MAX_KW}")

# ---------------------------------------------------------------------------
# HLS synthesis targets — one per platform JSON file in platforms/.
#
# Requires Vitis HLS (vitis-run for 2024.x+, vitis_hls for older versions).
# Each JSON must have a "part" field; "board" and "clock" are optional.
#
# Generated targets:
#   make synthesize_<name>   — C synthesis + ip_catalog export for that part
# ---------------------------------------------------------------------------
find_program(CONV_VITIS_HLS NAMES vitis-run vitis_hls
    HINTS
        $ENV{XILINX_HLS}/bin
        $ENV{XILINX_VITIS}/bin
        /mnt/data/xilinx/2025.2/Vitis/bin
    DOC "Vitis HLS executable (vitis-run for 2024.x+, vitis_hls for older)")

if(CONV_VITIS_HLS)
    if(CONV_VITIS_HLS MATCHES "vitis-run")
        set(CONV_HLS_TCL_FLAG "--tcl")
    else()
        set(CONV_HLS_TCL_FLAG "-f")
    endif()
    message(STATUS "ConvKernel: HLS synthesis enabled via ${CONV_VITIS_HLS} ${CONV_HLS_TCL_FLAG}")
else()
    message(STATUS "ConvKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

if(CONV_VITIS_HLS)
    set(CONV_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel/ConvKernel.cpp")

    set(_CONV_PLATFORM_FILES "${AXI_PLATFORM_FILES}")
    foreach(_CONV_PLATFORM_FILE IN LISTS _CONV_PLATFORM_FILES)
        get_filename_component(_pname "${_CONV_PLATFORM_FILE}" NAME_WE)
        file(READ "${_CONV_PLATFORM_FILE}" _pjson)

        # Required: FPGA part string
        string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
        if(_e1)
            message(WARNING "ConvKernel platform '${_pname}': missing 'part' — skipping.")
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

        # Per-platform build directory
        set(_p_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_pname}")
        file(MAKE_DIRECTORY "${_p_build_dir}")

        # Generate a platform-specific Config.h using the main type settings.
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h.in"
            "${_p_build_dir}/Config.h"
            @ONLY)

        if(_p_board)
            set(CONV_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
        else()
            set(CONV_BOARD_OPTION "set_part {${_p_part}}")
        endif()

        set(CONV_HLS_PROJECT_NAME  "conv_${_pname}")
        set(CONV_SRC_SYNTHESIS     "${CONV_KERNEL_SRC}")
        set(CONV_SYNTHESIS_FLAGS
            "-O3 -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
        set(CONV_TARGET_CLOCK_TCL  "${_p_clock}")
        set(CONV_IP_OUTPUT_DIR     "${_p_build_dir}/ip_catalog")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Synthesis.tcl.in"
            "${_p_build_dir}/synthesize_${_pname}.tcl"
            @ONLY)

        add_custom_target(
            "synthesize_conv_${_pname}"
            COMMAND ${CONV_VITIS_HLS} ${CONV_HLS_TCL_FLAG}
                    "${_p_build_dir}/synthesize_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synthesis: ConvKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${CONV_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ConvKernel.h"
                "${_p_build_dir}/Config.h")

        message(STATUS "ConvKernel: synthesis target 'synthesize_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")
    endforeach()
endif()

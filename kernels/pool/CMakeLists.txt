cmake_minimum_required(VERSION 3.19)
project(PoolingKernel VERSION 1.0 LANGUAGES CXX)

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
    message(STATUS "PoolingKernel: Vitis HLS headers found at ${HLS_INCLUDE_DIR}")
    set(_POOL_HAVE_HLS TRUE)
else()
    message(STATUS "PoolingKernel: Vitis HLS headers not found — "
                   "using float fallback (no saturation tests)")
    set(_POOL_HAVE_HLS FALSE)
endif()

# ---------------------------------------------------------------------------
# Data type configuration
# ---------------------------------------------------------------------------
if(_POOL_HAVE_HLS)
    set(POOL_DATA_TYPE     "ap_fixed<16,8>"  CACHE STRING "Element type")
    set(POOL_ACC_DATA_TYPE "ap_fixed<32,16>" CACHE STRING "Accumulator type")
    set(POOL_DATA_TYPE_INCLUDE  "#include \"ap_fixed.h\"")
    set(POOL_HAVE_APFIXED_DEFINE "#define POOL_HAVE_APFIXED")
    # Minimum representable values — used as MAX pool sentinels.
    # ap_fixed<16,8> min = -128; ap_fixed<32,16> min = -32768.
    # AP_WRAP (default) wraps on overflow, so we must stay in-range.
    set(POOL_DATA_MIN "-128.0f")
    set(POOL_ACC_MIN  "-32768.0f")
else()
    set(POOL_DATA_TYPE     "float"  CACHE STRING "Element type")
    set(POOL_ACC_DATA_TYPE "float"  CACHE STRING "Accumulator type")
    set(POOL_DATA_TYPE_INCLUDE  "")
    set(POOL_HAVE_APFIXED_DEFINE "")
    set(POOL_DATA_MIN "-1.0e30f")
    set(POOL_ACC_MIN  "-1.0e30f")
endif()

# ---------------------------------------------------------------------------
# Tile sizes and window limits
# ---------------------------------------------------------------------------
set(POOL_TILE_C  8  CACHE STRING
    "Channel tile (must be power-of-two; controls II=1 lane rotation depth)")
set(POOL_MAX_KH  7  CACHE STRING
    "Compile-time pool window height limit; models with pool_h > MAX_KH are rejected")
set(POOL_MAX_KW  7  CACHE STRING
    "Compile-time pool window width limit; models with pool_w > MAX_KW are rejected")

# ---------------------------------------------------------------------------
# Generate Config.h into the build directory
# ---------------------------------------------------------------------------
configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

# ---------------------------------------------------------------------------
# Kernel library — PoolingKernel.cpp compiled as a static lib so that both
# the C++ simulation target and the future HLS synthesis target can link it.
# ---------------------------------------------------------------------------
add_library(pool_kernel STATIC
    kernel/PoolingKernel.cpp
)
target_include_directories(pool_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # PoolingKernel.h
        ${CMAKE_CURRENT_BINARY_DIR}          # Config.h
)
if(_POOL_HAVE_HLS)
    target_include_directories(pool_kernel PUBLIC ${HLS_INCLUDE_DIR})
endif()
target_compile_options(pool_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# Float kernel library — same PoolingKernel.cpp compiled with Data_t=float.
# Used by any future float-reference tests without the ap_fixed dependency.
# ---------------------------------------------------------------------------
set(_SAVE_DATA_TYPE           "${POOL_DATA_TYPE}")
set(_SAVE_ACC_DATA_TYPE       "${POOL_ACC_DATA_TYPE}")
set(_SAVE_DATA_TYPE_INCLUDE   "${POOL_DATA_TYPE_INCLUDE}")
set(_SAVE_HAVE_APFIXED_DEFINE "${POOL_HAVE_APFIXED_DEFINE}")
set(_SAVE_DATA_MIN            "${POOL_DATA_MIN}")
set(_SAVE_ACC_MIN             "${POOL_ACC_MIN}")

set(POOL_DATA_TYPE           "float")
set(POOL_ACC_DATA_TYPE       "float")
set(POOL_DATA_TYPE_INCLUDE   "")
set(POOL_HAVE_APFIXED_DEFINE "")
set(POOL_DATA_MIN            "-1.0e30f")
set(POOL_ACC_MIN             "-1.0e30f")

configure_file(
    include/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/float_config/Config.h
    @ONLY
)

set(POOL_DATA_TYPE           "${_SAVE_DATA_TYPE}")
set(POOL_ACC_DATA_TYPE       "${_SAVE_ACC_DATA_TYPE}")
set(POOL_DATA_TYPE_INCLUDE   "${_SAVE_DATA_TYPE_INCLUDE}")
set(POOL_HAVE_APFIXED_DEFINE "${_SAVE_HAVE_APFIXED_DEFINE}")
set(POOL_DATA_MIN            "${_SAVE_DATA_MIN}")
set(POOL_ACC_MIN             "${_SAVE_ACC_MIN}")

add_library(pool_kernel_float STATIC
    kernel/PoolingKernel.cpp
)
# float_config must come before include/ so its Config.h shadows the primary one.
target_include_directories(pool_kernel_float
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/float_config  # float Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include       # PoolingKernel.h
)
target_compile_options(pool_kernel_float PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

# ---------------------------------------------------------------------------
# TestPoolingSim — C++ simulation test, no Vitis HLS synthesis required.
#
# Compiles and runs entirely with standard GCC.  Links against pool_kernel
# (ap_fixed<16,8> when HLS headers are available, float otherwise).
# ---------------------------------------------------------------------------
add_executable(TestPoolingSim
    test/TestPoolingSim.cpp
)
target_link_libraries(TestPoolingSim PRIVATE pool_kernel)
target_compile_options(TestPoolingSim PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME TestPoolingSim COMMAND TestPoolingSim)

message(STATUS "PoolingKernel: targets — pool_kernel, pool_kernel_float, TestPoolingSim")
message(STATUS "PoolingKernel: Data_t=${POOL_DATA_TYPE}  "
               "AccData_t=${POOL_ACC_DATA_TYPE}  "
               "TILE_C=${POOL_TILE_C}  "
               "MAX_KH=${POOL_MAX_KH}  MAX_KW=${POOL_MAX_KW}")

# ---------------------------------------------------------------------------
# HLS synthesis targets — one per platform JSON file in platforms/.
#
# Requires Vitis HLS (vitis-run for 2024.x+, vitis_hls for older versions).
# Each JSON must have a "part" field; "board" and "clock" are optional.
#
# Generated targets:
#   make synthesize_<name>   — C synthesis + ip_catalog export for that part
# ---------------------------------------------------------------------------
find_program(POOL_VITIS_HLS NAMES vitis-run vitis_hls
    HINTS
        $ENV{XILINX_HLS}/bin
        $ENV{XILINX_VITIS}/bin
        /mnt/data/xilinx/2025.2/Vitis/bin
    DOC "Vitis HLS executable (vitis-run for 2024.x+, vitis_hls for older)")

if(POOL_VITIS_HLS)
    if(POOL_VITIS_HLS MATCHES "vitis-run")
        set(POOL_HLS_TCL_FLAG "--tcl")
    else()
        set(POOL_HLS_TCL_FLAG "-f")
    endif()
    message(STATUS "PoolingKernel: HLS synthesis enabled via ${POOL_VITIS_HLS} ${POOL_HLS_TCL_FLAG}")
else()
    message(STATUS "PoolingKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

if(POOL_VITIS_HLS)
    set(POOL_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel/PoolingKernel.cpp")

    file(GLOB _POOL_PLATFORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/*.json")
    foreach(_POOL_PLATFORM_FILE IN LISTS _POOL_PLATFORM_FILES)
        get_filename_component(_pname "${_POOL_PLATFORM_FILE}" NAME_WE)
        file(READ "${_POOL_PLATFORM_FILE}" _pjson)

        # Required: FPGA part string
        string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
        if(_e1)
            message(WARNING "PoolingKernel platform '${_pname}': missing 'part' — skipping.")
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
            set(POOL_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
        else()
            set(POOL_BOARD_OPTION "set_part {${_p_part}}")
        endif()

        set(POOL_HLS_PROJECT_NAME  "pool_${_pname}")
        set(POOL_SRC_SYNTHESIS     "${POOL_KERNEL_SRC}")
        set(POOL_SYNTHESIS_FLAGS
            "-O3 -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
        set(POOL_TARGET_CLOCK_TCL  "${_p_clock}")
        set(POOL_IP_OUTPUT_DIR     "${_p_build_dir}/ip_catalog")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Synthesis.tcl.in"
            "${_p_build_dir}/synthesize_${_pname}.tcl"
            @ONLY)

        add_custom_target(
            "synthesize_pool_${_pname}"
            COMMAND ${POOL_VITIS_HLS} ${POOL_HLS_TCL_FLAG}
                    "${_p_build_dir}/synthesize_${_pname}.tcl"
            WORKING_DIRECTORY "${_p_build_dir}"
            COMMENT "HLS synthesis: PoolingKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz)"
            DEPENDS
                "${POOL_KERNEL_SRC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/PoolingKernel.h"
                "${_p_build_dir}/Config.h")

        message(STATUS "PoolingKernel: synthesis target 'synthesize_${_pname}' "
                       "(${_p_part} @ ${_p_clock} MHz)")
    endforeach()
endif()

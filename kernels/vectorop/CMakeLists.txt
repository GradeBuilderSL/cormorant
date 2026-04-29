cmake_minimum_required(VERSION 3.19)  # 3.19+ required for string(JSON ...)
project(VectorOPKernel)

# ---------------------------------------------------------------------------
# User-facing parameters
# ---------------------------------------------------------------------------

# Vitis platform and clock — only needed for the Vitis hw/hw_emu flow.
set(VA_PLATFORM "xilinx_u250_gen3x16_xdma_3_1_202020_1" CACHE STRING
    "Vitis platform string.")
set(VA_TARGET_CLOCK "" CACHE STRING
    "Target clock in MHz for the Vitis flow (empty = platform default).")

# ---------------------------------------------------------------------------
# Auto-detect Vitis HLS headers (ap_fixed.h)
# ---------------------------------------------------------------------------
find_path(VA_HLS_INCLUDE_DIR NAMES ap_fixed.h
    HINTS
        $ENV{XILINX_HLS}/include
        $ENV{XILINX_VITIS}/include
        /mnt/data/xilinx/2025.2/Vitis/include
    NO_DEFAULT_PATH
    DOC "Vitis HLS include directory containing ap_fixed.h"
)

if(VA_HLS_INCLUDE_DIR)
    message(STATUS "VectorOP: Vitis HLS headers found at ${VA_HLS_INCLUDE_DIR}")
    set(_VA_DATA_TYPE_DEFAULT "ap_fixed<16,8>")
else()
    message(STATUS "VectorOP: Vitis HLS headers not found — defaulting to float")
    set(_VA_DATA_TYPE_DEFAULT "float")
endif()

# Domain parameters
set(VA_DATA_TYPE "${_VA_DATA_TYPE_DEFAULT}" CACHE STRING
    "Element data type: float, double, half, uint8_t, or ap_fixed<W,I>.")
set(VA_VECTOR_SIZE 1024 CACHE STRING
    "Default vector length shown in generated Config.h (informational).")

# ---------------------------------------------------------------------------
# Resolve data-type byte width
# ---------------------------------------------------------------------------
include(CheckTypeSize)
check_type_size(${VA_DATA_TYPE} _VA_TMP_WIDTH_${VA_DATA_TYPE})

if(NOT _VA_TMP_WIDTH_${VA_DATA_TYPE})
  if(VA_DATA_TYPE STREQUAL "half")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} 2)
  elseif(VA_DATA_TYPE STREQUAL "uint8_t")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} 1)
  elseif(VA_DATA_TYPE MATCHES "^ap_(u?)fixed<([0-9]+)")
    math(EXPR _bits  "${CMAKE_MATCH_2}")
    math(EXPR _bytes "(${_bits} + 7) / 8")
    set(_VA_TMP_WIDTH_${VA_DATA_TYPE} ${_bytes})
  else()
    message(FATAL_ERROR "Cannot determine size of data type '${VA_DATA_TYPE}'.")
  endif()
endif()

set(VA_DATA_WIDTH_BYTES "${_VA_TMP_WIDTH_${VA_DATA_TYPE}}")
math(EXPR VA_DATA_WIDTH_BITS "${VA_DATA_WIDTH_BYTES} * 8")

# Include directive for types that need a special header
if(VA_DATA_TYPE STREQUAL "half")
  set(VA_DATA_TYPE_INCLUDE "#include \"hls_half.h\"")
elseif(VA_DATA_TYPE MATCHES "^ap_(u?)fixed")
  set(VA_DATA_TYPE_INCLUDE "#include \"ap_fixed.h\"")
elseif(VA_DATA_TYPE STREQUAL "uint8_t")
  set(VA_DATA_TYPE_INCLUDE "#include <cstdint>")
else()
  set(VA_DATA_TYPE_INCLUDE "")
endif()

# ---------------------------------------------------------------------------
# Locate Vitis HLS (synthesis targets only; not required for C-sim).
# FindVitis.cmake is bundled in <repo-root>/cmake/.
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
find_package(Vitis)

# ---------------------------------------------------------------------------
# Generate configuration header
# ---------------------------------------------------------------------------
configure_file(include/Config.h.in Config.h)

# ---------------------------------------------------------------------------
# Simulation (C++ test) build
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VA_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernel/VectorOP.cpp)

add_library(vadd_kernel ${VA_KERNEL_SRC})
target_include_directories(vadd_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR})
if(VA_HLS_INCLUDE_DIR)
    target_include_directories(vadd_kernel SYSTEM PUBLIC ${VA_HLS_INCLUDE_DIR})
endif()
target_compile_options(vadd_kernel PRIVATE -Wall -Wextra -O2 -Wno-unknown-pragmas)

add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation PRIVATE vadd_kernel)
target_compile_options(TestSimulation PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME TestSimulation COMMAND TestSimulation)

message(STATUS "VectorOPKernel: Data_t=${VA_DATA_TYPE}")

# ---------------------------------------------------------------------------
# HLS synthesis flags (used for both Vitis and standalone HLS flows)
# ---------------------------------------------------------------------------
set(VA_SYNTHESIS_FLAGS "-O3 -DVA_SYNTHESIS")

# ---------------------------------------------------------------------------
# Vitis kernel + program targets (hw_emu / hw xclbin flow)
#
# Disabled by default — requires an installed Vitis platform (e.g. U250).
# Enable with:  cmake -DVA_ENABLE_VITIS_FLOW=ON -DVA_PLATFORM=<platform> …
# ---------------------------------------------------------------------------
option(VA_ENABLE_VITIS_FLOW
    "Enable Vitis hw/hw_emu xclbin targets (add_vitis_kernel/add_vitis_program). Requires an installed Vitis platform."
    OFF)

if(VA_ENABLE_VITIS_FLOW)
    add_vitis_kernel(VectorOP
                     KERNEL "VectorOPKernel"
                     FILES  ${VA_KERNEL_SRC}
                     HLS_FLAGS ${VA_SYNTHESIS_FLAGS}
                     INCLUDE_DIRS
                         ${CMAKE_CURRENT_SOURCE_DIR}/include
                         ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS
                         ${CMAKE_CURRENT_SOURCE_DIR}/include/VectorOP.h
                         ${CMAKE_CURRENT_BINARY_DIR}/Config.h)

    add_vitis_program(VectorOP ${VA_PLATFORM}
                      CLOCK ${VA_TARGET_CLOCK})
endif()

# ---------------------------------------------------------------------------
# Per-platform standalone HLS synthesis + IP export targets.
#
# Each JSON file in platforms/ defines one target.
# Required fields:  part
# Optional fields:  board, clock (default 300 MHz)
#
# Generated targets:
#   make synthesize_vectorop_<name>   — HLS synthesis + ip_catalog export
# ---------------------------------------------------------------------------
if(Vitis_HLS)
    message(STATUS "VectorOPKernel: HLS synthesis enabled via ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG}")

    string(REPLACE ";" " " _VA_KERNEL_SRC_STR "${VA_KERNEL_SRC}")

    set(_VA_PLATFORM_FILES "${AXI_PLATFORM_FILES}")
    foreach(_VA_PLATFORM_FILE IN LISTS _VA_PLATFORM_FILES)
      get_filename_component(_pname "${_VA_PLATFORM_FILE}" NAME_WE)
      file(READ "${_VA_PLATFORM_FILE}" _pjson)

      # Required: FPGA part string
      string(JSON _p_part ERROR_VARIABLE _e1 GET "${_pjson}" part)
      if(_e1)
        message(WARNING "VectorOPKernel platform '${_pname}': missing 'part' — skipping.")
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
        message(WARNING "VectorOPKernel platform '${_pname}': invalid axi_bus_width "
                        "'${_p_axi_bus_width}' — must be 32, 64, 128, 256, or 512. "
                        "Falling back to global AXI_BUS_WIDTH=$CACHE{AXI_BUS_WIDTH}.")
        set(_p_axi_bus_width "$CACHE{AXI_BUS_WIDTH}")
      endif()

      # Per-platform build directory and Config.h
      set(_p_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_pname}")
      file(MAKE_DIRECTORY "${_p_build_dir}")
      configure_file(include/Config.h.in "${_p_build_dir}/Config.h")

      # Build the set_part TCL command (with or without -board)
      if(_p_board)
        set(VA_BOARD_OPTION "set_part {${_p_part}} -board {${_p_board}}")
      else()
        set(VA_BOARD_OPTION "set_part {${_p_part}}")
      endif()

      # Variables consumed by Synthesis.tcl.in
      set(VA_HLS_PROJECT_NAME       "vadd_${_pname}")
      set(VA_PART_NAME              "${_p_part}")
      set(VA_SRC_SYNTHESIS          "${_VA_KERNEL_SRC_STR}")
      set(VA_SYNTHESIS_FLAGS_PLATFORM
          "${VA_SYNTHESIS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${_p_build_dir}")
      set(VA_TARGET_CLOCK_TCL       "${_p_clock}")
      set(VA_IP_OUTPUT_DIR          "${_p_build_dir}/ip_catalog")
      # Shadow the global AXI_BUS_WIDTH so configure_file picks up the per-platform value.
      set(AXI_BUS_WIDTH             "${_p_axi_bus_width}")

      configure_file(scripts/Synthesis.tcl.in
                     "${_p_build_dir}/synthesize_${_pname}.tcl")

      add_custom_target(
          "synthesize_vectorop_${_pname}"
          COMMAND ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG} "${_p_build_dir}/synthesize_${_pname}.tcl"
          WORKING_DIRECTORY "${_p_build_dir}"
          COMMENT "HLS synthesis: VectorOPKernel for '${_pname}' (${_p_part} @ ${_p_clock} MHz, ${_p_axi_bus_width}-bit AXI)"
          DEPENDS
              ${VA_KERNEL_SRC}
              ${CMAKE_CURRENT_SOURCE_DIR}/include/VectorOP.h
              "${_p_build_dir}/Config.h")

      message(STATUS "VectorOPKernel: synthesis target 'synthesize_vectorop_${_pname}' "
                     "(${_p_part} @ ${_p_clock} MHz, ${_p_axi_bus_width}-bit AXI)")
    endforeach()
else()
    message(STATUS "VectorOPKernel: vitis_hls / vitis-run not found — synthesis targets disabled")
endif()

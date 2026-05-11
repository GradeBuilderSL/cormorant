#pragma once

#include <cstdint>



// ---------------------------------------------------------------------------
// Element and accumulator types — configured by CMake.
//
// Default (Vitis HLS available):
//   Data_t    = ap_fixed<16,8>   range [-128, 127.996], 1/256 LSB
//   AccData_t = ap_fixed<32,16>  range [-32768, 32767.999], safe for K ≤ 2
//                                with max-value inputs; typical neural-net
//                                inputs (|v| ≤ 1) are safe for K ≤ 32768.
//
// Fallback (no Vitis HLS):
//   Data_t    = float
//   AccData_t = double           no overflow concern; no saturation at output
//
// AccData_t overflow budget for ap_fixed<16,8> inputs:
//   worst-case product  = 127.996^2 ≈ 16383
//   ap_fixed<32,16> max ≈ 32767
//   → safe for K ≤ 2 (worst case) / K ≤ ~200 (|values| ≤ 8 typical)
//   Increase to ap_fixed<40,24> via -DMM_ACC_DATA_TYPE for larger K budgets.
// ---------------------------------------------------------------------------
using Data_t    = float;
using AccData_t = float;

// ---------------------------------------------------------------------------
// Tile-size constants — compile-time parameters for the HLS kernel.
//
// kTileN  Row interleave depth.  Must be ≥ MAC latency (≈3 for ap_fixed<16,8>)
//         to achieve II=1 in the inner pipeline.  Must be a power of two so
//         that ki % kTileN compiles to a bitwise AND.
//
// kTileM  Parallel output columns per pipeline cycle.  Each column maps to
//         one DSP accumulator lane.  Must be a power of two (AXI burst
//         alignment, and ki % kTileM analogues if ever used).
//
// kTileK  On-chip B-buffer row count (BRAM K-slice).
//         Must be a power of two (Q6 in design doc: avoids divider in HLS).
//
// kMaxK   Compile-time upper bound on the inner dimension K.
//         Determines a_buf row length: a_buf[kTileN][kMaxK].
//         Models with K > kMaxK are rejected by the inference scheduler.
// ---------------------------------------------------------------------------
static constexpr unsigned kTileN = 4;
static constexpr unsigned kTileM = 16;
static constexpr unsigned kTileK = 256;
static constexpr unsigned kMaxK  = 2048;

static constexpr unsigned kSeed  = 42;

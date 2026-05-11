#pragma once

#include <cstdint>
#include "ap_fixed.h"
#define CONV_HAVE_APFIXED

// ---------------------------------------------------------------------------
// Element and accumulator types — configured by CMake.
//
// Default (Vitis HLS available):
//   Data_t    = ap_fixed<16,8>   range [-128, 127.996], 1/256 LSB
//   AccData_t = ap_fixed<32,16>  range [-32768, 32767.999]
//
// Fallback (no Vitis HLS):
//   Data_t    = float
//   AccData_t = float           (no saturation; reference only)
//
// AccData_t overflow for ap_fixed<16,8> inputs:
//   worst-case product  ≈ 128^2 = 16384
//   ap_fixed<32,16> max ≈ 32767
//   safe for C×kH×kW ≤ 2 (worst case) / ≤ ~32768 (|values| ≤ 1 typical)
//   Use ap_fixed<40,24> via CONV_ACC_DATA_TYPE for deeper reductions.
// ---------------------------------------------------------------------------
using Data_t    = ap_fixed<16,8>;
using AccData_t = ap_fixed<32,16>;

// ---------------------------------------------------------------------------
// Tile-size constants — compile-time parameters for the HLS kernel.
//
// kTileM   Output channel tile.  Must be a power of two so that
//          ri & (kTileM-1) compiles to a bitwise AND for the II=1 lane
//          rotation, and so that the dependency distance for acc[m1] equals
//          kTileM ≥ MAC latency (~3 cycles for ap_fixed<16,8>).
//
// kTileIC  Input channel tile.  Power of two; sets the w_buf and patch
//          buffer depth in the IC dimension.
//
// kMaxKH   Compile-time upper bound on kernel height.
// kMaxKW   Compile-time upper bound on kernel width.
//          Models with kH > kMaxKH or kW > kMaxKW are rejected by the
//          inference scheduler at validation time.
// ---------------------------------------------------------------------------
static constexpr unsigned kTileM  = 8;
static constexpr unsigned kTileIC = 16;
static constexpr unsigned kMaxKH  = 7;
static constexpr unsigned kMaxKW  = 7;

static constexpr unsigned kSeed   = 42;

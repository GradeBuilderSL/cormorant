#pragma once

#include <cstddef>
#include "ap_fixed.h"

// Element data type
using Data_t = ap_fixed<16,8>;

// Bit width of a single element (used for AXI stream TDATA sizing)
constexpr unsigned kDataWidthBits = 16;

// Seed used to initialise random inputs in TestSimulation
constexpr unsigned kSeed = 42;

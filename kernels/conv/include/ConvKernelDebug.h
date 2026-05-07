#pragma once

// ---------------------------------------------------------------------------
// ConvKernelDebug.h — C-simulation-only debug interface for ConvKernel.
//
// When ConvKernel.cpp is compiled with DEBUG_LOAD_DATA_CACHING (auto-enabled
// for non-synthesis builds), each input_patch_producer tracks the DDR
// addresses it reads.  Any address read more than once during a single
// ConvKernel invocation is counted here as a "duplicate read" — a regression
// in the on-chip line-buffer caching.
//
// Usage from a testbench:
//   conv_debug_reset_duplicate_reads();
//   ConvKernel(...);
//   if (conv_debug_duplicate_read_count() != 0) { /* fail */ }
//
// In synthesis builds these symbols still link (no-op / always 0) so the
// header is safe to include unconditionally.
// ---------------------------------------------------------------------------

void     conv_debug_reset_duplicate_reads();
unsigned conv_debug_duplicate_read_count();

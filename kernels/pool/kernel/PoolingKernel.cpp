// ---------------------------------------------------------------------------
// PoolingKernel.cpp — 2-D pooling kernel.
//
// Implements the ONNX MaxPool / AveragePool / LpPool operators (and their
// Global variants) in a channel-tiled structure that maps cleanly to Vitis
// HLS synthesis.
//
// Top-level dataflow (HLS DATAFLOW):
//
//   PoolingKernel
//     input_window_producer  ──window_pipe──►  process_pool_kernel_tile
//                            ──denom_pipe ──►          │
//                                                      └──acc_stream──► write_output_tile
//                                                                              │
//     x (DDR gmem0)                                                           ▼
//                                                                       y (DDR gmem1)
//
// Stage responsibilities:
//   * input_window_producer  — for each (ni, oh, ow) emits one denom value
//                              (= valid_count for AVG, or pool_h*pool_w when
//                              count_include_pad=1) and, for each channel
//                              tile ct, kTileC*pool_h*pool_w window pixels
//                              into window_pipe.  Out-of-bounds positions
//                              are filled with the pool-type identity so the
//                              consumer's reduce loop needs no bounds check.
//                              Channel-padding lanes (c_l >= c_valid) are
//                              also filled with the identity so the producer
//                              emits a fixed-size block per (ni, ct, oh, ow).
//   * process_pool_kernel_tile — drains kTileC*pool_h*pool_w pixels into
//                              win_buf, runs the II=1 flat-counter reduction,
//                              applies the post-reduction op (multiply by
//                              precomputed inv_denom for AVG, sqrt for LP-2)
//                              and pushes c_valid AccData_t results to
//                              acc_stream.
//   * write_output_tile      — saturates AccData_t → Data_t and writes to y
//                              with the channel-major (stride = out_h*out_w)
//                              addressing the original kernel used.
//
// II=1 reduction strategy (process_pool_kernel_tile):
//   The flat counter ri runs 0 .. pool_h*pool_w*kTileC - 1.
//   c1 = ri & (kTileC - 1)  — compile-time bitmask, no divider.
//   acc[c1] is written every kTileC cycles, satisfying the dependency
//   distance requirement (kTileC ≥ operation latency ≈ 1–3 cycles for
//   compare / add / multiply on ap_fixed<16,8>).
//
// Global pool support:
//   No special code.  Caller passes pool_h=in_h, pool_w=in_w, stride=1,
//   pad_top=pad_left=0.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include "hls_stream.h"

#include "PoolingKernel.h"
#include "PoolingKernelDebug.h"

// ---------------------------------------------------------------------------
// Debug-only DDR-read tracking.  Enabled automatically for C-simulation
// builds; disabled (zero-cost) under HLS synthesis.  Mirrors the pattern in
// kernels/conv/kernel/ConvKernel.cpp so the testbench can flag any cell
// read from DDR more than once per invocation.
// ---------------------------------------------------------------------------
#ifndef __SYNTHESIS__
#define DEBUG_LOAD_DATA_CACHING
#endif

#ifdef DEBUG_LOAD_DATA_CACHING
static unsigned g_pool_debug_duplicate_reads = 0;
#endif

void pool_debug_reset_duplicate_reads() {
#ifdef DEBUG_LOAD_DATA_CACHING
    g_pool_debug_duplicate_reads = 0;
#endif
}

unsigned pool_debug_duplicate_read_count() {
#ifdef DEBUG_LOAD_DATA_CACHING
    return g_pool_debug_duplicate_reads;
#else
    return 0;
#endif
}

#ifdef DEBUG_LOAD_DATA_CACHING
#include <cstddef>
#include <iostream>
#include <list>
#include <map>

// One record per DDR fetch — captures the loop-nest position so a duplicate
// dump can show *where* the kernel went back to the same address.
struct PoolReadCounters {
    unsigned ni;
    unsigned ct;
    unsigned c_l;
    unsigned oh;
    unsigned ow;
    unsigned khi;
    unsigned kwi;
};

inline std::ostream& operator<<(std::ostream& os, const PoolReadCounters& c) {
    os << "{ni=" << c.ni
       << " ct=" << c.ct
       << " c_l=" << c.c_l
       << " oh=" << c.oh
       << " ow=" << c.ow
       << " khi=" << c.khi
       << " kwi=" << c.kwi
       << "}";
    return os;
}

typedef std::map<std::size_t, std::list<PoolReadCounters>> PoolAddressMap_t;
#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// WindowLanes — kTileC parallel pixel lanes carried through window_pipe in a
// single FIFO transaction.  Letting the producer write one struct per cycle
// (instead of one Data_t per cycle) and the consumer read one struct per
// cycle drops the reduce loop from pool_h*pool_w*kTileC iterations down to
// pool_h*pool_w iterations.  HLS packs the array into a single FIFO word
// (kTileC * sizeof(Data_t) * 8 bits, e.g. 128 b for ap_fixed<16,8> × 8).
// Plain C array inside a POD struct keeps the type trivially copyable so
// hls::stream can copy it efficiently in C-sim and infer a wide FIFO in HW.
// ---------------------------------------------------------------------------
struct WindowLanes {
    Data_t lanes[kTileC];
};

// ---------------------------------------------------------------------------
// compute_ow_tile — runtime W-tile width.
//
// Returns the largest ow chunk size whose loaded input-column span fits in
// kMaxLineBufCols.  When in_w <= kMaxLineBufCols the result is out_w (single tile, no
// duplication — same as the zero-duplication path).  When the input is
// wider than the cache we split out_w into multiple tiles; adjacent tiles
// re-read overlapping boundary columns from DDR — the explicit relaxation
// that lets the kernel handle in_w > kMaxLineBufCols.
//
// Span(OW) = (OW - 1) * stride_w + (pool_w - 1) * dil_w + 1
// Solve Span(OW) <= kMaxLineBufCols for the largest OW.
// ---------------------------------------------------------------------------
static inline unsigned compute_ow_tile(
    unsigned out_w,
    unsigned pool_w,
    unsigned stride_w,
    unsigned dil_w
) {
    const unsigned win_w_span = (pool_w - 1) * dil_w + 1;
    unsigned ow_tile = 0;
    if (win_w_span <= kMaxLineBufCols && stride_w > 0) {
        ow_tile = (kMaxLineBufCols - win_w_span) / stride_w + 1;
    }
    if (ow_tile == 0) ow_tile = 1;
    if (ow_tile > out_w) ow_tile = out_w;
    return ow_tile;
}

// ---------------------------------------------------------------------------
// row_loader — DATAFLOW source (Phase 1).
//
// Reads input rows from DDR and pushes them onto row_data_pipe in the order
// the window_emitter consumes them.  Owns no on-chip buffer: it's a thin DDR
// reader that lets the window_emitter run concurrently with row fetching
// (Phase 1 of oh+1 overlaps Phase 2 of oh).
//
// Loop nest is (ni, ct, owt, oh, ih, c_l, iw) with the SAME load_start /
// load_end / iw_load_lo / iw_load_hi schedule the window_emitter mirrors.
// Both functions derive these values from the geometry parameters
// independently — no metadata stream is needed because per-oh row counts
// are deterministic.
// ---------------------------------------------------------------------------
static void row_loader(
    const Data_t*        x,
    hls::stream<Data_t>& row_data_pipe,
    unsigned             batch,
    unsigned             channels,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             pool_h,
    unsigned             pool_w,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             pad_top,
    unsigned             pad_left,
    unsigned             dil_h,
    unsigned             dil_w,
    unsigned             ow_tile
) {
    const unsigned c_tiles    = (channels + kTileC - 1) / kTileC;
    const unsigned in_hw      = in_h * in_w;
    const unsigned ow_tiles_w =
        (ow_tile > 0) ? ((out_w + ow_tile - 1) / ow_tile) : 1u;

#ifdef DEBUG_LOAD_DATA_CACHING
    // Local to one kernel invocation — collects every DDR cell read.
    // Walked at function exit; any address with more than one record bumps
    // the global duplicate counter.  When in_w > kMaxLineBufCols, W-tile
    // boundary columns appear multiple times — the documented relaxation.
    PoolAddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                const int iw_start =
                    (int)(ow_lo * stride_w) - (int)pad_left;
                const int iw_end =
                    (int)((ow_hi - 1) * stride_w + (pool_w - 1) * dil_w)
                  - (int)pad_left;
                const int iw_load_lo = (iw_start < 0) ? 0 : iw_start;
                const int iw_load_hi = (iw_end >= (int)in_w)
                                     ? (int)in_w - 1 : iw_end;

                int last_loaded_row = -1;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    const int ih_window_max = (int)(oh * stride_h)
                                            - (int)pad_top
                                            + (int)((pool_h - 1) * dil_h);
                    int load_start = last_loaded_row + 1;
                    if (load_start < 0) load_start = 0;
                    int load_end = ih_window_max;
                    if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                    for (int ih = load_start; ih <= load_end; ih++) {
                        for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                            const unsigned c     = c_off + c_l;
                            const unsigned x_row = (ni * channels + c) * in_hw
                                                 + (unsigned)ih * in_w;
                            for (int iw = iw_load_lo; iw <= iw_load_hi; iw++) {
                                #pragma HLS PIPELINE II=1
                                const std::size_t addr =
                                    (std::size_t)x_row + (unsigned)iw;
                                row_data_pipe.write(x[addr]);

#ifdef DEBUG_LOAD_DATA_CACHING
                                PoolReadCounters c_rc;
                                c_rc.ni  = ni;
                                c_rc.ct  = ct;
                                c_rc.c_l = c_l;
                                c_rc.oh  = oh;
                                c_rc.ow  = owt;
                                c_rc.khi = (unsigned)ih;
                                c_rc.kwi = (unsigned)iw;
                                read_addresses[addr].push_back(c_rc);
#endif /* DEBUG_LOAD_DATA_CACHING */
                            }
                        }
                    }
                    if (load_end > last_loaded_row) {
                        last_loaded_row = load_end;
                    }
                }
            }
        }
    }

#ifdef DEBUG_LOAD_DATA_CACHING
    for (const auto& it : read_addresses) {
        if (it.second.size() > 1) {
            ++g_pool_debug_duplicate_reads;
            std::cout << it.first << " --> " << std::endl;
            for (const auto& l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}

// ---------------------------------------------------------------------------
// window_emitter — DATAFLOW stage (Phase 2).
//
// Owns line_buf.  Drains row_data_pipe into line_buf, then emits one
// WindowLanes vector per (khi, kwi) into window_pipe and one denom value
// per output position into denom_pipe.  Mirrors the row_loader's
// (ni, ct, owt, oh) schedule so stream consumption matches production.
//
// valid_count (denom) is computed via a fully-unrolled kMaxPoolH × kMaxPoolW
// pass — collapses to a parallel adder tree (~1 cycle) instead of the
// previous sequential pool_h*pool_w-cycle loop.
//
// Constraints (validated by the inference scheduler):
//   (pool_w - 1) * dil_w + 1 <= kMaxLineBufCols   (single window fits)
//   (pool_h - 1) * dil_h + 1 <= kMaxLineBufRows   (vertical span fits)
//   kMaxLineBufRows is a power of two (slot = ih & (kMaxLineBufRows-1)).
// ---------------------------------------------------------------------------
static void window_emitter(
    hls::stream<Data_t>&      row_data_pipe,
    hls::stream<WindowLanes>& window_pipe,
    hls::stream<unsigned>&    denom_pipe,
    unsigned                  batch,
    unsigned                  channels,
    unsigned                  in_h,
    unsigned                  in_w,
    unsigned                  out_h,
    unsigned                  out_w,
    unsigned                  pool_h,
    unsigned                  pool_w,
    unsigned                  stride_h,
    unsigned                  stride_w,
    unsigned                  pad_top,
    unsigned                  pad_left,
    unsigned                  dil_h,
    unsigned                  dil_w,
    unsigned                  pool_type,
    unsigned                  count_include_pad,
    unsigned                  ow_tile
) {
    const unsigned c_tiles    = (channels + kTileC - 1) / kTileC;
    const unsigned ow_tiles_w =
        (ow_tile > 0) ? ((out_w + ow_tile - 1) / ow_tile) : 1u;

    // line_buf:  kTileC * kMaxLineBufRows * kMaxLineBufCols * sizeof(Data_t)
    //         =      8  *       16        *       64        *      2     =  16 KB
    // ARRAY_PARTITION dim=1 complete → kTileC independent BRAMs of
    // [kMaxLineBufRows][kMaxLineBufCols], one per channel lane — required
    // for the vectorised Phase 2 emit below.
    static Data_t line_buf[kTileC][kMaxLineBufRows][kMaxLineBufCols];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    const Data_t pad_val = (pool_type == kPoolMax)
        ? Data_t(kDataMin)
        : Data_t(0);

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                const int iw_start =
                    (int)(ow_lo * stride_w) - (int)pad_left;
                const int iw_end =
                    (int)((ow_hi - 1) * stride_w + (pool_w - 1) * dil_w)
                  - (int)pad_left;
                const int iw_load_lo = (iw_start < 0) ? 0 : iw_start;
                const int iw_load_hi = (iw_end >= (int)in_w)
                                     ? (int)in_w - 1 : iw_end;

                int last_loaded_row = -1;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    const int ih_window_max = (int)(oh * stride_h)
                                            - (int)pad_top
                                            + (int)((pool_h - 1) * dil_h);

                    // -----------------------------------------------
                    // Phase 1 drain: pull row pixels from row_data_pipe
                    // into line_buf using the same (ih, c_l, iw) order
                    // the row_loader emits them.
                    // -----------------------------------------------
                    int load_start = last_loaded_row + 1;
                    if (load_start < 0) load_start = 0;
                    int load_end = ih_window_max;
                    if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                    for (int ih = load_start; ih <= load_end; ih++) {
                        const unsigned slot = (unsigned)ih & (kMaxLineBufRows - 1);
                        for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                            for (int iw = iw_load_lo; iw <= iw_load_hi; iw++) {
                                #pragma HLS PIPELINE II=1
                                const unsigned local_iw =
                                    (unsigned)(iw - iw_load_lo);
                                line_buf[c_l][slot][local_iw] =
                                    row_data_pipe.read();
                            }
                        }
                    }
                    if (load_end > last_loaded_row) {
                        last_loaded_row = load_end;
                    }

                    for (unsigned ow = ow_lo; ow < ow_hi; ow++) {

                        // -----------------------------------------------
                        // valid_count via fully-unrolled adder tree.
                        // kMaxPoolH × kMaxPoolW = 49 conditional 1-bit
                        // increments collapse to a ~3-4 level adder tree;
                        // entries with khi >= pool_h or kwi >= pool_w
                        // contribute 0 by construction.
                        // -----------------------------------------------
                        unsigned valid_count = 0;
                        for (unsigned khi = 0; khi < kMaxPoolH; khi++) {
                            #pragma HLS UNROLL
                            for (unsigned kwi = 0; kwi < kMaxPoolW; kwi++) {
                                #pragma HLS UNROLL
                                if (khi < pool_h && kwi < pool_w) {
                                    const int ih_v = (int)(oh * stride_h + khi * dil_h)
                                                   - (int)pad_top;
                                    const int iw_v = (int)(ow * stride_w + kwi * dil_w)
                                                   - (int)pad_left;
                                    if (ih_v >= 0 && (unsigned)ih_v < in_h &&
                                        iw_v >= 0 && (unsigned)iw_v < in_w)
                                        valid_count++;
                                }
                            }
                        }
                        const unsigned denom_u = count_include_pad
                            ? (pool_h * pool_w)
                            : valid_count;
                        denom_pipe.write(denom_u);

                        // -----------------------------------------------
                        // Phase 2 emit: pool_h*pool_w vectorised window
                        // entries.  Each WindowLanes carries kTileC
                        // channel lanes in parallel.
                        // -----------------------------------------------
                        for (unsigned khi = 0; khi < pool_h; khi++) {
                            const int ih =
                                (int)(oh * stride_h + khi * dil_h) - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned slot = ih_ok
                                ? ((unsigned)ih & (kMaxLineBufRows - 1))
                                : 0u;

                            for (unsigned kwi = 0; kwi < pool_w; kwi++) {
                                #pragma HLS PIPELINE II=1
                                const int iw =
                                    (int)(ow * stride_w + kwi * dil_w) - (int)pad_left;
                                const bool spatial_ok =
                                    ih_ok && iw >= 0 && (unsigned)iw < in_w;
                                const unsigned local_iw = spatial_ok
                                    ? (unsigned)(iw - iw_load_lo)
                                    : 0u;

                                WindowLanes v;
                                for (unsigned c_l = 0; c_l < kTileC; c_l++) {
                                    #pragma HLS UNROLL
                                    const bool valid = spatial_ok && (c_l < c_valid);
                                    v.lanes[c_l] = valid
                                        ? line_buf[c_l][slot][local_iw]
                                        : pad_val;
                                }
                                window_pipe.write(v);
                            }
                        }
                    } // ow loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// process_pool_kernel_tile — DATAFLOW processor.
//
// Loop nest matches the producer: (ni, ct, owt, oh, ow) with ct OUTER of
// oh AND a W-tile dimension owt OUTER of oh.  ow_tile = out_w yields a
// single tile and degenerates to (ni, ct, oh, ow); ow_tile < out_w
// processes ow chunks in turn so the producer's line buffer stays bounded.
//
// Per (ni, ct, owt, oh, ow):
//   * Read denom_u from denom_pipe; precompute inv_denom = 1/denom_u
//     (multiply beats divide).
//   * Fused drain + reduce: a single II=1 loop reads kTileC*pool_h*pool_w
//     pixels directly from window_pipe and folds them into the per-lane
//     accumulators acc[kTileC].  ri counts 0..pool_h*pool_w*kTileC-1 with
//     c1 = ri & (kTileC-1) cycling fastest — matching the producer's
//     (khi, kwi, c_l) emit order.  acc[c1] is written every kTileC cycles
//     so the dependency distance (= kTileC) covers the MAX/AVG/LP latency
//     of ap_fixed<32,16> at 300 MHz.
//   * Finalise: multiply by inv_denom for AVG, sqrt for LP-2, identity
//     otherwise; push c_valid lanes to acc_stream as AccData_t.  The
//     writer saturates AccData_t → Data_t at the boundary.
//
// Fusion eliminates the previous separate drain (window_pipe → win_buf)
// loop, halving the consumer's per-output cycle count and balancing it
// against the producer.  win_buf is gone; only acc[kTileC] survives.
// ---------------------------------------------------------------------------
static void process_pool_kernel_tile(
    hls::stream<WindowLanes>& window_pipe,
    hls::stream<unsigned>&    denom_pipe,
    hls::stream<AccData_t>&   acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                pool_h,
    unsigned                pool_w,
    unsigned                pool_type,
    unsigned                lp_order,
    unsigned                ow_tile
) {
    AccData_t acc[kTileC];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    const unsigned c_tiles    = (channels + kTileC - 1) / kTileC;
    const unsigned ow_tiles_w =
        (ow_tile > 0) ? ((out_w + ow_tile - 1) / ow_tile) : 1u;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = ow_lo; ow < ow_hi; ow++) {

                    const unsigned denom_u   = denom_pipe.read();
                    const float    inv_denom =
                        (denom_u > 0u) ? 1.0f / (float)denom_u : 0.0f;

                    // ---------------------------------------------------
                    // Initialise accumulators (1 cycle, fully unrolled).
                    //   MAX: kAccMin sentinel (any valid input beats it
                    //        on the first comparison).
                    //   AVG / LP: 0 (sum starts at zero).
                    // ---------------------------------------------------
                    for (unsigned c1 = 0; c1 < kTileC; c1++) {
                        #pragma HLS UNROLL
                        acc[c1] = (pool_type == kPoolMax)
                            ? AccData_t(kAccMin)
                            : AccData_t(0);
                    }

                    // ---------------------------------------------------
                    // Vectorised reduce: one WindowLanes struct per cycle
                    // updates all kTileC accumulator lanes in parallel.
                    //
                    // ri runs 0 .. pool_h*pool_w - 1 (no kTileC factor —
                    // that's what the vectorisation buys us).  All kTileC
                    // lane updates happen on the same cycle, fully unrolled.
                    //
                    // Per-lane RAW distance on acc[c1] is 1 cycle, so HLS
                    // schedules the reduce loop at II=1 for MAX (compare
                    // is single-cycle) and at II=L for AVG/LP where L is
                    // the ap_fixed<32,16> add (~2-3 cyc on DSP).  Either
                    // way the cycle count drops by ~kTileC over the prior
                    // scalar fused reduce.
                    // ---------------------------------------------------
                    const unsigned ri_bound = pool_h * pool_w;
                    for (unsigned ri = 0; ri < ri_bound; ri++) {
                        #pragma HLS PIPELINE II=1
                        const WindowLanes v = window_pipe.read();
                        for (unsigned c1 = 0; c1 < kTileC; c1++) {
                            #pragma HLS UNROLL
                            const AccData_t val = AccData_t(v.lanes[c1]);

                            if (pool_type == kPoolMax) {
                                if (val > acc[c1]) acc[c1] = val;
                            } else if (pool_type == kPoolAvg) {
                                acc[c1] += val;
                            } else {
                                // LP: p=1 → |val|,  p=2 → val²
                                const AccData_t contrib = (lp_order == 1u)
                                    ? (val < AccData_t(0) ? AccData_t(-val) : val)
                                    : AccData_t(val * val);
                                acc[c1] += contrib;
                            }
                        }
                    }

                    // ---------------------------------------------------
                    // Finalise and push c_valid lanes to acc_stream.
                    //
                    //   MAX: identity — acc is already the max value.
                    //   AVG: multiply by precomputed reciprocal denominator.
                    //   LP p=1: identity — acc is already Σ|x_i|.
                    //   LP p=2: sqrt(acc) via float (DSP-friendly in HLS).
                    // ---------------------------------------------------
                    for (unsigned c1 = 0; c1 < c_valid; c1++) {
                        #pragma HLS PIPELINE II=1
                        AccData_t result;
                        if (pool_type == kPoolMax) {
                            result = acc[c1];
                        } else if (pool_type == kPoolAvg) {
                            result = AccData_t((float)acc[c1] * inv_denom);
                        } else {
                            result = (lp_order == 1u)
                                ? acc[c1]
                                : AccData_t(sqrtf((float)acc[c1]));
                        }
                        acc_stream.write(result);
                    }

                    } // ow loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// write_output_tile — DATAFLOW sink.
//
// Drains acc_stream in the consumer's emit order — (ni, ct, owt, oh, ow, c1)
// with c1 cycling 0..c_valid-1.  Output addresses are non-contiguous in C
// (stride = out_h*out_w per channel); y_addr is advanced by a counter to
// avoid a multiplier inside the pipeline.  Saturates AccData_t → Data_t
// at the boundary.  ow_tile = out_w degenerates to (ni, ct, oh, ow, c1).
// ---------------------------------------------------------------------------
static void write_output_tile(
    Data_t*                 y,
    hls::stream<AccData_t>& acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                ow_tile
) {
    const unsigned c_tiles    = (channels + kTileC - 1) / kTileC;
    const unsigned hw_stride  = out_h * out_w;
    const unsigned ow_tiles_w =
        (ow_tile > 0) ? ((out_w + ow_tile - 1) / ow_tile) : 1u;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);
            const unsigned y_base  = (ni * channels + c_off) * hw_stride;

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = ow_lo; ow < ow_hi; ow++) {
                        unsigned y_addr = y_base + oh * out_w + ow;
                        for (unsigned c1 = 0; c1 < c_valid; c1++) {
                            #pragma HLS PIPELINE II=1
                            y[y_addr] = saturate_cast<Data_t>(acc_stream.read());
                            y_addr += hw_stride;
                        }
                    } // ow loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

void PoolingKernel(
    const Data_t* x,
    Data_t*       y,
    unsigned      batch,
    unsigned      channels,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      pool_h,
    unsigned      pool_w,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      dil_h,
    unsigned      dil_w,
    unsigned      pool_type,
    unsigned      lp_order,
    unsigned      count_include_pad
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    //
    // Two m_axi ports: gmem0 for the read-only input, gmem1 for the write-
    // only output.  All scalar arguments go into the s_axilite ctrl register
    // file accessed by the PS driver.
    // -----------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=x  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=y  offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=x                 bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=y                 bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=channels          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_h             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_w             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_h            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_w            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_h          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_w          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_top           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_left          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dil_h             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dil_w             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_type         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=lp_order          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=count_include_pad bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return            bundle=ctrl

    // -----------------------------------------------------------------------
    // STABLE: read-only m_axi base pointer x and every s_axilite scalar are
    // latched at ap_start and never written during the DATAFLOW region's
    // execution.  Marking them STABLE tells HLS not to insert auto-generated
    // synchronization stages or fan-out FIFOs into the producers — see the
    // ConvKernel.cpp note for the full rationale.  `y` is intentionally NOT
    // listed because write_output_tile writes through it during the dataflow
    // region.
    // -----------------------------------------------------------------------
    #pragma HLS STABLE variable=x
    #pragma HLS STABLE variable=batch
    #pragma HLS STABLE variable=channels
    #pragma HLS STABLE variable=in_h
    #pragma HLS STABLE variable=in_w
    #pragma HLS STABLE variable=out_h
    #pragma HLS STABLE variable=out_w
    #pragma HLS STABLE variable=pool_h
    #pragma HLS STABLE variable=pool_w
    #pragma HLS STABLE variable=stride_h
    #pragma HLS STABLE variable=stride_w
    #pragma HLS STABLE variable=pad_top
    #pragma HLS STABLE variable=pad_left
    #pragma HLS STABLE variable=dil_h
    #pragma HLS STABLE variable=dil_w
    #pragma HLS STABLE variable=pool_type
    #pragma HLS STABLE variable=lp_order
    #pragma HLS STABLE variable=count_include_pad

    // -----------------------------------------------------------------------
    // Top-level DATAFLOW region.
    //
    //   window_pipe  carries kTileC * pool_h * pool_w pixels per output
    //                position; depth covers one full window block so the
    //                producer can stage the next tile while the consumer
    //                is reducing.
    //   denom_pipe   one entry per output position — small FIFO is enough.
    //   acc_stream   c_valid AccData_t entries per (ni, ct, owt, oh, ow);
    //                depth kTileC matches the writer's per-tile drain burst.
    //
    // ow_tile is the W-tile chunk size (single source of truth — passed to
    // all three stages so they iterate (ni, ct, owt, oh, ow) in lockstep).
    // When in_w <= kMaxLineBufCols the formula yields ow_tile = out_w and the
    // owt loop runs once: the kernel matches the zero-duplication path.
    // When in_w > kMaxLineBufCols, ow_tile < out_w and boundary input columns are
    // re-read once per W-tile transition — the documented relaxation.
    // -----------------------------------------------------------------------
    const unsigned ow_tile = compute_ow_tile(out_w, pool_w, stride_w, dil_w);

    #pragma HLS DATAFLOW

    // row_data_pipe carries raw input pixels from row_loader (DDR) to
    // window_emitter (line_buf cache).  Depth holds enough rows for the
    // window_emitter to lag a full Phase-1 row load behind the row_loader,
    // so Phase 1 of oh+1 overlaps Phase 2 of oh.  kTileC * kMaxLineBufRows
    // * kMaxLineBufCols (= 16 KB / 8 lanes * 64 cols ≈ 8192 entries at
    // depth, but FIFO stores a single Data_t per slot so HLS will use
    // BRAM/LUTRAM as appropriate).
    hls::stream<Data_t> row_data_pipe;
    #pragma HLS STREAM variable=row_data_pipe depth=kTileC*kMaxLineBufCols*4

    // window_pipe carries one WindowLanes struct (kTileC scalar lanes packed)
    // per (khi, kwi) — pool_h*pool_w writes per output position.  Depth holds
    // a full window block so the producer can stage the next tile while the
    // consumer is still reducing the current one.
    hls::stream<WindowLanes> window_pipe;
    #pragma HLS STREAM variable=window_pipe depth=kMaxPoolH*kMaxPoolW

    hls::stream<unsigned> denom_pipe;
    #pragma HLS STREAM variable=denom_pipe depth=4

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileC

    row_loader(
        x, row_data_pipe,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w,
        pad_top, pad_left, dil_h, dil_w,
        ow_tile);

    window_emitter(
        row_data_pipe, window_pipe, denom_pipe,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w,
        pad_top, pad_left, dil_h, dil_w,
        pool_type, count_include_pad, ow_tile);

    process_pool_kernel_tile(
        window_pipe, denom_pipe, acc_stream,
        batch, channels, out_h, out_w,
        pool_h, pool_w, pool_type, lp_order, ow_tile);

    write_output_tile(
        y, acc_stream,
        batch, channels, out_h, out_w, ow_tile);
}

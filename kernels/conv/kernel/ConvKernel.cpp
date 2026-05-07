// ---------------------------------------------------------------------------
// ConvKernel.cpp — 2-D convolution kernel.
//
// Implements the ONNX Conv operator (group=1 or group=in_ch) in a tiled
// structure that maps cleanly to Vitis HLS synthesis.
// See doc/CONV_PLAN.md for a full explanation of the architecture, tiling
// strategy, and II=1 rationale.
//
// Top-level dataflow (HLS DATAFLOW):
//
//   ConvKernel
//     bias_producer           ──bias_stream──►  process_conv_kernel_tile
//     input_patch_producer    ──patch_stream──►        │
//                                                      └──acc_stream──► write_output_tile
//                                                                              │
//     x/bias (DDR gmem0/2)                                                     ▼
//                                                                       y (DDR gmem3)
//
// mt-hoist (the m_tile loop is INSIDE the spatial nest):
//   * input_patch_producer_standard reads each unique x[] pixel from DDR
//     exactly once per (ni, oh, ow), buffers the patch on-chip, then
//     broadcasts it m_tiles times into patch_stream — eliminating the
//     m_tiles× DDR re-read of x.
//   * bias_producer loads the entire bias vector once into bias_buf and
//     streams it in (r, mt, m1) order to match the consumer's new nest.
//   * process_conv_kernel_tile iterates (ni, oh, ow, mt) — same patch is
//     consumed by every output-channel tile back-to-back.
//   * write_output_tile drains acc_stream in (ni, oh, ow, mt, m1) order.
//
// Loop structure — standard conv (is_depthwise=0):
//
//   batch / oh / ow loops    — iterate over output spatial positions
//     m_tile loop            — tiles the output channel dimension (TILE_M)
//       compute tile (producer):
//         [acc init]   — zero acc[], overlay m_valid lanes from bias_stream
//         ic_tile loop — tiles the input channel dimension (TILE_IC)
//           load patch  — current (oh,ow) input patch to on-chip BRAM
//           load w_buf  — weight tile for (m_tile, ic_tile) to on-chip BRAM
//           accumulate  — II=1 K-reduction over ic×kH×kW with TILE_M lanes
//         drain        — push m_valid lanes of acc[] to acc_stream
//       write output (consumer):
//         saturate_cast acc_stream → DDR y
//
// Loop structure — depthwise conv (is_depthwise=1): same outer nest, the
// per-tile compute swaps in compute_depthwise_conv_tile (no ic_tile loop,
// per-lane patch and weight loads, kH×kW reduction).
//
// II=1 strategy (§2.4):
//   The flat counter ri runs 0 .. ic_valid*kh*kw*TILE_M-1 (standard) or
//   0 .. kh*kw*TILE_M-1 (depthwise).
//   m1 = ri & (TILE_M - 1)  — compile-time bitmask, no divider.
//   acc[m1] is written every TILE_M cycles, satisfying the dependency distance
//   requirement (TILE_M ≥ ap_fixed<16,8> MAC latency ≈ 3 cycles).
//
// Depthwise weight layout: weight[out_ch][1][kh][kw] (no in_ch dimension).
//   Offset for channel m: m * kh * kw + khi * kw + kwi
// ---------------------------------------------------------------------------

#include <algorithm>
#include "hls_stream.h"

#include "ConvKernel.h"
#include "ConvKernelDebug.h"

#ifndef __SYNTHESIS__
#define DEBUG_LOAD_DATA_CACHING
#endif

#ifdef DEBUG_LOAD_DATA_CACHING
static unsigned g_conv_debug_duplicate_reads = 0;
#endif

void conv_debug_reset_duplicate_reads() {
#ifdef DEBUG_LOAD_DATA_CACHING
    g_conv_debug_duplicate_reads = 0;
#endif
}

unsigned conv_debug_duplicate_read_count() {
#ifdef DEBUG_LOAD_DATA_CACHING
    return g_conv_debug_duplicate_reads;
#else
    return 0;
#endif
}

#ifdef DEBUG_LOAD_DATA_CACHING
#include <map>
#include <list>
#include <iostream>

struct CycleCounters {
    unsigned mt;
    unsigned ni;
    unsigned oh;
    unsigned ow;
    unsigned ict;
    unsigned ic_l;
    unsigned khi;
    unsigned kwi;
};

inline std::ostream& operator<<(std::ostream& os, const CycleCounters& c) {
    os << "{mt=" << c.mt
       << " ni=" << c.ni
       << " oh=" << c.oh
       << " ow=" << c.ow
       << " ict=" << c.ict
       << " ic_l=" << c.ic_l
       << " khi=" << c.khi
       << " kwi=" << c.kwi
       << "}";
    return os;
}

typedef std::map<size_t, std::list<CycleCounters>> AddressMap_t;

// Weight DDR reads are tracked across one ConvKernel invocation by the
// weight_producer; the file-scope map is reset on entry and dumped on exit
// (after duplicates increment g_conv_debug_duplicate_reads).  ConvKernel
// runs serially so a single static is safe.
static AddressMap_t g_weight_read_addresses;

#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// Standard: II=1 K-reduction that consumes weights from a stream, fused
// with the MAC.  Identical schedule to accumulate_standard, but the weight
// arrives one value per cycle from weight_stream (in the order the lane
// rotation needs: m1 cycles fastest, then kwi, khi, ic_l).  This removes
// the separate "drain weight_stream into w_buf, then accumulate" two-pass
// pattern in process_conv_kernel_tile that was halving inner-loop
// throughput.
//
// ri runs 0 .. kTileIC * kh * kw * kTileM - 1 — the FULL tile, including
// padding lanes ic_l>=ic_valid and m1>=m_valid.  The producer zero-pads
// those weights so the MACs into padding lanes are no-ops; reading the
// padding from the stream keeps producer/consumer counts in lockstep.
// ---------------------------------------------------------------------------
static void accumulate_standard_streamed(
    const Data_t         patch[kTileIC][kMaxKH][kMaxKW],
    hls::stream<Data_t>& weight_stream,
    AccData_t            acc[kTileM],
    unsigned             kh,
    unsigned             kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0, ic_cnt = 0;
    const unsigned ri_bound = kTileIC * kh * kw * kTileM;
    for (unsigned ri = 0; ri < ri_bound; ri++) {
        #pragma HLS PIPELINE II=1
        const unsigned m1 = ri & (kTileM - 1);
        const Data_t   w  = weight_stream.read();
        acc[m1] +=
            AccData_t(patch[ic_cnt][khi_cnt][kwi_cnt]) *
            AccData_t(w);

        if ((ri & (kTileM - 1)) == kTileM - 1) {
            if (++kwi_cnt == kw) {
                kwi_cnt = 0;
                if (++khi_cnt == kh) {
                    khi_cnt = 0;
                    ++ic_cnt;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Depthwise: II=1 kH×kW reduction with kTileM lanes.
//
// ri runs 0 .. kh*kw*kTileM - 1.  m1 = ri & (kTileM - 1) cycles through
// lanes; acc[m1] is written every kTileM cycles, so the RAW dependence
// distance ≥ MAC latency.
// ---------------------------------------------------------------------------
//#include <iostream>
static void accumulate_depthwise(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kMaxKH][kMaxKW],
    AccData_t    acc[kTileM],
    unsigned     kh,
    unsigned     kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0;
    const unsigned ri_bound_dw = kh * kw * kTileM;
    for (unsigned ri = 0; ri < ri_bound_dw; ri++) {
        #pragma HLS PIPELINE II=1
        const unsigned m1 = ri & (kTileM - 1);
        //std::cout << "m1=" << m1 << " khi_cnt=" << khi_cnt << " kwi_cnt=" << kwi_cnt << std::endl;
        acc[m1] +=
            AccData_t(patch[m1][khi_cnt][kwi_cnt]) *
            AccData_t(w_buf[m1][khi_cnt][kwi_cnt]);

        if ((ri & (kTileM - 1)) == kTileM - 1) {
            if (++kwi_cnt == kw) {
                kwi_cnt = 0;
                ++khi_cnt;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Write outputs to DDR.
//
// Loop nest matches the new consumer's drain order — (ni, oh, ow, mt, m1).
// For each (ni, oh, ow) the m_tiles × m_valid lanes are written in
// channel-major order; y_addr advances by ohw between m1 lanes.
// ---------------------------------------------------------------------------
static void write_output_tile(
    Data_t*                 y,
    hls::stream<AccData_t>& acc_stream,
    unsigned                out_ch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                batch
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;
    const unsigned ohw     = out_h * out_w;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                const unsigned base = ni * out_ch * ohw + oh * out_w + ow;
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    unsigned       y_addr  = base + m_off * ohw;
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        y[y_addr] = saturate_cast<Data_t>(acc_stream.read());
                        y_addr += ohw;
                    }
                } // m_tile loop
            } // ow loop
        } // oh loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// Bias producer (DATAFLOW source).
//
// Streams bias values to process_conv_kernel_tile in the order the consumer
// reads them after the mt-hoist optimisation:
//
//     for (r in batch*out_h*out_w):
//         for (mt in m_tiles):
//             for (m1 in m_valid): write(bias[m_off+m1])
//
// Because the consumer's mt loop now lives INSIDE the spatial nest, the
// producer needs all m_tiles' bias slices available simultaneously per
// `r` iteration.  The entire bias vector is therefore loaded once into
// bias_buf[kMaxOutCh] and replayed `reps` times.  When has_bias=0 no DDR
// transactions are issued and the producer pushes AccData_t(0) directly.
// ---------------------------------------------------------------------------
static void bias_producer(
    const Data_t*           bias,
    hls::stream<AccData_t>& bias_stream,
    unsigned                out_ch,
    unsigned                reps,
    unsigned                has_bias
) {
    Data_t bias_buf[kMaxOutCh];

    if (has_bias) {
        for (unsigned m = 0; m < out_ch; m++) {
            #pragma HLS PIPELINE II=1
            bias_buf[m] = bias[m];
        }
    }

    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    for (unsigned r = 0; r < reps; r++) {
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                #pragma HLS PIPELINE II=1
                bias_stream.write(has_bias
                                  ? AccData_t(bias_buf[m_off + m1])
                                  : AccData_t(0));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Standard-conv input patch ASSEMBLER (DATAFLOW source).
//
// Tiled-IC version (Option-A): the producer iterates
//
//     for ni: for ict: for oh: for ow: for ic_l, kh, kw
//
// with ict OUTER of oh.  This lets line_buf shrink from
//   [kMaxInCh][rows][kMaxInW]  →  [kTileIC][rows][kMaxInW]
// while keeping the "x read once per (ni, c)" invariant — line_buf is
// reused across the oh loop within the same ic-tile (slid forward by
// the load_start/last_loaded_row tracker).  When ict advances the line
// buffer is overwritten with the next ic-tile's channels; we never
// revisit a previous ict, so no cached row is ever needed twice.
//
// Per (ni, ict, oh):
//   Phase 1 — load any new input rows for the current (oh, ict) window.
//             kTileIC channels at offset ic_off..ic_off+ic_valid-1.
// Per (ni, ict, oh, ow):
//   Phase 2 — stream one ic-tile's worth of patch values
//             (ic_valid × kh × kw entries) into patch_pipe.
//
// Constraints: in_w <= kMaxInW,
//              (kh-1)*dilation_h + 1 <= kMaxLineBufRows.
// ---------------------------------------------------------------------------
static void input_patch_producer_standard(
    const Data_t*        x,
    hls::stream<Data_t>& patch_stream,
    unsigned             num_groups,
    unsigned             group_size,
    unsigned             batch,
    unsigned             in_ch,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_ch,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             kh,
    unsigned             kw,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             dilation_h,
    unsigned             dilation_w,
    unsigned             pad_top,
    unsigned             pad_left
) {
    (void)out_ch;  // m_tiles is computed in the broadcaster; not needed here.

    const unsigned ic_tiles = (in_ch + kTileIC - 1) / kTileIC;
    const unsigned in_hw    = in_h * in_w;

    Data_t line_buf[kTileIC][kMaxLineBufRows][kMaxInW];

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    // Loop order: g OUTER of ict OUTER of ni — matches the consumer's
    // (g, ict, ni, oh, ow, mt) iteration so weights stream once per
    // (g, ict).  line_buf is reset per (g, ict, ni); each x is read
    // once per (ni, c) regardless of grouping (group iteration just
    // partitions ni's into chunks that each fit in partial_outputs).
    for (unsigned g = 0; g < num_groups; g++) {
    const unsigned ni_lo = g * group_size;
    const unsigned ni_hi = std::min(ni_lo + group_size, batch);
    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        for (unsigned ni = ni_lo; ni < ni_hi; ni++) {
            int last_loaded_row = -1;

            for (unsigned oh = 0; oh < out_h; oh++) {
                const int ih_window_max = (int)(oh * stride_h)
                                        - (int)pad_top
                                        + (int)((kh - 1) * dilation_h);

                // -------------------------------------------------------
                // Phase 1: load any rows the current (oh, ict) window
                // needs that are not yet in line_buf.  Only kTileIC
                // channels are loaded here (ic_off..ic_off+ic_valid-1).
                // -------------------------------------------------------
                int load_start = last_loaded_row + 1;
                if (load_start < 0) load_start = 0;
                int load_end = ih_window_max;
                if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                for (int ih = load_start; ih <= load_end; ih++) {
                    const unsigned slot = (unsigned)ih & (kMaxLineBufRows - 1);
                    for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                        const unsigned c     = ic_off + ic_l;
                        const unsigned x_row = (ni * in_ch + c) * in_hw
                                             + (unsigned)ih * in_w;
                        for (unsigned iw = 0; iw < in_w; iw++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr = x_row + iw;
                            line_buf[ic_l][slot][iw] = x[addr];

#ifdef DEBUG_LOAD_DATA_CACHING
                            CycleCounters counters;
                            counters.mt   = 0;
                            counters.ni   = ni;
                            counters.ict  = ict;
                            counters.ic_l = ic_l;
                            counters.oh   = oh;
                            counters.ow   = 0;
                            counters.khi  = (unsigned)ih;
                            counters.kwi  = iw;
                            read_addresses[addr].push_back(counters);
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
                if (load_end > last_loaded_row) {
                    last_loaded_row = load_end;
                }

                for (unsigned ow = 0; ow < out_w; ow++) {

                    // ---------------------------------------------------
                    // Phase 2: stream a fixed kTileIC × kh × kw block of
                    // patch values into patch_pipe.  Lanes ic_l >=
                    // ic_valid are zero-padded so the broadcaster can
                    // operate with a compile-time-fixed input_per_iter
                    // (= kTileIC*kh*kw); the consumer's accumulate uses
                    // ic_valid bound and ignores the padding lanes.
                    // ---------------------------------------------------
                    for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                        const bool ic_ok = (ic_l < ic_valid);
                        for (unsigned khi = 0; khi < kh; khi++) {
                            const int ih = (int)(oh * stride_h + khi * dilation_h)
                                        - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned slot = ih_ok
                                ? ((unsigned)ih & (kMaxLineBufRows - 1))
                                : 0u;
                            for (unsigned kwi = 0; kwi < kw; kwi++) {
                                #pragma HLS PIPELINE II=1
                                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                            - (int)pad_left;
                                const bool iw_ok = (iw >= 0 && (unsigned)iw < in_w);
                                patch_stream.write((ic_ok && ih_ok && iw_ok)
                                    ? line_buf[ic_l][slot][(unsigned)iw]
                                    : Data_t(0));
                            }
                        }
                    }
                } // ow loop
            } // oh loop
        } // ni loop
    } // ict loop
    } // group loop

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second.size() > 1) {
            ++g_conv_debug_duplicate_reads;
            std::cout << it.first << " --> " << std::endl;

            for (auto l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}

// ---------------------------------------------------------------------------
// broadcast_patches — DATAFLOW stage between input_patch_producer and
// process_conv_kernel_tile.
//
// Per outer iteration (standard path: one (ni, ict, oh, ow); depthwise
// passthrough: one full per-mt patch):
//   First pass:   read input_per_iter values from patch_pipe into local_buf
//                 AND simultaneously forward them to patch_stream.  This
//                 single II=1 pipeline absorbs the assembler's output and
//                 emits the first broadcast copy with no extra cycle.
//   Subsequent:   write (broadcast_factor - 1) more copies of local_buf to
//                 patch_stream, flattened into one continuous II=1 pipeline.
//
// broadcast_factor = m_tiles for the standard path, = 1 for depthwise.
// input_per_iter   = kTileIC * kh * kw for the standard path.
// local_buf is sized for one ic-tile worth of patch values
// (kTileIC * kMaxKH * kMaxKW), down from the full-image kMaxInCh-sized
// buffer in the pre-Option-A design.
// ---------------------------------------------------------------------------
static void broadcast_patches(
    hls::stream<Data_t>& patch_pipe,
    hls::stream<Data_t>& patch_stream,
    unsigned             outer_iters,
    unsigned             input_per_iter,
    unsigned             broadcast_factor
) {
    Data_t local_buf[kTileIC * kMaxKH * kMaxKW];

    for (unsigned r = 0; r < outer_iters; r++) {
        for (unsigned i = 0; i < input_per_iter; i++) {
            #pragma HLS PIPELINE II=1
            const Data_t val = patch_pipe.read();
            local_buf[i] = val;
            patch_stream.write(val);
        }

        const unsigned subsequent = (broadcast_factor > 0
                                     ? broadcast_factor - 1
                                     : 0) * input_per_iter;
        unsigned i = 0;
        for (unsigned k = 0; k < subsequent; k++) {
            #pragma HLS PIPELINE II=1
            patch_stream.write(local_buf[i]);
            i = (i + 1 == input_per_iter) ? 0u : i + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Depthwise input patch ASSEMBLER (DATAFLOW source).
//
// Tiled-M version (Option-A, mirror of the standard producer):
//
//     for ni: for mt: for oh: for ow: for m1, kh, kw
//
// with mt OUTER of oh.  line_buf is sized [kTileM][rows][kMaxInW] and
// retained across the oh loop within a single output-channel tile, so
// each x pixel is fetched from DDR exactly once per (ni, c).  When mt
// advances the line buffer is overwritten; we never revisit a previous
// mt-tile so no cached row is needed twice.
//
// Per (ni, mt, oh):
//   Phase 1 — load any new rows for the current (oh, mt) window for the
//             m_valid channels at offset m_off..m_off+m_valid-1.
// Per (ni, mt, oh, ow):
//   Phase 2 — stream a fixed kTileM × kh × kw block of patch values into
//             patch_pipe.  Lanes m1 >= m_valid are zero-padded so the
//             broadcaster can operate with a compile-time-fixed
//             input_per_iter (= kTileM*kh*kw).  The consumer iterates
//             only the valid m1 lanes in its inner accumulate.
// ---------------------------------------------------------------------------
static void input_patch_producer_depthwise(
    const Data_t*        x,
    hls::stream<Data_t>& patch_stream,
    unsigned             num_groups,
    unsigned             group_size,
    unsigned             batch,
    unsigned             in_ch,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_ch,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             kh,
    unsigned             kw,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             dilation_h,
    unsigned             dilation_w,
    unsigned             pad_top,
    unsigned             pad_left
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;
    const unsigned in_hw   = in_h * in_w;

    Data_t line_buf[kTileM][kMaxLineBufRows][kMaxInW];

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    // Loop order: g OUTER of mt OUTER of ni — matches the consumer's
    // (g, mt, ni, oh, ow) iteration.  line_buf is reset per (g, mt, ni);
    // each x is read once per (ni, c).
    for (unsigned g = 0; g < num_groups; g++) {
    const unsigned ni_lo = g * group_size;
    const unsigned ni_hi = std::min(ni_lo + group_size, batch);
    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = ni_lo; ni < ni_hi; ni++) {
            int last_loaded_row = -1;

            for (unsigned oh = 0; oh < out_h; oh++) {
                const int ih_window_max = (int)(oh * stride_h)
                                        - (int)pad_top
                                        + (int)((kh - 1) * dilation_h);

                // ----- Phase 1: load any new rows for this (oh, mt) -----
                int load_start = last_loaded_row + 1;
                if (load_start < 0) load_start = 0;
                int load_end = ih_window_max;
                if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                for (int ih = load_start; ih <= load_end; ih++) {
                    const unsigned slot = (unsigned)ih & (kMaxLineBufRows - 1);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        const unsigned c     = m_off + m1;
                        const unsigned x_row = (ni * in_ch + c) * in_hw
                                             + (unsigned)ih * in_w;
                        for (unsigned iw = 0; iw < in_w; iw++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr = x_row + iw;
                            line_buf[m1][slot][iw] = x[addr];

#ifdef DEBUG_LOAD_DATA_CACHING
                            CycleCounters counters;
                            counters.mt   = mt;
                            counters.ni   = ni;
                            counters.ict  = -1;
                            counters.ic_l = m1;
                            counters.oh   = oh;
                            counters.ow   = 0;
                            counters.khi  = (unsigned)ih;
                            counters.kwi  = iw;
                            read_addresses[addr].push_back(counters);
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
                if (load_end > last_loaded_row) {
                    last_loaded_row = load_end;
                }

                for (unsigned ow = 0; ow < out_w; ow++) {
                    // ----- Phase 2: stream kTileM × kh × kw values -----
                    for (unsigned m1 = 0; m1 < kTileM; m1++) {
                        const bool m_ok = (m1 < m_valid);
                        for (unsigned khi = 0; khi < kh; khi++) {
                            const int ih = (int)(oh * stride_h + khi * dilation_h)
                                        - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned slot = ih_ok
                                ? ((unsigned)ih & (kMaxLineBufRows - 1))
                                : 0u;
                            for (unsigned kwi = 0; kwi < kw; kwi++) {
                                #pragma HLS PIPELINE II=1
                                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                            - (int)pad_left;
                                const bool iw_ok = (iw >= 0 && (unsigned)iw < in_w);
                                patch_stream.write((m_ok && ih_ok && iw_ok)
                                    ? line_buf[m1][slot][(unsigned)iw]
                                    : Data_t(0));
                            }
                        }
                    }
                }
            } // oh loop
        } // ni loop
    } // mt loop
    } // group loop

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second.size() > 1) {
            ++g_conv_debug_duplicate_reads;
            std::cout << it.first << " --> " << std::endl;

            for (auto l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}

static void input_patch_producer(
    const Data_t*        x,
    hls::stream<Data_t>& patch_stream,
    unsigned             num_groups,
    unsigned             group_size,
    unsigned             batch,
    unsigned             in_ch,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_ch,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             kh,
    unsigned             kw,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             dilation_h,
    unsigned             dilation_w,
    unsigned             pad_top,
    unsigned             pad_left,
    unsigned             is_depthwise
) {
    if (!is_depthwise) {
        input_patch_producer_standard(
            x, patch_stream, num_groups, group_size,
            batch, in_ch, in_h, in_w,
            out_ch, out_h, out_w, kh, kw,
            stride_h, stride_w, dilation_h, dilation_w,
            pad_top, pad_left);
    } else {
        input_patch_producer_depthwise(
            x, patch_stream, num_groups, group_size,
            batch, in_ch, in_h, in_w,
            out_ch, out_h, out_w, kh, kw,
            stride_h, stride_w, dilation_h, dilation_w,
            pad_top, pad_left);
    }
}

// ---------------------------------------------------------------------------
// Weight DDR ASSEMBLER (DATAFLOW source) — standard path.
//
// Iterates (g, ict, mt) where g is the batch-group dimension.  For
// num_groups=1 (the case where batch * out_h * out_w * out_ch fits the
// partial_outputs buffer) each weight DDR address is read exactly ONCE
// per ConvKernel invocation.  When num_groups>1 the same weights are
// re-read once per group — the cost of supporting batches that don't fit
// the on-chip accumulator.  The downstream broadcaster replays each
// (g, ict) slab this_group_size * out_h * out_w times.
//
// Two-phase per (ict, mt) tile:
//   Phase 1 — read DDR sequentially in (m1, ic_l, khi, kwi) order into
//             tile_cache.  Each m1 lane reads a contiguous in_ch*kh*kw
//             span (AXI burst-friendly).
//   Phase 2 — emit pipe in (ic_l, khi, kwi, m1) order with m1 cycling
//             fastest.  This matches the consumer's accumulate ri counter
//             (m1 = ri & (kTileM-1)) so the consumer can fuse the pipe
//             read with the MAC at II=1 (see accumulate_standard_streamed).
//
// Lanes m1 >= m_valid and ic_l >= ic_valid are zero-padded so the
// broadcaster operates with a compile-time-fixed input_per_iter
// (= kTileM * kTileIC * kh * kw); zero-padded weights make MACs into
// padding lanes no-ops without bound checks in the inner loop.
// ---------------------------------------------------------------------------
static void weight_producer_standard(
    const Data_t*        weight,
    hls::stream<Data_t>& weight_pipe,
    unsigned             num_groups,
    unsigned             in_ch,
    unsigned             out_ch,
    unsigned             kh,
    unsigned             kw
) {
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;
    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;

    // Per-tile transpose buffer.  Partition on dim=1 (m1) so Phase 2 can
    // read a different m1 each cycle at II=1.
    Data_t tile_cache[kTileM][kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=tile_cache complete dim=1

    for (unsigned g = 0; g < num_groups; g++) {
    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);

            // Phase 1: sequential DDR read per m1 lane.
            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                const bool m_ok = (m1 < m_valid);
                for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                    const bool ic_ok = (ic_l < ic_valid);
                    for (unsigned khi = 0; khi < kh; khi++) {
                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr =
                                (m_off + m1) * in_ch * kh * kw
                              + (ic_off + ic_l) * kh * kw
                              + khi * kw + kwi;
                            const bool valid_lane = m_ok && ic_ok;
                            tile_cache[m1][ic_l][khi][kwi] =
                                valid_lane ? weight[addr] : Data_t(0);

#ifdef DEBUG_LOAD_DATA_CACHING
                            if (valid_lane) {
                                CycleCounters c;
                                c.mt   = mt;
                                c.ni   = 0;
                                c.oh   = 0;
                                c.ow   = 0;
                                c.ict  = ict;
                                c.ic_l = ic_l;
                                c.khi  = khi;
                                c.kwi  = kwi;
                                g_weight_read_addresses[addr]
                                    .push_back(c);
                            }
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
            }

            // Phase 2: emit transposed (ic_l, khi, kwi, m1) — m1 fastest.
            for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                for (unsigned khi = 0; khi < kh; khi++) {
                    for (unsigned kwi = 0; kwi < kw; kwi++) {
                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            #pragma HLS PIPELINE II=1
                            weight_pipe.write(
                                tile_cache[m1][ic_l][khi][kwi]);
                        }
                    }
                }
            }
        }
    }
    } // group loop
}

// ---------------------------------------------------------------------------
// Weight DDR ASSEMBLER (DATAFLOW source) — depthwise path.
//
// Iterates (g, mt) where g is the batch-group dimension.  num_groups=1
// (the typical fits-in-buffer case) means each weight DDR address is
// read exactly ONCE per ConvKernel call; num_groups>1 implies per-group
// re-reads (the relaxation that lets large-batch cases run).  The
// broadcaster forwards each tile passthrough (replay_iters=1); the
// consumer caches it for the group's (ni, oh, ow) sweep.  Lanes
// m1 >= m_valid are zero-padded.
// ---------------------------------------------------------------------------
static void weight_producer_depthwise(
    const Data_t*        weight,
    hls::stream<Data_t>& weight_pipe,
    unsigned             num_groups,
    unsigned             out_ch,
    unsigned             kh,
    unsigned             kw
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    for (unsigned g = 0; g < num_groups; g++) {
    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned m1 = 0; m1 < kTileM; m1++) {
            const bool m_ok = (m1 < m_valid);
            for (unsigned khi = 0; khi < kh; khi++) {
                for (unsigned kwi = 0; kwi < kw; kwi++) {
                    #pragma HLS PIPELINE II=1
                    const size_t addr = (m_off + m1) * kh * kw
                                      + khi * kw + kwi;
                    weight_pipe.write(m_ok ? weight[addr] : Data_t(0));

#ifdef DEBUG_LOAD_DATA_CACHING
                    if (m_ok) {
                        CycleCounters c;
                        c.mt   = mt;
                        c.ni   = 0;
                        c.oh   = 0;
                        c.ow   = 0;
                        c.ict  = 0;
                        c.ic_l = m1;
                        c.khi  = khi;
                        c.kwi  = kwi;
                        g_weight_read_addresses[addr].push_back(c);
                    }
#endif /* DEBUG_LOAD_DATA_CACHING */
                }
            }
        }
    }
    } // group loop
}

static void weight_producer(
    const Data_t*        weight,
    hls::stream<Data_t>& weight_pipe,
    unsigned             num_groups,
    unsigned             in_ch,
    unsigned             out_ch,
    unsigned             kh,
    unsigned             kw,
    unsigned             is_depthwise
) {
    if (!is_depthwise) {
        weight_producer_standard(weight, weight_pipe,
                                 num_groups, in_ch, out_ch, kh, kw);
    } else {
        weight_producer_depthwise(weight, weight_pipe,
                                  num_groups, out_ch, kh, kw);
    }
}

// ---------------------------------------------------------------------------
// broadcast_weights — DATAFLOW stage between weight_producer and
// process_conv_kernel_tile.
//
// Wraps the producer/consumer in a group dimension to allow ni-splitting
// when batch * out_h * out_w * out_ch exceeds the partial_outputs budget.
// Per group g of size this_gs (= min(group_size, batch - g*group_size)):
//
//   for outer_per_group iterations (standard: ic_tiles; depthwise: m_tiles):
//     Phase 1: drain `cache_len` values from weight_pipe into local cache,
//              simultaneously emitting the first replay copy to weight_stream
//              (II=1, mirrors broadcast_patches' read-and-forward trick).
//     Phase 2: emit `replay_iters - 1` more copies of the cache, where
//              replay_iters = this_gs * replay_per_ni + replay_constant.
//
// Standard (replay_per_ni = out_h*out_w, replay_constant = 0,
// cache_len = m_tiles * kTileM * kTileIC * kh * kw):
//   every weight in the (g, ict) slab is read from DDR once per group
//   and reused across the group's (ni, oh, ow, mt) consumer iterations.
//
// Depthwise (replay_per_ni = 0, replay_constant = 1,
// cache_len = kTileM * kh * kw):
//   passthrough — the consumer holds each tile in its own w_buf for the
//   full (ni, oh, ow) sweep within the group.
//
// When num_groups = 1 (the typical batch=1 case) every weight is read from
// DDR exactly once per ConvKernel call.
//
// Memory: cache size <= kMaxWeightCacheEntries (validated by scheduler).
// ---------------------------------------------------------------------------
static void broadcast_weights(
    hls::stream<Data_t>& weight_pipe,
    hls::stream<Data_t>& weight_stream,
    unsigned             num_groups,
    unsigned             group_size,
    unsigned             batch,
    unsigned             outer_per_group,
    unsigned             cache_len,
    unsigned             replay_per_ni,
    unsigned             replay_constant
) {
    Data_t cache[kMaxWeightCacheEntries];

    for (unsigned g = 0; g < num_groups; g++) {
        const unsigned ni_lo    = g * group_size;
        const unsigned this_gs  = std::min(group_size, batch - ni_lo);
        const unsigned replay_iters = this_gs * replay_per_ni + replay_constant;

        for (unsigned o = 0; o < outer_per_group; o++) {
            for (unsigned i = 0; i < cache_len; i++) {
                #pragma HLS PIPELINE II=1
                const Data_t v = weight_pipe.read();
                cache[i] = v;
                weight_stream.write(v);
            }

            const unsigned subsequent = (replay_iters > 0
                                         ? replay_iters - 1
                                         : 0) * cache_len;
            unsigned i = 0;
            for (unsigned k = 0; k < subsequent; k++) {
                #pragma HLS PIPELINE II=1
                weight_stream.write(cache[i]);
                i = (i + 1 == cache_len) ? 0u : i + 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// process_conv_kernel_tile — DATAFLOW consumer.
//
// Both paths share a persistent partial-output accumulator that survives
// across ic-tiles (standard) or mt-tiles (depthwise):
//
//   Per ni:
//     Phase 1 (init):  drain out_h*out_w*out_ch bias values into
//                      partial_outputs[] (BRAM-resident).
//     Phase 2 (accum): standard   — for ict OUTER, (oh, ow, mt) inner;
//                                   load patch[kTileIC][kh][kw],
//                                   read kTileM*kTileIC*kh*kw weight values
//                                   from weight_stream into w_buf,
//                                   reduce ic_valid*kh*kw*kTileM at II=1.
//                      depthwise  — for mt OUTER, (oh, ow) inner;
//                                   read kTileM*kh*kw weight values from
//                                   weight_stream into w_buf ONCE per mt,
//                                   reduce kh*kw*kTileM at II=1.
//     Phase 3 (drain): push partial_outputs to acc_stream in
//                      (oh, ow, mt, m1) order.
//
//   Both input producers read each x pixel from DDR exactly once per
//   (ni, c).  The weight producer reads each weight from DDR exactly once
//   per (ni, ict) [standard] or per ni [depthwise]; the broadcaster
//   replays the cache to satisfy the consumer's (oh, ow, mt) iteration.
//
// Memory constraint: out_h*out_w*out_ch <= kMaxAccPersistEntries.
// ---------------------------------------------------------------------------
static void process_conv_kernel_tile(
    hls::stream<Data_t>&    patch_stream,
    hls::stream<Data_t>&    weight_stream,
    hls::stream<AccData_t>& bias_stream,
    hls::stream<AccData_t>& acc_stream,
    unsigned                num_groups,
    unsigned                group_size,
    unsigned                batch,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                out_ch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                kh,
    unsigned                kw,
    unsigned                stride_h,
    unsigned                stride_w,
    unsigned                dilation_h,
    unsigned                dilation_w,
    unsigned                pad_top,
    unsigned                pad_left,
    unsigned                is_depthwise
) {
    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;
    const unsigned ohw      = out_h * out_w;
    const unsigned ohw_x_oc = ohw * out_ch;

    // partial_outputs holds this_group_size * out_h * out_w * out_ch
    // accumulators for the current group.  Index uses ni_local (offset
    // within the group) so the buffer is reused across groups.
    AccData_t partial_outputs[kMaxAccPersistEntries];

    for (unsigned g = 0; g < num_groups; g++) {
    const unsigned ni_lo   = g * group_size;
    const unsigned this_gs = std::min(group_size, batch - ni_lo);

    // -------- Phase 1: init partial_outputs from bias_stream --------
    for (unsigned ni_local = 0; ni_local < this_gs; ni_local++) {
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = ni_local * ohw_x_oc
                                           + (oh * out_w + ow) * out_ch
                                           + m_off + m1;
                        partial_outputs[idx] = bias_stream.read();
                    }
                }
            }
        }
    }

    // -------- Phase 2a: standard accumulate (ict, mt OUTER of ni) --------
    if (!is_depthwise) {
        for (unsigned ict = 0; ict < ic_tiles; ict++) {
            const unsigned ic_off   = ict * kTileIC;
            const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);
            (void)ic_off; (void)ic_valid;  // padding lanes are zero-weight no-ops

            for (unsigned ni_local = 0; ni_local < this_gs; ni_local++) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        for (unsigned mt = 0; mt < m_tiles; mt++) {
                            const unsigned m_off   = mt * kTileM;
                            const unsigned m_valid = std::min(kTileM,
                                                              out_ch - m_off);

                            Data_t patch[kTileIC][kMaxKH][kMaxKW];
                            #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

                            AccData_t acc[kTileM];
                            #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

                            // Read kTileIC*kh*kw patch values from stream.
                            for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                                for (unsigned khi = 0; khi < kh; khi++) {
                                    for (unsigned kwi = 0; kwi < kw; kwi++) {
                                        #pragma HLS PIPELINE II=1
                                        patch_stream.read(patch[ic_l][khi][kwi]);
                                    }
                                }
                            }

                            const unsigned idx_base = ni_local * ohw_x_oc
                                                    + (oh * out_w + ow) * out_ch
                                                    + m_off;
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[m1] = AccData_t(0);
                            }
                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                acc[m1] = partial_outputs[idx_base + m1];
                            }

                            // Fused weight stream read + MAC at II=1.
                            accumulate_standard_streamed(patch, weight_stream,
                                                         acc, kh, kw);

                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                partial_outputs[idx_base + m1] = acc[m1];
                            }
                        }
                    }
                }
            } // ni_local
        } // ict
    } else {
        // -------- Phase 2b: depthwise accumulate (mt OUTER of ni) --------
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);

            Data_t w_buf[kTileM][kMaxKH][kMaxKW];
            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                for (unsigned khi = 0; khi < kh; khi++) {
                    for (unsigned kwi = 0; kwi < kw; kwi++) {
                        #pragma HLS PIPELINE II=1
                        w_buf[m1][khi][kwi] = weight_stream.read();
                    }
                }
            }

            for (unsigned ni_local = 0; ni_local < this_gs; ni_local++) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        Data_t patch[kTileIC][kMaxKH][kMaxKW];
                        #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

                        AccData_t acc[kTileM];
                        #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                for (unsigned kwi = 0; kwi < kw; kwi++) {
                                    #pragma HLS PIPELINE II=1
                                    patch_stream.read(patch[m1][khi][kwi]);
                                }
                            }
                        }

                        const unsigned idx_base = ni_local * ohw_x_oc
                                                + (oh * out_w + ow) * out_ch
                                                + m_off;
                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            #pragma HLS UNROLL
                            acc[m1] = AccData_t(0);
                        }
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            acc[m1] = partial_outputs[idx_base + m1];
                        }

                        accumulate_depthwise(patch, w_buf, acc, kh, kw);

                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            partial_outputs[idx_base + m1] = acc[m1];
                        }
                    }
                }
            } // ni_local
        } // mt
    } // depthwise

    // -------- Phase 3: drain partial_outputs to acc_stream --------
    // Output order matches write_output_tile's (ni, oh, ow, mt, m1) sink.
    for (unsigned ni_local = 0; ni_local < this_gs; ni_local++) {
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = ni_local * ohw_x_oc
                                           + (oh * out_w + ow) * out_ch
                                           + m_off + m1;
                        acc_stream.write(partial_outputs[idx]);
                    }
                }
            }
        }
    }
    } // group loop
}

#ifdef DEBUG_LOAD_DATA_CACHING
// C-sim-only: scan g_weight_read_addresses for any DDR weight address
// touched more than once, increment the global duplicate counter, and
// print the offending addresses with their access-context list.  Called
// from ConvKernel after the dataflow region completes.
static void dump_weight_duplicates() {
    for (auto it : g_weight_read_addresses) {
        if (it.second.size() > 1) {
            ++g_conv_debug_duplicate_reads;
            std::cout << "[weight] " << it.first << " --> " << std::endl;
            for (auto l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
    g_weight_read_addresses.clear();
}
#endif /* DEBUG_LOAD_DATA_CACHING */

void ConvKernel(
    const Data_t* x,
    const Data_t* weight,
    const Data_t* bias,
    Data_t*       y,
    unsigned      batch,
    unsigned      in_ch,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_ch,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      has_bias,
    unsigned      is_depthwise
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    //
    // Four m_axi ports allow the tool to issue input, weight, bias, and output
    // transactions on separate AXI buses.  All scalar arguments go into the
    // s_axilite ctrl register file accessed by the PS driver.
    // -----------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=x       offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weight  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=y       offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=x            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=weight       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=bias         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=y            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_ch        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_ch       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_h        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_w        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kh           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kw           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_h     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_w     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_h   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_w   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_top      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_left     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=has_bias     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=is_depthwise bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return       bundle=ctrl

    static_assert(kTileM <= kTileIC,
                  "depthwise mode reuses patch[kTileIC] for TILE_M lanes: "
                  "kTileM must be <= kTileIC");

    // -----------------------------------------------------------------------
    // Top-level DATAFLOW region.
    //
    // bias_producer streams initial-accumulator values to
    // process_conv_kernel_tile through bias_stream; both processes run
    // concurrently.  The producer's bias DDR fetch for tile mt+1 overlaps
    // the tail of the consumer's compute for tile mt, hiding the (small)
    // bias load latency entirely after the first tile.
    //
    // bias_stream depth = kTileM is enough to hold one full m_valid push
    // batch, so the producer can stage the next inner iteration's bias
    // while the consumer is still in the previous iteration's compute.
    // -----------------------------------------------------------------------
    #pragma HLS DATAFLOW
    const unsigned bias_rep_count   = batch * out_h * out_w;
    const unsigned ic_tiles         = (in_ch  + kTileIC - 1) / kTileIC;
    const unsigned m_tiles          = (out_ch + kTileM  - 1) / kTileM;

    // Group dimension: split ni into chunks of group_size such that one
    // group's outputs (group_size * out_h * out_w * out_ch) fit into
    // partial_outputs (kMaxAccPersistEntries).  When the whole batch
    // fits, num_groups = 1 and every weight is read from DDR exactly
    // once per ConvKernel call.  Otherwise weights are reread once per
    // group — the explicit relaxation that lets large-batch cases run.
    const unsigned ohw_x_oc       = out_h * out_w * out_ch;
    const unsigned max_group_size = (ohw_x_oc == 0)
        ? batch
        : (kMaxAccPersistEntries / ohw_x_oc);
    const unsigned group_size = std::min(batch, std::max(max_group_size, 1u));
    const unsigned num_groups = (batch + group_size - 1) / group_size;

    // Both paths now emit a fixed-size patch block per outer iter:
    //   Standard:  kTileIC*kh*kw per (g, ict, ni, oh, ow)   broadcast m_tiles×
    //   Depthwise: kTileM *kh*kw per (g, mt , ni, oh, ow)   passthrough
    const unsigned input_per_iter   = is_depthwise
        ? (kTileM  * kh * kw)
        : (kTileIC * kh * kw);
    const unsigned broadcast_iters  = is_depthwise
        ? (batch * m_tiles  * out_h * out_w)
        : (batch * ic_tiles * out_h * out_w);
    const unsigned broadcast_factor = is_depthwise ? 1u : m_tiles;

    // Weight pipeline parameters.  Per-group replay is computed inside
    // broadcast_weights to handle uneven last-group sizes:
    //   Standard:  outer_per_group = ic_tiles
    //              replay_iters    = this_group_size * out_h * out_w
    //   Depthwise: outer_per_group = m_tiles
    //              replay_iters    = 1 (passthrough)
    const unsigned weight_outer_per_group =
        is_depthwise ? m_tiles : ic_tiles;
    const unsigned weight_cache_len =
        is_depthwise
            ? (kTileM * kh * kw)
            : (m_tiles * kTileM * kTileIC * kh * kw);
    const unsigned weight_replay_per_ni  = is_depthwise ? 0u : (out_h * out_w);
    const unsigned weight_replay_constant = is_depthwise ? 1u : 0u;

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    // patch_pipe carries each unique patch value once from the assembler;
    // patch_stream carries the m_tiles× broadcast copy to the consumer.
    // patch_pipe depth is one ic-tile's patch (kTileIC*kh*kw) — the
    // broadcaster drains it as the assembler fills it under DATAFLOW.
    hls::stream<Data_t> patch_pipe;
    #pragma HLS STREAM variable=patch_pipe depth=kTileIC*kMaxKH*kMaxKW

    hls::stream<Data_t> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=kTileIC

    // weight_pipe: each unique weight tile from DDR (one per (ni, ict, mt)
    // standard or per (ni, mt) depthwise).  Sized for one full slab so the
    // broadcaster can drain it under DATAFLOW.
    hls::stream<Data_t> weight_pipe;
    #pragma HLS STREAM variable=weight_pipe depth=kTileM*kTileIC*kMaxKH*kMaxKW

    hls::stream<Data_t> weight_stream;
    #pragma HLS STREAM variable=weight_stream depth=kTileM

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileM

    bias_producer(bias, bias_stream,
                  out_ch, bias_rep_count, has_bias);

    input_patch_producer(x, patch_pipe, num_groups, group_size,
        batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise
    );

    broadcast_patches(patch_pipe, patch_stream,
                      broadcast_iters, input_per_iter, broadcast_factor);

#ifdef DEBUG_LOAD_DATA_CACHING
    g_weight_read_addresses.clear();
#endif

    weight_producer(weight, weight_pipe,
                    num_groups, in_ch, out_ch, kh, kw, is_depthwise);

    broadcast_weights(weight_pipe, weight_stream,
                      num_groups, group_size, batch,
                      weight_outer_per_group, weight_cache_len,
                      weight_replay_per_ni, weight_replay_constant);

    process_conv_kernel_tile(
        patch_stream, weight_stream, bias_stream, acc_stream,
        num_groups, group_size,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch);

#ifdef DEBUG_LOAD_DATA_CACHING
    dump_weight_duplicates();
#endif
}

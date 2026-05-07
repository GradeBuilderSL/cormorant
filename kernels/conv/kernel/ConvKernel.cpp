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

#ifndef __SYNTHESIS__
#define DEBUG_LOAD_DATA_CACHING
#endif

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

#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// Standard: load weight tile for (m_tile, ic_tile).
//
// Weight layout: [out_ch][in_ch][kh][kw].  Per lane offset:
//   (m_off+m1)*in_ch*kh*kw + ic_off*kh*kw
// w_buf is the local 4-D buffer owned by the standard tile compute.
// ---------------------------------------------------------------------------
static void load_standard_weights(
    const Data_t* weight,
    Data_t        w_buf[kTileM][kTileIC][kMaxKH][kMaxKW],
    unsigned      m_off,
    unsigned      m_valid,
    unsigned      ic_off,
    unsigned      ic_valid,
    unsigned      in_ch,
    unsigned      kh,
    unsigned      kw
) {
    #pragma HLS INLINE

    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        const Data_t* w_ptr = weight
            + (m_off + m1) * in_ch * kh * kw
            + ic_off * kh * kw;
        unsigned ic_l = 0, khi_l = 0, kwi_l = 0;
        const unsigned wt_len = ic_valid * kh * kw;
        for (unsigned r = 0; r < wt_len; r++) {
            #pragma HLS PIPELINE II=1
            w_buf[m1][ic_l][khi_l][kwi_l] = w_ptr[r];
            if (++kwi_l == kw) {
                kwi_l = 0;
                if (++khi_l == kh) {
                    khi_l = 0;
                    ++ic_l;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Standard: II=1 pipelined K-reduction over ic_valid × kh × kw × kTileM.
//
// ri runs 0 .. ic_valid*kh*kw*kTileM - 1.  m1 = ri & (kTileM - 1) cycles
// through lanes; acc[m1] is written every kTileM cycles, so the RAW
// dependence distance ≥ MAC latency.
// ---------------------------------------------------------------------------
static void accumulate_standard(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW],
    AccData_t    acc[kTileM],
    unsigned     ic_valid,
    unsigned     kh,
    unsigned     kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0, ic_cnt = 0;
    const unsigned ri_bound = ic_valid * kh * kw * kTileM;
    for (unsigned ri = 0; ri < ri_bound; ri++) {
        #pragma HLS PIPELINE II=1
        const unsigned m1 = ri & (kTileM - 1);
        acc[m1] +=
            AccData_t(patch[ic_cnt][khi_cnt][kwi_cnt]) *
            AccData_t(w_buf[m1][ic_cnt][khi_cnt][kwi_cnt]);

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
// Depthwise: load per-lane weight slices.
//
// Weight layout: [out_ch][1][kh][kw].  Offset for lane m: (m_off+m1)*kh*kw.
// w_buf is the local 3-D buffer owned by the depthwise tile compute.
// ---------------------------------------------------------------------------
static void load_depthwise_weights(
    const Data_t* weight,
    Data_t        w_buf[kTileM][kMaxKH][kMaxKW],
    unsigned      m_off,
    unsigned      m_valid,
    unsigned      kh,
    unsigned      kw
) {
    #pragma HLS INLINE

    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        const Data_t* w_ptr = weight + (m_off + m1) * kh * kw;
        unsigned khi_l = 0, kwi_l = 0;
        for (unsigned r = 0; r < kh * kw; r++) {
            #pragma HLS PIPELINE II=1
            w_buf[m1][khi_l][kwi_l] = w_ptr[r];
            if (++kwi_l == kw) {
                kwi_l = 0;
                ++khi_l;
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

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ict = 0; ict < ic_tiles; ict++) {
            const unsigned ic_off   = ict * kTileIC;
            const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

            // Highest absolute input row currently resident in line_buf
            // for THIS ic-tile.  Reset per (ni, ict) since line_buf is
            // overwritten when ict advances.
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
        } // ict loop
    } // batch loop

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second.size() > 1) {
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

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);

            // Highest absolute input row currently resident in line_buf
            // for THIS mt-tile.  Reset per (ni, mt) since line_buf is
            // overwritten when mt advances.
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
        } // mt loop
    } // batch loop

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second.size() > 1) {
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
            x, patch_stream, batch, in_ch, in_h, in_w,
            out_ch, out_h, out_w, kh, kw,
            stride_h, stride_w, dilation_h, dilation_w,
            pad_top, pad_left);
    } else {
        input_patch_producer_depthwise(
            x, patch_stream, batch, in_ch, in_h, in_w,
            out_ch, out_h, out_w, kh, kw,
            stride_h, stride_w, dilation_h, dilation_w,
            pad_top, pad_left);
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
//                                   weight[kTileM][kTileIC][kh][kw],
//                                   reduce ic_valid*kh*kw*kTileM at II=1.
//                      depthwise  — for mt OUTER, (oh, ow) inner;
//                                   load patch[kTileM][kh][kw],
//                                   load w_buf[kTileM][kh][kw] ONCE per mt,
//                                   reduce kh*kw*kTileM at II=1.
//     Phase 3 (drain): push partial_outputs to acc_stream in
//                      (oh, ow, mt, m1) order.
//
//   Both producers read each x pixel from DDR exactly once per (ni, c):
//   the standard producer's line_buf is retained across oh within a
//   single ic-tile; the depthwise producer's line_buf is retained
//   across oh within a single mt-tile.
//
// Memory constraint: out_h*out_w*out_ch <= kMaxAccPersistEntries.
// ---------------------------------------------------------------------------
static void process_conv_kernel_tile(
    hls::stream<Data_t>&    patch_stream,
    const Data_t*           weight,
    hls::stream<AccData_t>& bias_stream,
    hls::stream<AccData_t>& acc_stream,
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

    AccData_t partial_outputs[kMaxAccPersistEntries];

    for (unsigned ni = 0; ni < batch; ni++) {

        // -------- Phase 1: init partial_outputs from bias_stream --------
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = (oh * out_w + ow) * out_ch
                                             + m_off + m1;
                        partial_outputs[idx] = bias_stream.read();
                    }
                }
            }
        }

        // -------- Phase 2a: standard accumulate (ict OUTER) --------
        if (!is_depthwise) {
            for (unsigned ict = 0; ict < ic_tiles; ict++) {
                const unsigned ic_off   = ict * kTileIC;
                const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

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

                            Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];

                            // Read kTileIC*kh*kw patch values from stream.
                            for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                                for (unsigned khi = 0; khi < kh; khi++) {
                                    for (unsigned kwi = 0; kwi < kw; kwi++) {
                                        #pragma HLS PIPELINE II=1
                                        patch_stream.read(patch[ic_l][khi][kwi]);
                                    }
                                }
                            }

                            load_standard_weights(weight, w_buf,
                                                  m_off, m_valid,
                                                  ic_off, ic_valid,
                                                  in_ch, kh, kw);

                            const unsigned idx_base = (oh * out_w + ow) * out_ch
                                                      + m_off;
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[m1] = AccData_t(0);
                            }
                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                acc[m1] = partial_outputs[idx_base + m1];
                            }

                            accumulate_standard(patch, w_buf, acc,
                                                ic_valid, kh, kw);

                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                partial_outputs[idx_base + m1] = acc[m1];
                            }
                        }
                    }
                }
            } // ict
        } else {
            // -------- Phase 2b: depthwise accumulate (mt OUTER) --------
            for (unsigned mt = 0; mt < m_tiles; mt++) {
                const unsigned m_off   = mt * kTileM;
                const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                // Load weights ONCE per mt (held in BRAM across the
                // entire (oh, ow) sweep below).
                Data_t w_buf[kTileM][kMaxKH][kMaxKW];
                load_depthwise_weights(weight, w_buf, m_off, m_valid, kh, kw);

                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        Data_t patch[kTileIC][kMaxKH][kMaxKW];
                        #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

                        AccData_t acc[kTileM];
                        #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

                        // Read kTileM*kh*kw patch values from stream.
                        // Stored in patch[0..kTileM-1] (depthwise reuses
                        // the [kTileIC]-deep buffer; kTileM <= kTileIC).
                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                for (unsigned kwi = 0; kwi < kw; kwi++) {
                                    #pragma HLS PIPELINE II=1
                                    patch_stream.read(patch[m1][khi][kwi]);
                                }
                            }
                        }

                        const unsigned idx_base = (oh * out_w + ow) * out_ch
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
            } // mt
        } // depthwise

        // -------- Phase 3: drain partial_outputs to acc_stream --------
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = (oh * out_w + ow) * out_ch
                                             + m_off + m1;
                        acc_stream.write(partial_outputs[idx]);
                    }
                }
            }
        }
    } // ni
}

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

    // Both paths now emit a fixed-size patch block per outer iter:
    //   Standard:  kTileIC*kh*kw per (ni, ict, oh, ow)   broadcast m_tiles×
    //   Depthwise: kTileM *kh*kw per (ni, mt , oh, ow)   passthrough
    const unsigned input_per_iter   = is_depthwise
        ? (kTileM  * kh * kw)
        : (kTileIC * kh * kw);
    const unsigned broadcast_iters  = is_depthwise
        ? (batch * m_tiles  * out_h * out_w)
        : (batch * ic_tiles * out_h * out_w);
    const unsigned broadcast_factor = is_depthwise ? 1u : m_tiles;

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

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileM

    bias_producer(bias, bias_stream,
                  out_ch, bias_rep_count, has_bias);

    input_patch_producer(x, patch_pipe, batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise
    );

    broadcast_patches(patch_pipe, patch_stream,
                      broadcast_iters, input_per_iter, broadcast_factor);

    process_conv_kernel_tile(
        patch_stream, weight, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch);
}

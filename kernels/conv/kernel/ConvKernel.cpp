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
// Initialise the accumulator tile from bias_stream.
//
// Zeroes all kTileM lanes (full unroll, one cycle) so unused tail lanes start
// clean, then overlays the m_valid valid lanes by reading from bias_stream.
// The producer always pushes exactly m_valid items per (ni, oh, ow) iteration,
// regardless of has_bias.
// ---------------------------------------------------------------------------
static void init_accumulators(
    AccData_t               acc[kTileM],
    hls::stream<AccData_t>& bias_stream,
    unsigned                m_valid
) {
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        acc[m1] = AccData_t(0);
    }
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        #pragma HLS PIPELINE II=1
        acc[m1] = bias_stream.read();
    }
}

// ---------------------------------------------------------------------------
// Standard: load input patch for (oh, ow, ic_tile).
//
// patch[ic_l][khi][kwi] holds the spatial patch for the current ic_tile.
// First dim indexes input channels within the current ic_tile.
// ---------------------------------------------------------------------------
static void load_standard_patch(
    hls::stream<Data_t>& patch_stream,
    Data_t        patch[kTileIC][kMaxKH][kMaxKW],
    unsigned      ni,
    unsigned      oh,
    unsigned      ow,
    unsigned      ic_off,
    unsigned      ic_valid,
    unsigned      in_ch,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left
) {
    #pragma HLS INLINE

    for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
        for (unsigned khi = 0; khi < kh; khi++) {
            const int ih = (int)(oh * stride_h + khi * dilation_h)
                         - (int)pad_top;
            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
            const unsigned x_row = (ni * in_ch + ic_off + ic_l)
                                  * in_h * in_w
                                  + (ih_ok ? (unsigned)ih * in_w : 0u);
            for (unsigned kwi = 0; kwi < kw; kwi++) {
                #pragma HLS PIPELINE II=1
                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                             - (int)pad_left;
                patch_stream.read(patch[ic_l][khi][kwi]);
            }
        }
    }
}

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
// Drain m_valid accumulator lanes into acc_stream.  One II=1 push per lane.
// ---------------------------------------------------------------------------
static void drain_acc_to_stream(
    const AccData_t         acc[kTileM],
    hls::stream<AccData_t>& acc_stream,
    unsigned                m_valid
) {
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        #pragma HLS PIPELINE II=1
        acc_stream.write(acc[m1]);
    }
}

// ---------------------------------------------------------------------------
// Standard convolution path (group=1).
//
// Tiles the input channel dimension (TILE_IC) and accumulates over
// ic × kH × kW.  For each ic_tile:
//   1. load_standard_patch    — input patch for (oh, ow, ic_tile)
//   2. load_standard_weights  — weight tile for (m_tile, ic_tile)
//   3. accumulate_standard    — II=1 reduction
// After the ic_tile loop the m_valid valid lanes are pushed to acc_stream.
//
// Initial accumulator values are sourced from bias_stream (filled by the
// concurrent bias_producer); acc[] is a private, fully-partitioned scratch
// buffer.
//
// INLINE to preserve the array partitioning of `patch` declared in the
// caller.
// ---------------------------------------------------------------------------
static void compute_standard_conv_tile(
    hls::stream<Data_t>&    patch_stream,
    const Data_t*           weight,
    hls::stream<AccData_t>& bias_stream,
    hls::stream<AccData_t>& acc_stream,
    unsigned                ni,
    unsigned                m_off,
    unsigned                m_valid,
    unsigned                oh,
    unsigned                ow,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                kh,
    unsigned                kw,
    unsigned                stride_h,
    unsigned                stride_w,
    unsigned                dilation_h,
    unsigned                dilation_w,
    unsigned                pad_top,
    unsigned                pad_left,
    unsigned                ic_tiles
) {
    #pragma HLS INLINE

    Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

    AccData_t acc[kTileM];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];

    init_accumulators(acc, bias_stream, m_valid);

    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        load_standard_patch(patch_stream, patch,
                            ni, oh, ow, ic_off, ic_valid,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left);

        load_standard_weights(weight, w_buf,
                              m_off, m_valid, ic_off, ic_valid,
                              in_ch, kh, kw);

        accumulate_standard(patch, w_buf, acc, ic_valid, kh, kw);
    }

    drain_acc_to_stream(acc, acc_stream, m_valid);
}

// ---------------------------------------------------------------------------
// Depthwise: load per-lane input patches.
//
// patch[m1][khi][kwi] holds the spatial patch for input channel m_off+m1
// (uses first kTileM slots of the kTileIC-deep patch buffer; kTileM ≤
// kTileIC required, enforced by static_assert in ConvKernel).
// ---------------------------------------------------------------------------
static void load_depthwise_patch(
    hls::stream<Data_t>& patch_stream,
    Data_t        patch[kTileIC][kMaxKH][kMaxKW],
    unsigned      ni,
    unsigned      m_off,
    unsigned      m_valid,
    unsigned      oh,
    unsigned      ow,
    unsigned      in_ch,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left
) {
    #pragma HLS INLINE

    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        for (unsigned khi = 0; khi < kh; khi++) {
            const int ih = (int)(oh * stride_h + khi * dilation_h)
                         - (int)pad_top;
            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
            const unsigned x_row = (ni * in_ch + m_off + m1)
                                  * in_h * in_w
                                  + (ih_ok ? (unsigned)ih * in_w : 0u);
            for (unsigned kwi = 0; kwi < kw; kwi++) {
                #pragma HLS PIPELINE II=1
                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                             - (int)pad_left;
                patch_stream.read(patch[m1][khi][kwi]);
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
// Depthwise convolution path (group=in_ch).
//
// Each output channel m is convolved with only its corresponding input
// channel c=m.  No ic_tile loop.  Phases: init acc (from bias_stream) → patch
// load → weight load → accumulate → drain to acc_stream.
//
// Initial accumulator values are sourced from bias_stream (filled by the
// concurrent bias_producer); acc[] is a private, fully-partitioned scratch
// buffer.
//
// INLINE to preserve the array partitioning of `patch` declared in the
// caller.
// ---------------------------------------------------------------------------
static void compute_depthwise_conv_tile(
    hls::stream<Data_t>&    patch_stream,
    const Data_t*           weight,
    hls::stream<AccData_t>& bias_stream,
    hls::stream<AccData_t>& acc_stream,
    unsigned                ni,
    unsigned                m_off,
    unsigned                m_valid,
    unsigned                oh,
    unsigned                ow,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                kh,
    unsigned                kw,
    unsigned                stride_h,
    unsigned                stride_w,
    unsigned                dilation_h,
    unsigned                dilation_w,
    unsigned                pad_top,
    unsigned                pad_left
) {
    #pragma HLS INLINE

    Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

    AccData_t acc[kTileM];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    Data_t w_buf[kTileM][kMaxKH][kMaxKW];

    init_accumulators(acc, bias_stream, m_valid);

    load_depthwise_patch(patch_stream, patch,
                         ni, m_off, m_valid, oh, ow,
                         in_ch, in_h, in_w, kh, kw,
                         stride_h, stride_w, dilation_h, dilation_w,
                         pad_top, pad_left);

    load_depthwise_weights(weight, w_buf, m_off, m_valid, kh, kw);

    accumulate_depthwise(patch, w_buf, acc, kh, kw);

    drain_acc_to_stream(acc, acc_stream, m_valid);
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
// Combines a per-channel sliding line buffer with on-the-fly streaming
// of patch values into patch_pipe.  Each unique x[] pixel is fetched from
// DDR exactly once per (ni) iteration:
//
//   line_buf[c][slot][iw] = x[(ni*in_ch + c)*in_h*in_w + ih*in_w + iw]
//     where slot = ih & (kMaxLineBufRows - 1)
//
// The mt-broadcast (m_tiles× replay of every patch) is now done by the
// downstream broadcast_patches stage so the assembler emits each patch
// value exactly once into patch_pipe.  Under HLS DATAFLOW the assembler's
// production for (ni, oh, ow=N+1) overlaps the broadcaster's m_tiles×
// replay of (ni, oh, ow=N), hiding the assemble latency behind the
// broadcast.
//
// Per (ni, oh):
//   Phase 1 — load any new input rows for the current oh's kh-window into
//             line_buf (one DDR read per pixel that just entered the
//             window; rows already resident from earlier oh iterations
//             are reused).
// Per (ni, oh, ow):
//   Phase 2 — assemble one in_ch*kh*kw patch from line_buf and write it
//             into patch_pipe (no on-chip patch buffer in this stage —
//             values flow straight to the downstream broadcaster).
//
// Constraints: in_ch <= kMaxInCh, in_w <= kMaxInW,
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

    Data_t line_buf[kMaxInCh][kMaxLineBufRows][kMaxInW];

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
        // Highest absolute input row currently resident in line_buf.
        // Reset per ni since x[] is offset by ni*in_ch*in_hw.
        int last_loaded_row = -1;

        for (unsigned oh = 0; oh < out_h; oh++) {
            const int ih_window_max = (int)(oh * stride_h)
                                    - (int)pad_top
                                    + (int)((kh - 1) * dilation_h);

            // -------------------------------------------------------
            // Phase 1: load any rows the current oh-window needs that
            // are not yet in line_buf.  Rows are loaded in increasing
            // ih order so the circular buffer evicts only obsolete
            // rows (those outside the largest possible window we'll
            // see while these slots stay live).
            // -------------------------------------------------------
            int load_start = last_loaded_row + 1;
            if (load_start < 0) load_start = 0;
            int load_end = ih_window_max;
            if (load_end >= (int)in_h) load_end = (int)in_h - 1;

            for (int ih = load_start; ih <= load_end; ih++) {
                const unsigned slot = (unsigned)ih & (kMaxLineBufRows - 1);
                for (unsigned c = 0; c < in_ch; c++) {
                    const unsigned x_row = (ni * in_ch + c) * in_hw
                                         + (unsigned)ih * in_w;
                    for (unsigned iw = 0; iw < in_w; iw++) {
                        #pragma HLS PIPELINE II=1
                        const size_t addr = x_row + iw;
                        line_buf[c][slot][iw] = x[addr];

#ifdef DEBUG_LOAD_DATA_CACHING
                        CycleCounters counters;
                        counters.mt   = 0;
                        counters.ni   = ni;
                        counters.ict  = c / kTileIC;
                        counters.ic_l = c % kTileIC;
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
                // Phase 2: stream patch values straight from the line
                // buffer to patch_pipe — no intermediate patch_buf.
                // ---------------------------------------------------
                for (unsigned ict = 0; ict < ic_tiles; ict++) {
                    const unsigned ic_off   = ict * kTileIC;
                    const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

                    for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
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
                                patch_stream.write((ih_ok && iw_ok)
                                    ? line_buf[ic_off + ic_l][slot][(unsigned)iw]
                                    : Data_t(0));
                            }
                        }
                    }
                }
            } // ow loop
        } // oh loop
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
// Per outer iteration (one (ni, oh, ow)):
//   First pass:   read input_per_iter values from patch_pipe into local_buf
//                 AND simultaneously forward them to patch_stream.  This
//                 single II=1 pipeline absorbs the assembler's output and
//                 emits the first broadcast copy with no extra cycle.
//   Subsequent:   write (broadcast_factor - 1) more copies of local_buf to
//                 patch_stream, flattened into one continuous II=1 pipeline
//                 (no per-mt drain — saves the loop-restart cycle that
//                 the previous nested mt/i loops paid m_tiles times).
//
// broadcast_factor = m_tiles for the standard path, = 1 for depthwise
// (the depthwise producer already emits per-(mt,m1) data, so this stage
// degenerates to a passthrough).
// ---------------------------------------------------------------------------
static void broadcast_patches(
    hls::stream<Data_t>& patch_pipe,
    hls::stream<Data_t>& patch_stream,
    unsigned             outer_iters,
    unsigned             input_per_iter,
    unsigned             broadcast_factor
) {
    Data_t local_buf[kMaxInCh * kMaxKH * kMaxKW];

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

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    // Loop order matches the new consumer: (ni, oh, ow, mt, m1, khi, kwi).
    // Depthwise lanes read disjoint input channels, so no broadcast buffer
    // is required — only the loop nest is reshuffled.
    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        for (unsigned khi = 0; khi < kh; khi++) {
                            const int ih = (int)(oh * stride_h + khi * dilation_h)
                                        - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned x_row = (ni * in_ch + m_off + m1)
                                                * in_h * in_w
                                                + (ih_ok ? (unsigned)ih * in_w : 0u);
                            for (unsigned kwi = 0; kwi < kw; kwi++) {
                                #pragma HLS PIPELINE II=1
                                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                            - (int)pad_left;

#ifdef DEBUG_LOAD_DATA_CACHING
                                if (ih_ok && iw >= 0 && (unsigned)iw < in_w) {
                                    size_t read_addr = x_row + (unsigned)iw;

                                    CycleCounters counters;
                                    counters.mt   = mt;
                                    counters.ni   = ni;
                                    counters.ict  = -1;
                                    counters.ic_l = m1;
                                    counters.oh   = oh;
                                    counters.ow   = ow;
                                    counters.khi  = khi;
                                    counters.kwi  = kwi;
                                    read_addresses[read_addr].push_back(counters);
                                }
#endif /* DEBUG_LOAD_DATA_CACHING */

                                patch_stream.write(
                                    (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                                    ? x[x_row + (unsigned)iw]
                                    : Data_t(0)
                                );
                            }
                        }
                    }
                }
            } // ow loop
        } // oh loop
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
// Iterates over (ni, oh, ow, mt) — mt is INSIDE the spatial nest so each
// (ni, oh, ow) patch produced by input_patch_producer_standard is consumed
// by every m_tile back-to-back, eliminating the m_tiles× re-read of x[].
//
// Each inner iteration: pop m_valid bias values from bias_stream into the
// accumulator, run the per-tile compute (standard or depthwise), and push
// m_valid output lanes through acc_stream → DDR via write_output_tile.
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

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                    if (!is_depthwise) {
                        compute_standard_conv_tile(
                            patch_stream, weight, bias_stream, acc_stream,
                            ni, m_off, m_valid, oh, ow,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left, ic_tiles);
                    } else {
                        compute_depthwise_conv_tile(
                            patch_stream, weight, bias_stream, acc_stream,
                            ni, m_off, m_valid, oh, ow,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left);
                    }
                } // m_tile loop
            } // ow loop
        } // oh loop
    } // batch loop
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
    const unsigned rep_count        = batch * out_h * out_w;
    const unsigned m_tiles          = (out_ch + kTileM - 1) / kTileM;
    const unsigned input_per_iter   = (is_depthwise ? out_ch : in_ch) * kh * kw;
    const unsigned broadcast_factor = is_depthwise ? 1u : m_tiles;

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    // patch_pipe carries each unique patch value once from the assembler;
    // patch_stream carries the m_tiles× broadcast copy to the consumer.
    hls::stream<Data_t> patch_pipe;
    #pragma HLS STREAM variable=patch_pipe depth=kMaxInCh*kMaxKH*kMaxKW

    hls::stream<Data_t> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=kTileIC

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileM

    bias_producer(bias, bias_stream,
                  out_ch, rep_count, has_bias);

    input_patch_producer(x, patch_pipe, batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise
    );

    broadcast_patches(patch_pipe, patch_stream,
                      rep_count, input_per_iter, broadcast_factor);

    process_conv_kernel_tile(
        patch_stream, weight, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch);
}

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
//     bias_producer            ──bias_stream──► process_conv_kernel_tile
//                                                        │
//     bias (DDR/gmem2)                                   └──► y (DDR/gmem3)
//
// bias_producer iterates over every m_tile, fetches the m_valid bias values
// for the tile from DDR once on gmem2, then streams that slice batch*out_h*
// out_w times to bias_stream — exactly the rate the consumer expects.  When
// has_bias=0 no DDR transactions are issued; the producer pushes
// AccData_t(0) directly.
//
// process_conv_kernel_tile contains the m_tile/ni/oh/ow loop nest.  Each
// inner iteration: pop m_valid bias values into the accumulator, run the
// per-tile compute (standard or depthwise), push m_valid output lanes
// through acc_stream → DDR via write_output_tile.
//
// Loop structure — standard conv (is_depthwise=0):
//
//   m_tile loop      — tiles the output channel dimension (TILE_M)
//     batch loop     — iterates over input images
//       oh / ow loops — slide over output spatial positions
//         compute tile (producer):
//           [acc init]   — zero acc[], overlay m_valid lanes from bias_stream
//           ic_tile loop — tiles the input channel dimension (TILE_IC)
//             load patch  — current (oh,ow) input patch to on-chip BRAM
//             load w_buf  — weight tile for (m_tile, ic_tile) to on-chip BRAM
//             accumulate  — II=1 K-reduction over ic×kH×kW with TILE_M lanes
//           drain        — push m_valid lanes of acc[] to acc_stream
//         write output (consumer):
//           saturate_cast acc_stream → DDR y
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
#include <iostream>
#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// Spatial output-tile dimensions for the standard-conv path.
//
// Output is processed in (kTileOH × kTileOW) tiles by both the producer
// (input_patch_producer_standard) and consumer (compute_standard_conv_tile).
// kBufIH / kBufIW are the worst-case input-tile extents the producer's BRAM
// must hold given the compile-time bounds on stride and dilation:
//
//   stride_h, stride_w     ≤ kMaxStride
//   dilation_h, dilation_w ≤ kMaxDilation
//
// Caller is trusted to respect these bounds (no runtime check in synthesis).
// ---------------------------------------------------------------------------
static constexpr unsigned kTileOH      = 4;
static constexpr unsigned kTileOW      = 4;
static constexpr unsigned kMaxStride   = 2;
static constexpr unsigned kMaxDilation = 2;
static constexpr unsigned kBufIH =
    (kTileOH - 1) * kMaxStride + (kMaxKH - 1) * kMaxDilation + 1;
static constexpr unsigned kBufIW =
    (kTileOW - 1) * kMaxStride + (kMaxKW - 1) * kMaxDilation + 1;

// Compile-time upper bound on the runtime number of output-channel tiles.
// The standard-conv consumer keeps one (kTileOH × kTileOW × kTileM) accumulator
// set per m_tile so it can fold all m_tiles into a single (oh_t, ow_t) input
// fetch — eliminating the m_tile DDR redundancy.  Caller must guarantee
//   ceil(out_ch / kTileM) ≤ kMaxMTiles.
// At kTileM=8, kMaxMTiles=32 covers out_ch up to 256.  Bump if needed.
static constexpr unsigned kMaxMTiles = 32;

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
// Standard convolution path (group=1) — spatial + mt-folded consumer.
//
// Processes one (ni, oh_t, ow_t) output tile of size up to kTileOH × kTileOW
// for ALL m_tiles in a single pass.  Accumulators are kept per (mt, oh_off,
// ow_off) so the input data fetched by the producer is consumed by every mt
// before moving to the next spatial tile — eliminating the m_tile DDR
// redundancy on both input and weight.
//
// Loop order (matches input_patch_producer_standard's new emission order):
//   init  — for mt: for (oh_off, ow_off): read m_valid bias values into
//           accs[mt][oh_off][ow_off]
//   for ict:
//     for mt:
//       load_standard_weights for (mt, ict)
//       for (oh_off, ow_off):
//         load_standard_patch from patch_stream
//         accumulate_standard into accs[mt][oh_off][ow_off]
//   drain — for mt: for (oh_off, ow_off): push m_valid lanes to acc_stream
//
// Memory:
//   accs[kMaxMTiles][kTileOH][kTileOW][kTileM] — partitioned only on the
//   m1 dim (kTileM) so II=1 MAC interleaving works; the (mt, oh_off, ow_off)
//   address bits go through a single BRAM port per lane.
//
// INLINE to preserve the array partitioning of `accs`/`patch` in the caller.
// ---------------------------------------------------------------------------
static void compute_standard_conv_tile(
    hls::stream<Data_t>&    patch_stream,
    const Data_t*           weight,
    hls::stream<AccData_t>& bias_stream,
    hls::stream<AccData_t>& acc_stream,
    unsigned                ni,
    unsigned                oh_t,
    unsigned                ow_t,
    unsigned                oh_valid,
    unsigned                ow_valid,
    unsigned                in_ch,
    unsigned                out_ch,
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

    AccData_t accs[kMaxMTiles][kTileOH][kTileOW][kTileM];
    #pragma HLS ARRAY_PARTITION variable=accs complete dim=4

    Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];

    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);
        for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
            for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                init_accumulators(accs[mt][oh_off][ow_off],
                                  bias_stream, m_valid);
            }
        }
    }

    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);

            load_standard_weights(weight, w_buf,
                                  m_off, m_valid, ic_off, ic_valid,
                                  in_ch, kh, kw);

            for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
                for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                    load_standard_patch(patch_stream, patch,
                                        ni, oh_t + oh_off, ow_t + ow_off,
                                        ic_off, ic_valid,
                                        in_ch, in_h, in_w, kh, kw,
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        pad_top, pad_left);

                    accumulate_standard(patch, w_buf,
                                        accs[mt][oh_off][ow_off],
                                        ic_valid, kh, kw);
                }
            }
        }
    }

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);
        for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
            for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                drain_acc_to_stream(accs[mt][oh_off][ow_off],
                                    acc_stream, m_valid);
            }
        }
    }
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
// Write one output tile (m_valid lanes drained from acc_stream) to DDR.
//
// Each of the m_valid output channels writes to a non-contiguous DDR address
// (stride = out_h*out_w elements).  y_addr is advanced by ohw each iteration
// using a counter to avoid a multiplier inside the pipeline.
// ---------------------------------------------------------------------------
static void write_output_tile(
    Data_t*                 y,
    hls::stream<AccData_t>& acc_stream,
    unsigned                out_ch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                batch,
    unsigned                is_depthwise
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;
    const unsigned ohw     = out_h * out_w;

    if (!is_depthwise) {
        // Standard drains per (ni, oh_t, ow_t, mt, oh_off, ow_off, m1).
        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh_t = 0; oh_t < out_h; oh_t += kTileOH) {
                const unsigned oh_valid = std::min(kTileOH, out_h - oh_t);
                for (unsigned ow_t = 0; ow_t < out_w; ow_t += kTileOW) {
                    const unsigned ow_valid = std::min(kTileOW, out_w - ow_t);
                    for (unsigned mt = 0; mt < m_tiles; mt++) {
                        const unsigned m_off   = mt * kTileM;
                        const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                        for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
                            for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                                const unsigned oh = oh_t + oh_off;
                                const unsigned ow = ow_t + ow_off;
                                unsigned y_addr =
                                    (ni * out_ch + m_off) * ohw + oh * out_w + ow;
                                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                    #pragma HLS PIPELINE II=1
                                    y[y_addr] =
                                        saturate_cast<Data_t>(acc_stream.read());
                                    y_addr += ohw;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Depthwise drains per (mt, ni, oh, ow, m1) in original row-major order.
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned ni = 0; ni < batch; ni++) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        unsigned y_addr =
                            (ni * out_ch + m_off) * ohw + oh * out_w + ow;
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            y[y_addr] =
                                saturate_cast<Data_t>(acc_stream.read());
                            y_addr += ohw;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bias producer (DATAFLOW source).
//
// Pre-loads all out_ch bias values into a single on-chip buffer (one DDR
// pass on gmem2), then streams them to bias_stream in the order the consumer
// reads them.  The ORDER differs by path:
//
//   Standard (is_depthwise=0):
//     (ni, oh_t, ow_t, mt, oh_off, ow_off, m1)
//     — matches compute_standard_conv_tile, which inits accs[mt][oh_off][ow_off]
//       lanes with all m_tiles folded inside a single (ni, oh_t, ow_t) tile.
//
//   Depthwise (is_depthwise=1):
//     (mt, ni, oh, ow, m1)
//     — unchanged; matches the per-(oh, ow) compute_depthwise_conv_tile.
//
// Total pushes are the same for both paths: out_ch * batch * out_h * out_w.
// When has_bias=0 no DDR transactions are issued and the producer pushes
// AccData_t(0).
// ---------------------------------------------------------------------------
static void bias_producer(
    const Data_t*           bias,
    hls::stream<AccData_t>& bias_stream,
    unsigned                out_ch,
    unsigned                batch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                has_bias,
    unsigned                is_depthwise
) {
    Data_t bias_buf[kMaxMTiles * kTileM];

    if (has_bias) {
        for (unsigned m = 0; m < out_ch; m++) {
            #pragma HLS PIPELINE II=1
            bias_buf[m] = bias[m];
        }
    }

    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    if (!is_depthwise) {
        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh_t = 0; oh_t < out_h; oh_t += kTileOH) {
                const unsigned oh_valid = std::min(kTileOH, out_h - oh_t);
                for (unsigned ow_t = 0; ow_t < out_w; ow_t += kTileOW) {
                    const unsigned ow_valid = std::min(kTileOW, out_w - ow_t);
                    for (unsigned mt = 0; mt < m_tiles; mt++) {
                        const unsigned m_off   = mt * kTileM;
                        const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                        for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
                            for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                    #pragma HLS PIPELINE II=1
                                    bias_stream.write(has_bias
                                        ? AccData_t(bias_buf[m_off + m1])
                                        : AccData_t(0));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned ni = 0; ni < batch; ni++) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            bias_stream.write(has_bias
                                ? AccData_t(bias_buf[m_off + m1])
                                : AccData_t(0));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Standard producer: spatial-tile + halo BRAM cache, mt-folded.
//
// Output is processed in (kTileOH × kTileOW) tiles.  Per
// (ni, oh_t, ow_t, ict) the producer fetches a halo'd input region
// (kTileIC × ih_extent × iw_extent) from DDR into buf — ONCE — and replays
// the kh × kw patches for every (oh_off, ow_off) in the tile, repeated for
// every mt.  Two redundancies are gone vs. the per-(oh, ow) approach:
//   1. sliding-window overlap within a tile (replayed from buf, not DDR);
//   2. m_tile loop (the same input region is now consumed by all m_tiles
//      before the producer advances).
//
// Stream order emitted (must match the spatial+mt-folded consumer):
//   (ni, oh_t, ow_t, ict, mt, oh_off, ow_off, ic_l, khi, kwi)
//
// kBufIH / kBufIW are sized for the worst-case stride/dilation bounds (see
// declarations near the top of this file); the runtime ih_extent / iw_extent
// loops only fill the rows/cols actually needed by the current (oh_valid ×
// ow_valid) tile.
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
    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    Data_t buf[kTileIC][kBufIH][kBufIW];
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1

#ifdef DEBUG_LOAD_DATA_CACHING
    std::map<size_t, int> read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned oh_t = 0; oh_t < out_h; oh_t += kTileOH) {
            const unsigned oh_valid =
                std::min(kTileOH, out_h - oh_t);
            const unsigned ih_extent =
                (oh_valid - 1) * stride_h + (kh - 1) * dilation_h + 1;

            for (unsigned ow_t = 0; ow_t < out_w; ow_t += kTileOW) {
                const unsigned ow_valid =
                    std::min(kTileOW, out_w - ow_t);
                const unsigned iw_extent =
                    (ow_valid - 1) * stride_w + (kw - 1) * dilation_w + 1;

                for (unsigned ict = 0; ict < ic_tiles; ict++) {
                    const unsigned ic_off   = ict * kTileIC;
                    const unsigned ic_valid =
                        std::min(kTileIC, in_ch - ic_off);

                    // Phase 1: fill buf from DDR with zero-fill outside the
                    // input bounds — ONCE per (ni, oh_t, ow_t, ict).
                    for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                        for (unsigned ih_buf = 0; ih_buf < ih_extent; ih_buf++) {
                            const int ih = (int)(oh_t * stride_h + ih_buf)
                                         - (int)pad_top;
                            const bool ih_ok =
                                (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned x_row =
                                (ni * in_ch + ic_off + ic_l) * in_h * in_w
                                + (ih_ok ? (unsigned)ih * in_w : 0u);
                            for (unsigned iw_buf = 0; iw_buf < iw_extent; iw_buf++) {
                                #pragma HLS PIPELINE II=1
                                const int iw = (int)(ow_t * stride_w + iw_buf)
                                             - (int)pad_left;
                                const bool iw_ok =
                                    (iw >= 0 && (unsigned)iw < in_w);

#ifdef DEBUG_LOAD_DATA_CACHING
                                if (ih_ok && iw_ok) {
                                    size_t read_addr = x_row + (unsigned)iw;
                                    if (read_addresses.find(read_addr)
                                        != read_addresses.end()) {
                                        read_addresses[read_addr] += 1;
                                    } else {
                                        read_addresses.insert(
                                            std::make_pair(read_addr, 0));
                                    }
                                }
#endif /* DEBUG_LOAD_DATA_CACHING */

                                buf[ic_l][ih_buf][iw_buf] =
                                    (ih_ok && iw_ok)
                                    ? x[x_row + (unsigned)iw]
                                    : Data_t(0);
                            }
                        }
                    }

                    // Phase 2: replay patches for every mt × (oh_off, ow_off)
                    // — same buf reused across all m_tiles.  Stream order:
                    //   (mt, oh_off, ow_off, ic_l, khi, kwi)
                    for (unsigned mt = 0; mt < m_tiles; mt++) {
                        for (unsigned oh_off = 0; oh_off < oh_valid; oh_off++) {
                            for (unsigned ow_off = 0; ow_off < ow_valid; ow_off++) {
                                for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                                    for (unsigned khi = 0; khi < kh; khi++) {
                                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                                            #pragma HLS PIPELINE II=1
                                            const unsigned ih_buf =
                                                oh_off * stride_h + khi * dilation_h;
                                            const unsigned iw_buf =
                                                ow_off * stride_w + kwi * dilation_w;
                                            patch_stream.write(
                                                buf[ic_l][ih_buf][iw_buf]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second) {
            std::cout << it.first << " --> " << it.second << std::endl;
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}
#undef DEBUG_LOAD_DATA_CACHING
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
    std::map<size_t, int> read_addresses;
#endif

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
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
                                    if (read_addresses.find(read_addr) != read_addresses.end()) {
                                        read_addresses[read_addr] += 1;
                                    } else {
                                        read_addresses.insert(std::make_pair(read_addr, 0));
                                    }
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
                } // ow loop
            } // oh loop
        } // batch loop
    } // m_tile loop

#ifdef DEBUG_LOAD_DATA_CACHING
    for (auto it : read_addresses) {
        if (it.second) {
            std::cout << it.first << " --> " << it.second << std::endl;
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
// Iterates over m_tile, batch, output spatial position.  Each inner
// iteration: pop m_valid bias values from bias_stream (filled by the
// concurrent bias_producer) into the accumulator, run the per-tile compute
// (standard or depthwise), stream m_valid output lanes through acc_stream
// → DDR via write_output_tile.
//
// `patch` and `acc_stream` are local scratch — the only externally-visible
// streaming dependency is bias_stream (input) and y (output via DDR).
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

    if (!is_depthwise) {
        // Standard: ni outermost; compute_standard_conv_tile folds all m_tiles
        // inside one (oh_t, ow_t) pass — no outer mt loop here.
        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh_t = 0; oh_t < out_h; oh_t += kTileOH) {
                const unsigned oh_valid = std::min(kTileOH, out_h - oh_t);
                for (unsigned ow_t = 0; ow_t < out_w; ow_t += kTileOW) {
                    const unsigned ow_valid = std::min(kTileOW, out_w - ow_t);
                    compute_standard_conv_tile(
                        patch_stream, weight, bias_stream, acc_stream,
                        ni, oh_t, ow_t, oh_valid, ow_valid,
                        in_ch, out_ch, in_h, in_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_left, ic_tiles);
                }
            }
        }
    } else {
        // Depthwise: original (mt, ni, oh, ow) nest, unchanged.
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned ni = 0; ni < batch; ni++) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        compute_depthwise_conv_tile(
                            patch_stream, weight, bias_stream, acc_stream,
                            ni, m_off, m_valid, oh, ow,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left);
                    }
                }
            }
        }
    }
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

    // Standard path bursts up to m_tiles*kTileOH*kTileOW init/drain ops per
    // (ni, oh_t, ow_t) tile; size streams to absorb that worst-case burst.
    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kMaxMTiles*kTileM

    hls::stream<Data_t> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=kTileIC

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kMaxMTiles*kTileM

    bias_producer(bias, bias_stream,
                  out_ch, batch, out_h, out_w, has_bias, is_depthwise);
    input_patch_producer(x, patch_stream, batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise
    );

    process_conv_kernel_tile(
        patch_stream, weight, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch, is_depthwise);
}

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
    const Data_t* x,
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
                patch[ic_l][khi][kwi] =
                    (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                    ? x[x_row + (unsigned)iw]
                    : Data_t(0);
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
    const Data_t*           x,
    const Data_t*           weight,
    Data_t                  patch[kTileIC][kMaxKH][kMaxKW],
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

    AccData_t acc[kTileM];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];

    init_accumulators(acc, bias_stream, m_valid);

    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        load_standard_patch(x, patch,
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
    const Data_t* x,
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
                patch[m1][khi][kwi] =
                    (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                    ? x[x_row + (unsigned)iw]
                    : Data_t(0);
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
    const Data_t*           x,
    const Data_t*           weight,
    Data_t                  patch[kTileIC][kMaxKH][kMaxKW],
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

    AccData_t acc[kTileM];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    Data_t w_buf[kTileM][kMaxKH][kMaxKW];

    init_accumulators(acc, bias_stream, m_valid);

    load_depthwise_patch(x, patch,
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
    unsigned                batch
) {
    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
                    const unsigned ohw    = out_h * out_w;
                    unsigned       y_addr = (ni * out_ch + m_off) * ohw + oh * out_w + ow;
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        y[y_addr] = saturate_cast<Data_t>(acc_stream.read());
                        y_addr += ohw;
                    }
                } // ow loop
            } // oh loop
        } // batch loop
    } // m_tile loop
}

// ---------------------------------------------------------------------------
// Bias producer (DATAFLOW source).
//
// Loops over every m_tile.  For each tile:
//   1. fetches m_valid bias values from DDR into a small on-chip buffer
//      (one DDR transaction per tile on gmem2);
//   2. streams that slice `reps` times to bias_stream — one push-cycle of
//      m_valid items per (ni, oh, ow) inner iteration of the consumer.
//
// Total pushes per tile = m_valid * reps.  Total bias DDR reads per tile =
// m_valid (when has_bias=1).  When has_bias=0 no DDR transactions are
// issued and the producer pushes AccData_t(0) directly.
//
// Designed to run concurrently with process_conv_kernel_tile under HLS
// DATAFLOW: while the consumer is still computing the tail of tile mt the
// producer can already be fetching bias for tile mt+1.
// ---------------------------------------------------------------------------
static void bias_producer(
    const Data_t*           bias,
    hls::stream<AccData_t>& bias_stream,
    unsigned                out_ch,
    unsigned                reps,
    unsigned                has_bias
) {
    Data_t bias_buf[kTileM];
    #pragma HLS ARRAY_PARTITION variable=bias_buf complete dim=0

    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        if (has_bias) {
            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                #pragma HLS PIPELINE II=1
                bias_buf[m1] = bias[m_off + m1];
            }
        }

        for (unsigned r = 0; r < reps; r++) {
            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                #pragma HLS PIPELINE II=1
                bias_stream.write(has_bias
                                  ? AccData_t(bias_buf[m1])
                                  : AccData_t(0));
            }
        }
    }
}

static void input_patch_producer(
    const Data_t*           x,
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
    Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=1

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {

                    if (!is_depthwise) {
                        for (unsigned ict = 0; ict < ic_tiles; ict++) {
                            const unsigned ic_off   = ict * kTileIC;
                            const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

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
                                        patch[ic_l][khi][kwi] =
                                            (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                                            ? x[x_row + (unsigned)iw]
                                            : Data_t(0);
                                    }
                                }
                            }
                        }
                    } else {
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
                                    patch[m1][khi][kwi] =
                                        (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                                        ? x[x_row + (unsigned)iw]
                                        : Data_t(0);
                                }
                            }
                        }
                    }
                } // ow loop
            } // oh loop
        } // batch loop
    } // m_tile loop
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
    const Data_t*           x,
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
    Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=1

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {

                    if (!is_depthwise) {
                        compute_standard_conv_tile(
                            x, weight, patch, bias_stream, acc_stream,
                            ni, m_off, m_valid, oh, ow,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left, ic_tiles);
                    } else {
                        compute_depthwise_conv_tile(
                            x, weight, patch, bias_stream, acc_stream,
                            ni, m_off, m_valid, oh, ow,
                            in_ch, in_h, in_w, kh, kw,
                            stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left);
                    }
                } // ow loop
            } // oh loop
        } // batch loop
    } // m_tile loop
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
    unsigned rep_count = batch * out_h * out_w;

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileM

    bias_producer(bias, bias_stream,
                  out_ch, rep_count, has_bias);

    process_conv_kernel_tile(
        x, weight, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch);
}

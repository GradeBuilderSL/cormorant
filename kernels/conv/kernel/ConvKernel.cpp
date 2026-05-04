// ---------------------------------------------------------------------------
// ConvKernel.cpp — 2-D convolution kernel.
//
// Implements the ONNX Conv operator (group=1 or group=in_ch) in a tiled
// structure that maps cleanly to Vitis HLS synthesis.
// See doc/CONV_PLAN.md for a full explanation of the architecture, tiling
// strategy, and II=1 rationale.
//
// Loop structure — standard conv (is_depthwise=0), row-stationary:
//
//   m_tile loop      — tiles the output channel dimension (TILE_M); hoisted
//                      outermost so bias for the tile is fetched from DDR once
//     load_bias_tile — copy bias[m_off..m_off+m_valid-1] → on-chip bias_buf
//     batch loop     — iterates over input images
//       oh loop      — sweeps output rows
//         produce_bias (producer)  — push m_valid initial acc values from
//                                    bias_buf to bias_stream (zero if no bias)
//         compute_standard_conv_row:
//           init_y_row_from_bias  — replicate bias_stream across all out_w
//           ic_tile loop          — tiles the input channel dimension
//             load w_buf            — weight tile for (m_tile, ic_tile) ONCE
//             ow loop               — slide along the output row
//               load patch          — input patch for (oh, ow, ic_tile)
//               accumulate_into_row — II=1 K-reduction → y_row[ow][m1]
//         write_output_row:
//           saturate_cast y_row → DDR y, lane-major (per-lane AXI burst)
//
// Loop structure — depthwise conv (is_depthwise=1), row-stationary:
//
//   m_tile loop
//     load_bias_tile
//     batch loop
//       oh loop
//         produce_bias (producer)
//         compute_depthwise_conv_row:
//           init_y_row_from_bias
//           load_depthwise_weights  — w_buf for m_tile ONCE per row
//           ow loop:
//             load_depthwise_patch  — one spatial patch per m1 lane
//             accumulate_into_row   — II=1 kH×kW reduction → y_row[ow][m1]
//         write_output_row
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
// Compile-time upper bound on out_w (used to size the row-stationary
// y_row[kMaxOutW][kTileM] accumulator buffer).  Models with out_w > kMaxOutW
// are rejected by the inference scheduler / caller.
// ---------------------------------------------------------------------------
static constexpr unsigned kMaxOutW = 256u;

// ---------------------------------------------------------------------------
// Load the bias slice for one m_tile into an on-chip buffer (one DDR fetch
// per m_tile, reused across all (ni, oh, ow) iterations).
//
// has_bias=1: read bias[m_off..m_off+m_valid-1] from DDR (gmem2) into bias_buf.
// has_bias=0: no m_axi transactions issued on the bias bundle; bias_buf is
//             left untouched (produce_bias never reads it in this case).
// ---------------------------------------------------------------------------
static void load_bias_tile(
    const Data_t* bias,
    Data_t        bias_buf[kTileM],
    unsigned      m_off,
    unsigned      m_valid,
    unsigned      has_bias
) {
    if (has_bias) {
        for (unsigned m1 = 0; m1 < m_valid; m1++) {
            #pragma HLS PIPELINE II=1
            bias_buf[m1] = bias[m_off + m1];
        }
    }
}

// ---------------------------------------------------------------------------
// Bias producer.  Pushes m_valid AccData_t initialiser values onto bias_stream
// (one push per output-channel lane, II=1).
//
// Reads from the on-chip bias_buf populated once per m_tile by load_bias_tile;
// no DDR traffic happens here regardless of how many times produce_bias is
// invoked across (ni, oh, ow).
//
// has_bias=1: forward bias_buf[m1] cast to AccData_t.
// has_bias=0: emit AccData_t(0); bias_buf is not read.
//
// Lanes m_valid..kTileM-1 are not produced — the consumer zeroes those
// directly so they consume no bandwidth.
// ---------------------------------------------------------------------------
static void produce_bias(
    const Data_t            bias_buf[kTileM],
    hls::stream<AccData_t>& bias_stream,
    unsigned                m_valid,
    unsigned                has_bias
) {
    if (has_bias) {
        for (unsigned m1 = 0; m1 < m_valid; m1++) {
            #pragma HLS PIPELINE II=1
            bias_stream.write(AccData_t(bias_buf[m1]));
        }
    } else {
        for (unsigned m1 = 0; m1 < m_valid; m1++) {
            #pragma HLS PIPELINE II=1
            bias_stream.write(AccData_t(0));
        }
    }
}

// ---------------------------------------------------------------------------
// Initialise the row-stationary y_row buffer from bias_stream.
//
// Reads m_valid initial values from bias_stream once (producer always pushes
// exactly m_valid items per row), parks them in a small register file, then
// replicates that vector across all out_w positions of y_row.  Lanes
// m_valid..kTileM-1 are zeroed so unused tail lanes start clean.
// ---------------------------------------------------------------------------
static void init_y_row_from_bias(
    AccData_t               y_row[kMaxOutW][kTileM],
    hls::stream<AccData_t>& bias_stream,
    unsigned                out_w,
    unsigned                m_valid
) {
    AccData_t bias_init[kTileM];
    #pragma HLS ARRAY_PARTITION variable=bias_init complete dim=0

    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        bias_init[m1] = AccData_t(0);
    }
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        #pragma HLS PIPELINE II=1
        bias_init[m1] = bias_stream.read();
    }

    for (unsigned ow = 0; ow < out_w; ow++) {
        #pragma HLS PIPELINE II=1
        for (unsigned m1 = 0; m1 < kTileM; m1++) {
            #pragma HLS UNROLL
            y_row[ow][m1] = bias_init[m1];
        }
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
// Standard: II=1 pipelined K-reduction over ic_valid × kh × kw × kTileM,
// accumulating into one column of the row-stationary y_row buffer.
//
// ri runs 0 .. ic_valid*kh*kw*kTileM - 1.  m1 = ri & (kTileM - 1) cycles
// through lanes; y_row[ow][m1] is written every kTileM cycles for fixed ow,
// so the RAW dependence distance for the same lane is kTileM ≥ MAC latency.
// y_row is partitioned on dim=2 so each m1 sees an independent BRAM bank.
// ---------------------------------------------------------------------------
static void accumulate_standard_into_row(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW],
    AccData_t    y_row[kMaxOutW][kTileM],
    unsigned     ow,
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
        y_row[ow][m1] +=
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
// Standard convolution path (group=1) — row-stationary.
//
// Produces one full output row of m_valid lanes for (mt, ni, oh) in y_row.
// Loop order is ict outer, ow inner: each (mt, ni, oh, ict) weight tile is
// loaded from DDR once and reused across all out_w spatial positions in the
// row, cutting weight DDR traffic by an out_w factor versus the previous
// per-tile compute.
//
// Phases:
//   1. init_y_row_from_bias  — read m_valid from bias_stream, replicate to all out_w
//   2. ic_tile loop:
//        load_standard_weights(mt, ict)   once per (mt, ni, oh, ict)
//        for ow: load_standard_patch + accumulate_standard_into_row(ow)
//
// INLINE to preserve the array partitioning of `patch` and `y_row` declared
// in ConvKernel.
// ---------------------------------------------------------------------------
static void compute_standard_conv_row(
    const Data_t*           x,
    const Data_t*           weight,
    Data_t                  patch[kTileIC][kMaxKH][kMaxKW],
    AccData_t               y_row[kMaxOutW][kTileM],
    hls::stream<AccData_t>& bias_stream,
    unsigned                ni,
    unsigned                m_off,
    unsigned                m_valid,
    unsigned                oh,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                out_w,
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

    Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];

    init_y_row_from_bias(y_row, bias_stream, out_w, m_valid);

    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        load_standard_weights(weight, w_buf,
                              m_off, m_valid, ic_off, ic_valid,
                              in_ch, kh, kw);

        for (unsigned ow = 0; ow < out_w; ow++) {
            load_standard_patch(x, patch,
                                ni, oh, ow, ic_off, ic_valid,
                                in_ch, in_h, in_w, kh, kw,
                                stride_h, stride_w, dilation_h, dilation_w,
                                pad_top, pad_left);

            accumulate_standard_into_row(patch, w_buf, y_row, ow,
                                         ic_valid, kh, kw);
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
// Depthwise: II=1 kH×kW reduction with kTileM lanes, accumulating into one
// column of the row-stationary y_row buffer.
//
// ri runs 0 .. kh*kw*kTileM - 1.  m1 = ri & (kTileM - 1) cycles through
// lanes; y_row[ow][m1] is written every kTileM cycles for fixed ow, so the
// RAW dependence distance for the same lane is kTileM ≥ MAC latency.
// ---------------------------------------------------------------------------
static void accumulate_depthwise_into_row(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kMaxKH][kMaxKW],
    AccData_t    y_row[kMaxOutW][kTileM],
    unsigned     ow,
    unsigned     kh,
    unsigned     kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0;
    const unsigned ri_bound_dw = kh * kw * kTileM;
    for (unsigned ri = 0; ri < ri_bound_dw; ri++) {
        #pragma HLS PIPELINE II=1
        const unsigned m1 = ri & (kTileM - 1);
        y_row[ow][m1] +=
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
// Depthwise convolution path (group=in_ch) — row-stationary.
//
// Produces one full output row of m_valid lanes for (mt, ni, oh) in y_row.
// Depthwise weights depend only on mt (no ic dimension), so w_buf is loaded
// once per call and reused across all out_w spatial positions in the row,
// cutting weight DDR traffic by an out_w factor versus the previous per-tile
// compute.
//
// Phases:
//   1. init_y_row_from_bias        — read m_valid from bias_stream, replicate
//   2. load_depthwise_weights(mt)  once per (mt, ni, oh)
//   3. for ow:
//        load_depthwise_patch + accumulate_depthwise_into_row(ow)
//
// INLINE to preserve the array partitioning of `patch` and `y_row` declared
// in ConvKernel.
// ---------------------------------------------------------------------------
static void compute_depthwise_conv_row(
    const Data_t*           x,
    const Data_t*           weight,
    Data_t                  patch[kTileIC][kMaxKH][kMaxKW],
    AccData_t               y_row[kMaxOutW][kTileM],
    hls::stream<AccData_t>& bias_stream,
    unsigned                ni,
    unsigned                m_off,
    unsigned                m_valid,
    unsigned                oh,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                out_w,
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

    Data_t w_buf[kTileM][kMaxKH][kMaxKW];

    init_y_row_from_bias(y_row, bias_stream, out_w, m_valid);

    load_depthwise_weights(weight, w_buf, m_off, m_valid, kh, kw);

    for (unsigned ow = 0; ow < out_w; ow++) {
        load_depthwise_patch(x, patch,
                             ni, m_off, m_valid, oh, ow,
                             in_ch, in_h, in_w, kh, kw,
                             stride_h, stride_w, dilation_h, dilation_w,
                             pad_top, pad_left);

        accumulate_depthwise_into_row(patch, w_buf, y_row, ow, kh, kw);
    }
}

// ---------------------------------------------------------------------------
// Write one output row (m_valid lanes × out_w positions) to DDR from y_row.
//
// Iterates lane-major (m1 outer, ow inner) so the addresses written by each
// lane advance contiguously across (oh, ow's, ow=0..out_w-1), enabling HLS
// to infer a per-lane AXI burst on gmem3.  Across the m_valid lanes the base
// address jumps by ohw = out_h*out_w elements (next output channel).
// y_row is partitioned on dim=2 so each lane reads from an independent BRAM.
// ---------------------------------------------------------------------------
static void write_output_row(
    Data_t*         y,
    const AccData_t y_row[kMaxOutW][kTileM],
    unsigned        m_valid,
    unsigned        ni,
    unsigned        out_ch,
    unsigned        m_off,
    unsigned        oh,
    unsigned        out_h,
    unsigned        out_w
) {
    const unsigned ohw = out_h * out_w;
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        const unsigned y_base = (ni * out_ch + m_off + m1) * ohw + oh * out_w;
        for (unsigned ow = 0; ow < out_w; ow++) {
            #pragma HLS PIPELINE II=1
            y[y_base + ow] = saturate_cast<Data_t>(y_row[ow][m1]);
        }
    }
}

void conv_kernel_processing(
    const Data_t* x,
    const Data_t* weight,
    hls::stream<AccData_t>& bias_stream,
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
    static Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=1

    static AccData_t y_row[kMaxOutW][kTileM];
    #pragma HLS ARRAY_PARTITION variable=y_row complete dim=2

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    // Loop order: mt → ni → oh.  m_tile is hoisted outermost so the bias
    // slice is fetched from DDR once per tile.  The ow loop has been pushed
    // inside compute_*_conv_row so that each (mt, ni, oh, ict) weight tile
    // is loaded once and reused across all out_w positions of the row, and
    // so that y writes within a lane are contiguous (AXI-burstable).
    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {

                if (!is_depthwise) {
                    compute_standard_conv_row(
                        x, weight, patch, y_row, bias_stream,
                        ni, m_off, m_valid, oh,
                        in_ch, in_h, in_w, out_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_left, ic_tiles);
                } else {
                    compute_depthwise_conv_row(
                        x, weight, patch, y_row, bias_stream,
                        ni, m_off, m_valid, oh,
                        in_ch, in_h, in_w, out_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_left);
                }

                write_output_row(y, y_row, m_valid,
                                 ni, out_ch, m_off, oh, out_h, out_w);

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
    #pragma HLS INTERFACE m_axi port=x      offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weight  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=y       offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=x          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=weight      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=bias        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=y           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_ch       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_ch      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_h       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_w       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kh          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kw          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_h    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_w    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_h  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_w  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_top     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_left    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=has_bias     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=is_depthwise bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return       bundle=ctrl

    // -----------------------------------------------------------------------
    // On-chip buffers (BRAM in HLS).
    //
    // Declared static so HLS infers BRAM rather than registers.  All entries
    // are overwritten before use within each loop iteration.
    //
    // patch [kTileIC][kMaxKH][kMaxKW]
    //   Standard conv: holds the input patch for the current (oh,ow,ic_tile).
    //     First dim indexes input channels within the current ic_tile.
    //   Depthwise conv: first dim indexes output-channel lanes within the
    //     current m_tile (each lane loads from input channel m_off+m1).
    //     Requires kTileM ≤ kTileIC (satisfied by default: 8 ≤ 16).
    //   ARRAY_PARTITION complete dim=1 → independent arrays per lane.
    //
    // w_buf [kTileM][kTileIC][kMaxKH][kMaxKW]
    //   Holds the weight tile for the current (m_tile, ic_tile).
    //   Depthwise: only [m1][0][khi][kwi] slots are used.
    //   ARRAY_PARTITION complete dim=1 → kTileM independent arrays.
    //
    // bias_buf [kTileM]
    //   On-chip cache of the bias slice for the current m_tile.  Loaded once
    //   per m_tile by load_bias_tile (one DDR read on gmem2 of m_valid Data_t
    //   elements) and reused across the entire (ni, oh) sweep, so total bias
    //   DDR traffic is m_tiles · m_valid elements instead of
    //   m_tiles · batch · out_h · out_w · m_valid.  ARRAY_PARTITION complete
    //   dim=0 → independent registers per lane.  When has_bias=0 the buffer
    //   is never written or read.
    //
    // bias_stream
    //   FIFO carrying m_valid AccData_t initial accumulator values from
    //   produce_bias (sourced from bias_buf, not DDR) to the row-compute
    //   consumer (init_y_row_from_bias).  When has_bias=0 the producer emits
    //   zeros.  Depth = kTileM matches the per-row push/pop count.
    //
    // y_row [kMaxOutW][kTileM]
    //   Row-stationary output accumulator buffer.  Holds the partial sums for
    //   one output row (mt, ni, oh) across all out_w positions.  ic_tile loop
    //   inside compute_standard_conv_row sweeps over the full ic dimension
    //   while accumulating into y_row[ow][m1], so each (mt, ni, oh, ict)
    //   weight tile is loaded from DDR once and reused for out_w positions
    //   (an out_w-fold reduction in weight DDR traffic vs the previous
    //   per-(ni,oh,ow) compute).  ARRAY_PARTITION complete dim=2 → kTileM
    //   independent banks so the II=1 reduction can hit acc[m1] every cycle
    //   and write_output_row can stream one lane at a time.  Requires
    //   out_w ≤ kMaxOutW (caller-enforced).
    // -----------------------------------------------------------------------
    static_assert(kTileM <= kTileIC,
                  "depthwise mode reuses patch[kTileIC] for TILE_M lanes: "
                  "kTileM must be <= kTileIC");

    static Data_t patch[kTileIC][kMaxKH][kMaxKW];
    #pragma HLS ARRAY_PARTITION variable=patch complete dim=1

    static AccData_t y_row[kMaxOutW][kTileM];
    #pragma HLS ARRAY_PARTITION variable=y_row complete dim=2

    Data_t bias_buf[kTileM];
    #pragma HLS ARRAY_PARTITION variable=bias_buf complete dim=0

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    // Loop order: mt → ni → oh.  m_tile is hoisted outermost so the bias
    // slice is fetched from DDR once per tile.  The ow loop has been pushed
    // inside compute_*_conv_row so that each (mt, ni, oh, ict) weight tile
    // is loaded once and reused across all out_w positions of the row, and
    // so that y writes within a lane are contiguous (AXI-burstable).
    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        load_bias_tile(bias, bias_buf, m_off, m_valid, has_bias);

        for (unsigned ni = 0; ni < batch; ni++) {
            for (unsigned oh = 0; oh < out_h; oh++) {

                produce_bias(bias_buf, bias_stream, m_valid, has_bias);

                if (!is_depthwise) {
                    compute_standard_conv_row(
                        x, weight, patch, y_row, bias_stream,
                        ni, m_off, m_valid, oh,
                        in_ch, in_h, in_w, out_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_left, ic_tiles);
                } else {
                    compute_depthwise_conv_row(
                        x, weight, patch, y_row, bias_stream,
                        ni, m_off, m_valid, oh,
                        in_ch, in_h, in_w, out_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_left);
                }

                write_output_row(y, y_row, m_valid,
                                 ni, out_ch, m_off, oh, out_h, out_w);

            } // oh loop
        } // batch loop
    } // m_tile loop
}

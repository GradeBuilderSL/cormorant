// ---------------------------------------------------------------------------
// ConvKernel.cpp — 2-D convolution kernel.
//
// Implements the ONNX Conv operator (group=1 or group=in_ch) in a tiled
// structure that maps cleanly to Vitis HLS synthesis.
// See doc/CONV_PLAN.md for a full explanation of the architecture, tiling
// strategy, and II=1 rationale.
//
// Loop structure — standard conv (is_depthwise=0):
//
//   m_tile loop     — tiles the output channel dimension (TILE_M)
//     batch loop    — iterates over input images
//       oh / ow loops — slide over output spatial positions
//         [acc init] — zero or bias-initialise accumulators
//         ic_tile loop — tiles the input channel dimension (TILE_IC)
//           load patch  — current (oh,ow) input patch to on-chip BRAM
//           load w_buf  — weight tile for (m_tile, ic_tile) to on-chip BRAM
//           accumulate  — II=1 K-reduction over ic×kH×kW with TILE_M lanes
//         write output — saturate_cast acc → DDR y
//
// Loop structure — depthwise conv (is_depthwise=1):
//
//   m_tile loop
//     batch loop
//       oh / ow loops
//         [acc init]
//         load patch  — one spatial patch per m1 lane, from input channel m_off+m1
//         load w_buf  — weight slice per m1, weight[m*kh*kw .. +kh*kw-1]
//         accumulate  — II=1 kH×kW reduction with TILE_M lanes (no ic loop)
//         write output
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
#include <hls_stream.h>
#include "ConvKernel.h"

void write_output_data(
    unsigned int out_h, 
    unsigned int out_w, 
    unsigned int ni, 
    unsigned int out_ch, 
    const unsigned int m_off, 
    unsigned int oh, 
    unsigned int ow, 
    const unsigned int m_valid, 
    Data_t *y, 
    const AccData_t *acc)
{
    const unsigned ohw = out_h * out_w;
    unsigned y_addr = (ni * out_ch + m_off) * ohw + oh * out_w + ow;
    for (unsigned m1 = 0; m1 < m_valid; m1++)
    {
#pragma HLS PIPELINE II = 1
        y[y_addr] = saturate_cast<Data_t>(acc[m1]);
        y_addr += ohw;
    }
}

void initialize_accumulator(AccData_t *acc)
{
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        acc[m1] = AccData_t(0);
    }
}

void fill_acc_bias(
    unsigned has_bias,
    unsigned m_valid,
    unsigned m_off,
    const Data_t* bias,
    AccData_t *acc
)
{
    if (has_bias) {
        for (unsigned m1 = 0; m1 < m_valid; m1++) {
            #pragma HLS PIPELINE II=1
            acc[m1] = AccData_t(bias[m_off + m1]);
        }
    }
}

// ---------------------------------------------------------------------------
// setup_accumulator — fused init + optional bias load.
//
// Combined into a single task so that the per-spatial-iteration DATAFLOW
// region has a clean 3-stage chain: setup_accumulator → conv_step →
// write_output_data, with `acc` flowing as the inter-stage channel.
// ---------------------------------------------------------------------------
void setup_accumulator(
    unsigned has_bias,
    unsigned m_valid,
    unsigned m_off,
    const Data_t* bias,
    AccData_t *acc
)
{
    initialize_accumulator(acc);
    fill_acc_bias(has_bias, m_valid, m_off, bias, acc);
}

// ---------------------------------------------------------------------------
// Standard-conv per-tile primitives.
//
// The per-ic_tile body of standart_convolution_step is split into three
// independent processes so that an enclosing #pragma HLS DATAFLOW can run the
// two DDR loads concurrently (gmem0 for x, gmem1 for weight), then run the
// accumulate stage afterwards.
//
//   load_patch_tile_std  (gmem0 → patch)   ─┐
//                                            ├─► accumulate_tile_std
//   load_weight_tile_std (gmem1 → w_buf)   ─┘     (patch, w_buf → acc)
//
// Within one dataflow region:
//   • patch  is single-producer / single-consumer
//   • w_buf  is single-producer / single-consumer
//   • acc    is internal to accumulate_tile_std (no cross-process channel)
// ---------------------------------------------------------------------------

void load_patch_tile_std(
    unsigned ic_off,
    unsigned ic_valid,
    unsigned in_ch,
    unsigned in_h,
    unsigned in_w,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    hls::stream<Data_t> &patch_stream
)
{
    // Write order: (ic_l, khi, kwi).  Matches the consumer's restructured
    // read order in accumulate_tile_std, so this producer/consumer pair is a
    // pure SPSC FIFO — no on-chip ping-pong array needed.
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
                const Data_t v =
                    (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                    ? x[x_row + (unsigned)iw]
                    : Data_t(0);
                patch_stream.write(v);
            }
        }
    }
}

void load_weight_tile_std(
    unsigned ic_off,
    unsigned ic_valid,
    unsigned in_ch,
    unsigned m_valid,
    unsigned m_off,
    unsigned kh,
    unsigned kw,
    const Data_t *weight,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW]
)
{
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

void accumulate_tile_std(
    unsigned ic_valid,
    unsigned kh,
    unsigned kw,
    hls::stream<Data_t> &patch_stream,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    // Restructured K-reduction: outer (ic, kh, kw) reads ONE patch element
    // per iteration; inner m1 loop is II=1 with kTileM lanes.  acc[m1] is
    // updated every kTileM cycles → dependency distance ≥ MAC latency, II=1
    // preserved (same as the original flat-counter version).
    for (unsigned ic_cnt = 0; ic_cnt < ic_valid; ic_cnt++) {
        for (unsigned khi_cnt = 0; khi_cnt < kh; khi_cnt++) {
            for (unsigned kwi_cnt = 0; kwi_cnt < kw; kwi_cnt++) {
                const Data_t p = patch_stream.read();
                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                    #pragma HLS PIPELINE II=1
                    acc[m1] +=
                        AccData_t(p) *
                        AccData_t(w_buf[m1][ic_cnt][khi_cnt][kwi_cnt]);
                }
            }
        }
    }
}

void standart_convolution_step(
    unsigned ic_tiles,
    unsigned in_ch,
    unsigned m_valid,
    unsigned m_off,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned in_h,
    unsigned in_w,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    const Data_t *weight,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    for (unsigned ict = 0; ict < ic_tiles; ict++) {
        #pragma HLS DATAFLOW
        // patch_stream is local to the dataflow region: load_patch produces,
        // accumulate consumes — pure SPSC FIFO, no on-chip ping-pong array.
        hls::stream<Data_t> patch_stream;
        #pragma HLS STREAM variable=patch_stream depth=2

        const unsigned ic_off   = ict * kTileIC;
        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

        load_patch_tile_std(ic_off, ic_valid, in_ch, in_h, in_w,
                            kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                            pad_top, pad_left, oh, ow, ni, x, patch_stream);
        load_weight_tile_std(ic_off, ic_valid, in_ch, m_valid, m_off,
                             kh, kw, weight, w_buf);
        accumulate_tile_std(ic_valid, kh, kw, patch_stream, w_buf, acc);
    } // ic_tile loop
}

// ---------------------------------------------------------------------------
// Depthwise per-tile primitives.
// Same DATAFLOW shape as the standard path: two parallel loads + accumulate.
// ---------------------------------------------------------------------------

void load_patch_tile_dw(
    unsigned m_valid,
    unsigned m_off,
    unsigned in_ch,
    unsigned in_h,
    unsigned in_w,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    hls::stream<Data_t> &patch_stream
)
{
    // Write order: (m1, khi, kwi).  m_valid * kh * kw elements total.
    // Matches accumulate_tile_dw's read order → SPSC FIFO, no on-chip array.
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
                const Data_t v =
                    (ih_ok && iw >= 0 && (unsigned)iw < in_w)
                    ? x[x_row + (unsigned)iw]
                    : Data_t(0);
                patch_stream.write(v);
            }
        }
    }
}

void load_weight_tile_dw(
    unsigned m_valid,
    unsigned m_off,
    unsigned kh,
    unsigned kw,
    const Data_t *weight,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW]
)
{
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        const Data_t* w_ptr = weight + (m_off + m1) * kh * kw;
        unsigned khi_l = 0, kwi_l = 0;
        for (unsigned r = 0; r < kh * kw; r++) {
            #pragma HLS PIPELINE II=1
            w_buf[m1][0][khi_l][kwi_l] = w_ptr[r];
            if (++kwi_l == kw) {
                kwi_l = 0;
                ++khi_l;
            }
        }
    }
}

void accumulate_tile_dw(
    unsigned m_valid,
    unsigned kh,
    unsigned kw,
    hls::stream<Data_t> &patch_stream,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    // Read order: (m1, khi, kwi) — matches load_patch_tile_dw's write order.
    // m1 is now the OUTER loop (one full kh*kw reduction per lane), unlike
    // the standard path where m1 is inner.  The inner pipelined (khi, kwi)
    // loop has dep distance 1 on acc[m1] → HLS relaxes II to MAC latency
    // (~3) instead of 1.  Acceptable for depthwise (small per-position work);
    // a partial-sum tree could restore II=1 if synthesis flags this.
    for (unsigned m1 = 0; m1 < m_valid; m1++) {
        for (unsigned khi = 0; khi < kh; khi++) {
            for (unsigned kwi = 0; kwi < kw; kwi++) {
                #pragma HLS PIPELINE II=1
                const Data_t p = patch_stream.read();
                acc[m1] +=
                    AccData_t(p) *
                    AccData_t(w_buf[m1][0][khi][kwi]);
            }
        }
    }
}

void depthwise_convolution_step(
    unsigned m_valid,
    unsigned m_off,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned in_h,
    unsigned in_w,
    unsigned in_ch,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    const Data_t *weight,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    #pragma HLS DATAFLOW
    // patch_stream local to the dataflow region: load_patch produces,
    // accumulate consumes — pure SPSC FIFO, no on-chip ping-pong array.
    hls::stream<Data_t> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=2

    load_patch_tile_dw(m_valid, m_off, in_ch, in_h, in_w,
                       kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                       pad_top, pad_left, oh, ow, ni, x, patch_stream);
    load_weight_tile_dw(m_valid, m_off, kh, kw, weight, w_buf);
    accumulate_tile_dw(m_valid, kh, kw, patch_stream, w_buf, acc);
}

// ---------------------------------------------------------------------------
// Per-spatial-position dataflow regions.
//
// One function per conv mode so the if/else lives OUTSIDE the dataflow region
// (HLS canonical form: dataflow body must contain only function calls and
// variable declarations — warning 214-114).  All values used inside are
// passed as explicit function arguments rather than captured from outer
// scopes, satisfying warning 214-113.
//
// Three sequential tasks chained by `acc`:
//   setup_accumulator → conv_step → write_output_data
// Across spatial iterations DATAFLOW pipelines them: while iter t is on
// gmem3 (write_output), iter t+1 can already be on gmem2 (bias) and the
// conv_step's parallel gmem0 / gmem1 (patch / weight) loads.
// ---------------------------------------------------------------------------

void process_spatial_std(
    unsigned has_bias,
    unsigned m_valid,
    unsigned m_off,
    unsigned ic_tiles,
    unsigned in_ch,
    unsigned in_h,
    unsigned in_w,
    unsigned out_ch,
    unsigned out_h,
    unsigned out_w,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    const Data_t *weight,
    const Data_t *bias,
    Data_t *y,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    //#pragma HLS DATAFLOW
    setup_accumulator(has_bias, m_valid, m_off, bias, acc);
    standart_convolution_step(
        ic_tiles, in_ch, m_valid, m_off, kh, kw,
        stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, in_h, in_w, oh, ow, ni,
        x, weight, w_buf, acc);
    write_output_data(out_h, out_w, ni, out_ch, m_off, oh, ow, m_valid, y, acc);
}

void process_spatial_dw(
    unsigned has_bias,
    unsigned m_valid,
    unsigned m_off,
    unsigned in_ch,
    unsigned in_h,
    unsigned in_w,
    unsigned out_ch,
    unsigned out_h,
    unsigned out_w,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned oh,
    unsigned ow,
    unsigned ni,
    const Data_t *x,
    const Data_t *weight,
    const Data_t *bias,
    Data_t *y,
    Data_t w_buf[][kTileIC][kMaxKH][kMaxKW],
    AccData_t *acc
)
{
    //#pragma HLS DATAFLOW
    setup_accumulator(has_bias, m_valid, m_off, bias, acc);
    depthwise_convolution_step(
        m_valid, m_off, kh, kw,
        stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, in_h, in_w, in_ch, oh, ow, ni,
        x, weight, w_buf, acc);
    write_output_data(out_h, out_w, ni, out_ch, m_off, oh, ow, m_valid, y, acc);
}

// ---------------------------------------------------------------------------
// process_tile_range — one PE's body.
//
// Handles m_tiles in the half-open range [mt_start, mt_end).  Two of these
// run concurrently inside ConvKernel under #pragma HLS DATAFLOW, splitting
// the output-channel-tile dimension between two parallel processing engines.
//
// Per-PE state (w_buf, acc) is private to each call → no data hazard between
// PEs.  DDR ports (x, weight, bias, y) are shared across both PEs and HLS
// arbitrates AXI access at the bus level.
//
// The if/else on is_depthwise stays at the (ni)-loop body level so each
// (oh, ow) loop body is a single function call (HLS canonical dataflow form
// at the spatial level inside process_spatial_*).
// ---------------------------------------------------------------------------
void process_tile_range(
    unsigned mt_start,
    unsigned mt_end,
    unsigned batch,
    unsigned in_ch,
    unsigned in_h,
    unsigned in_w,
    unsigned out_ch,
    unsigned out_h,
    unsigned out_w,
    unsigned ic_tiles,
    unsigned kh,
    unsigned kw,
    unsigned stride_h,
    unsigned stride_w,
    unsigned dilation_h,
    unsigned dilation_w,
    unsigned pad_top,
    unsigned pad_left,
    unsigned has_bias,
    unsigned is_depthwise,
    const Data_t *x,
    const Data_t *weight,
    const Data_t *bias,
    Data_t *y
)
{
    // -----------------------------------------------------------------------
    // On-chip buffers (BRAM in HLS).
    //
    // Declared static so HLS infers BRAM rather than registers.  All entries
    // are overwritten before use within each loop iteration.
    //
    // The patch buffer that used to live here is now an hls::stream local to
    // each conv_step's dataflow region (load_patch produces, accumulate
    // consumes — pure SPSC FIFO, no on-chip ping-pong array needed).
    //
    // w_buf [kTileM][kTileIC][kMaxKH][kMaxKW]
    //   Holds the weight tile for the current (m_tile, ic_tile).
    //   Depthwise: only [m1][0][khi][kwi] slots are used.
    //   ARRAY_PARTITION complete dim=1 → kTileM independent arrays.
    //
    // acc [kTileM]
    //   Output channel accumulators.  ARRAY_PARTITION complete dim=0 → all
    //   kTileM registers independent.  acc[m1] is written every kTileM cycles
    //   in the accumulate loop, ensuring the RAW distance ≥ MAC latency.
    // -----------------------------------------------------------------------
    Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
    AccData_t acc  [kTileM];

    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=acc   complete dim=0

    for (unsigned mt = mt_start; mt < mt_end; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {

            if (!is_depthwise) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        process_spatial_std(
                            has_bias, m_valid, m_off,
                            ic_tiles, in_ch, in_h, in_w,
                            out_ch, out_h, out_w,
                            kh, kw, stride_h, stride_w,
                            dilation_h, dilation_w, pad_top, pad_left,
                            oh, ow, ni,
                            x, weight, bias, y,
                            w_buf, acc);
                    } // ow loop
                } // oh loop
            } else {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        process_spatial_dw(
                            has_bias, m_valid, m_off,
                            in_ch, in_h, in_w,
                            out_ch, out_h, out_w,
                            kh, kw, stride_h, stride_w,
                            dilation_h, dilation_w, pad_top, pad_left,
                            oh, ow, ni,
                            x, weight, bias, y,
                            w_buf, acc);
                    } // ow loop
                } // oh loop
            } // depthwise path

        } // ni loop
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
    // The patch buffer that used to live here is now an hls::stream local to
    // each conv_step's dataflow region (load_patch produces, accumulate
    // consumes — pure SPSC FIFO, no on-chip ping-pong array needed).
    //
    // w_buf [kTileM][kTileIC][kMaxKH][kMaxKW]
    //   Holds the weight tile for the current (m_tile, ic_tile).
    //   Depthwise: only [m1][0][khi][kwi] slots are used.
    //   ARRAY_PARTITION complete dim=1 → kTileM independent arrays.
    //
    // acc [kTileM]
    //   Output channel accumulators.  ARRAY_PARTITION complete dim=0 → all
    //   kTileM registers independent.  acc[m1] is written every kTileM cycles
    //   in the accumulate loop, ensuring the RAW distance ≥ MAC latency.
    // -----------------------------------------------------------------------
    static Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
    static AccData_t acc  [kTileM];

    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=acc   complete dim=0

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    for (unsigned mt = 0; mt < m_tiles; mt++) {
        const unsigned m_off   = mt * kTileM;
        const unsigned m_valid = std::min(kTileM, out_ch - m_off);

        for (unsigned ni = 0; ni < batch; ni++) {

            // The if/else stays at the (ni)-loop body level so the dataflow
            // region (inside process_spatial_std / process_spatial_dw) only
            // contains function calls — HLS canonical form.
            if (!is_depthwise) {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        process_spatial_std(
                            has_bias, m_valid, m_off,
                            ic_tiles, in_ch, in_h, in_w,
                            out_ch, out_h, out_w,
                            kh, kw, stride_h, stride_w,
                            dilation_h, dilation_w, pad_top, pad_left,
                            oh, ow, ni,
                            x, weight, bias, y,
                            w_buf, acc);
                    } // ow loop
                } // oh loop
            } else {
                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        process_spatial_dw(
                            has_bias, m_valid, m_off,
                            in_ch, in_h, in_w,
                            out_ch, out_h, out_w,
                            kh, kw, stride_h, stride_w,
                            dilation_h, dilation_w, pad_top, pad_left,
                            oh, ow, ni,
                            x, weight, bias, y,
                            w_buf, acc);
                    } // ow loop
                } // oh loop
            } // depthwise path

        } // ni loop

    } // m_tile loop
}

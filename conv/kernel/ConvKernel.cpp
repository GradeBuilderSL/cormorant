// ---------------------------------------------------------------------------
// ConvKernel.cpp — 2-D convolution kernel.
//
// Implements the ONNX Conv operator (group=1) in a tiled structure that maps
// cleanly to Vitis HLS synthesis.  See doc/CONV_PLAN.md for a full explanation
// of the architecture, tiling strategy, and II=1 rationale.
//
// Loop structure (see doc/CONV_PLAN.md §2.5):
//
//   batch loop      — iterates over input images
//     oh / ow loops — slide over output spatial positions
//       m_tile loop — tiles the output channel dimension (TILE_M)
//         [acc init] — zero or bias-initialise accumulators
//         ic_tile loop — tiles the input channel dimension (TILE_IC)
//           load patch  — current (oh,ow) input patch to on-chip BRAM
//           load w_buf  — weight tile for (m_tile, ic_tile) to on-chip BRAM
//           accumulate  — II=1 K-reduction over ic×kH×kW with TILE_M lanes
//         write output — saturate_cast acc → DDR y
//
// II=1 strategy (§2.4):
//   The flat counter ri runs 0 .. ic_valid*kh*kw*TILE_M-1.
//   m1 = ri & (TILE_M - 1)  — compile-time bitmask, no divider.
//   acc[m1] is written every TILE_M cycles, satisfying the dependency distance
//   requirement (TILE_M ≥ ap_fixed<16,8> MAC latency ≈ 3 cycles).
// ---------------------------------------------------------------------------

#include <algorithm>
#include "ConvKernel.h"

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
    unsigned      has_bias
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
    #pragma HLS INTERFACE s_axilite port=has_bias    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return      bundle=ctrl

    // -----------------------------------------------------------------------
    // On-chip buffers (BRAM in HLS).
    //
    // Declared static so HLS infers BRAM rather than registers.  All entries
    // are overwritten before use within each loop iteration.
    //
    // patch [kTileIC][kMaxKH][kMaxKW]
    //   Holds the input patch for the current output position (oh, ow) and
    //   current input channel tile.  Out-of-bounds positions are zero-padded
    //   during the load phase so the accumulate loop needs no bounds check.
    //   ARRAY_PARTITION complete dim=1 → kTileIC independent arrays, each
    //   [kMaxKH][kMaxKW].  All kTileIC channels readable in the same cycle.
    //
    // w_buf [kTileM][kTileIC][kMaxKH][kMaxKW]
    //   Holds the weight tile for the current (m_tile, ic_tile).
    //   ARRAY_PARTITION complete dim=1 → kTileM independent arrays, one per
    //   output channel lane.  The m1-indexed lane is selected by a mux;
    //   all lanes are independently readable.
    //
    // acc [kTileM]
    //   Output channel accumulators.  ARRAY_PARTITION complete dim=0 → all
    //   kTileM registers independent.  acc[m1] is written every kTileM cycles
    //   in the accumulate loop, ensuring the RAW distance ≥ MAC latency.
    // -----------------------------------------------------------------------
    static Data_t    patch[kTileIC][kMaxKH][kMaxKW];
    static Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
    static AccData_t acc  [kTileM];

    #pragma HLS ARRAY_PARTITION variable=patch complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=acc   complete dim=0

    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    for (unsigned ni = 0; ni < batch; ni++) {

        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {

                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                    // -------------------------------------------------------
                    // Initialise accumulators.
                    //
                    // All kTileM registers are zeroed first (full unroll, one
                    // cycle).  If has_bias=1, the valid m1 lanes are then
                    // overwritten with the bias values from DDR.  The bias
                    // reads are guarded by the if-statement so no m_axi
                    // transaction is issued on gmem2 when has_bias=0.
                    // -------------------------------------------------------
                    for (unsigned m1 = 0; m1 < kTileM; m1++) {
                        #pragma HLS UNROLL
                        acc[m1] = AccData_t(0);
                    }
                    if (has_bias) {
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            acc[m1] = AccData_t(bias[m_off + m1]);
                        }
                    }

                    for (unsigned ict = 0; ict < ic_tiles; ict++) {
                        const unsigned ic_off   = ict * kTileIC;
                        const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

                        // ---------------------------------------------------
                        // Load input patch for (oh, ow, ic_tile).
                        //
                        // For each (ic_l, khi), the kernel height row base
                        // address ih is computed once, then the kW elements
                        // along the width dimension are loaded in a burst
                        // (dilation_w=1 gives consecutive addresses).
                        //
                        // Boundary check: ih/iw outside [0,in_h)×[0,in_w)
                        // maps to zero (zero-padding), so the accumulate loop
                        // below requires no conditional.
                        // ---------------------------------------------------
                        for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                const int ih = (int)(oh * stride_h + khi * dilation_h)
                                             - (int)pad_top;
                                const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                                // Row base in DDR (valid only when ih_ok)
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

                        // ---------------------------------------------------
                        // Load weight tile for (m_tile, ic_tile).
                        //
                        // For each m1, the weight elements for (ic_tile, kH, kW)
                        // are contiguous in DDR:
                        //   weight[(m_off+m1)*in_ch*kh*kw + ic_off*kh*kw .. +ic_valid*kh*kw-1]
                        // Each m1 iteration issues a burst of ic_valid*kh*kw
                        // elements.  Counter chain tracks the destination
                        // index in w_buf without runtime division.
                        // ---------------------------------------------------
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

                        // ---------------------------------------------------
                        // Accumulate: II=1 pipelined K-reduction.
                        //
                        // ri runs 0 .. ic_valid*kh*kw*kTileM - 1.
                        //
                        //   m1 = ri & (kTileM-1)  — bitwise AND (kTileM is a
                        //        compile-time power of 2); selects output lane.
                        //
                        // acc[m1] is written at ri and next at ri+kTileM, so
                        // the dependency distance = kTileM ≥ MAC latency (~3)
                        // → II=1 achievable.
                        //
                        // Spatial counters (kwi_cnt, khi_cnt, ic_cnt) advance
                        // every kTileM iterations.  The compare against kw/kh
                        // (runtime values) synthesises as comparators, not
                        // dividers.
                        // ---------------------------------------------------
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
                    } // ic_tile loop

                    // -------------------------------------------------------
                    // Write output tile.
                    //
                    // Each of the m_valid output channels writes to a non-
                    // contiguous DDR address (stride = out_h*out_w elements).
                    // y_addr is advanced by ohw each iteration using a counter
                    // to avoid a multiplier inside the pipeline.
                    // -------------------------------------------------------
                    const unsigned ohw   = out_h * out_w;
                    unsigned y_addr = (ni * out_ch + m_off) * ohw + oh * out_w + ow;
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        y[y_addr] = saturate_cast<Data_t>(acc[m1]);
                        y_addr += ohw;
                    }
                } // m_tile loop

            } // ow loop
        } // oh loop

    } // batch loop
}

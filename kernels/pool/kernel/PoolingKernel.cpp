// ---------------------------------------------------------------------------
// PoolingKernel.cpp — 2-D pooling kernel.
//
// Implements the ONNX MaxPool / AveragePool / LpPool operators (and their
// Global variants) in a channel-tiled structure that maps cleanly to Vitis
// HLS synthesis.
//
// Loop structure:
//
//   batch loop      — iterates over input images
//     oh / ow loops — slide over output spatial positions
//       valid_count — count valid (non-padded) pixels for AVG denominator
//       inv_denom   — precomputed reciprocal for multiply-instead-of-divide
//       c_tile loop — tiles the channel dimension (kTileC)
//         load win_buf  — pool window into on-chip buffer
//         acc init      — −∞ for MAX, 0 for AVG/LP  (fully unrolled, 1 cycle)
//         reduce loop   — II=1 flat counter over pool_h*pool_w*kTileC
//         finalize+write — apply post-reduction op and write to DDR
//
// II=1 strategy (reduction loop):
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
#include "PoolingKernel.h"


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
    // On-chip buffers (BRAM in HLS).
    //
    // win_buf[kTileC][kMaxPoolH][kMaxPoolW]
    //   Holds the pool window for the current output position (oh, ow) and
    //   current channel tile.  Out-of-bounds positions are filled with the
    //   identity element during the load phase so the reduce loop needs no
    //   bounds check.
    //   ARRAY_PARTITION complete dim=1 → kTileC independent BRAM arrays,
    //   each [kMaxPoolH][kMaxPoolW].  All kTileC channels readable same cycle.
    //
    // acc[kTileC]
    //   Per-channel accumulators.  ARRAY_PARTITION complete dim=0 → all
    //   kTileC registers independent.  acc[c1] is written every kTileC cycles
    //   in the reduce loop, satisfying the dependency distance requirement.
    // -----------------------------------------------------------------------
    static Data_t    win_buf[kTileC][kMaxPoolH][kMaxPoolW];
    static AccData_t acc    [kTileC];

    #pragma HLS ARRAY_PARTITION variable=win_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=acc     complete dim=0

    const unsigned c_tiles = (channels + kTileC - 1) / kTileC;

    for (unsigned ni = 0; ni < batch; ni++) {

        for (unsigned oh = 0; oh < out_h; oh++) {
            for (unsigned ow = 0; ow < out_w; ow++) {

                // -------------------------------------------------------
                // Count valid (non-padded) pixels for this output position.
                // Used as the AVG denominator when count_include_pad=0.
                // Computed once per (oh, ow) — same for all channels.
                // -------------------------------------------------------
                unsigned valid_count = 0;
                for (unsigned khi = 0; khi < pool_h; khi++) {
                    for (unsigned kwi = 0; kwi < pool_w; kwi++) {
                        const int ih = (int)(oh * stride_h + khi * dil_h) - (int)pad_top;
                        const int iw = (int)(ow * stride_w + kwi * dil_w) - (int)pad_left;
                        if (ih >= 0 && (unsigned)ih < in_h &&
                            iw >= 0 && (unsigned)iw < in_w)
                            valid_count++;
                    }
                }

                // Precompute reciprocal denominator for AVG (multiply beats divide).
                const unsigned denom_u = count_include_pad
                    ? (pool_h * pool_w)
                    : valid_count;
                const float inv_denom = (denom_u > 0u) ? 1.0f / (float)denom_u : 0.0f;

                for (unsigned ct = 0; ct < c_tiles; ct++) {
                    const unsigned c_off   = ct * kTileC;
                    const unsigned c_valid = std::min(kTileC, channels - c_off);

                    // ---------------------------------------------------
                    // Load pool window into win_buf.
                    //
                    // For each (c_l, khi), compute the DDR row address once,
                    // then burst-load kw elements along the width dimension.
                    // Out-of-bounds pixels are filled with the identity:
                    //   MAX → kNegInfSentinel (saturates to min representable)
                    //   AVG / LP → 0
                    // The reduce loop therefore needs no bounds checks.
                    // ---------------------------------------------------
                    for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                        const unsigned x_c_base =
                            (ni * channels + c_off + c_l) * in_h * in_w;
                        for (unsigned khi = 0; khi < pool_h; khi++) {
                            const int ih =
                                (int)(oh * stride_h + khi * dil_h) - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned x_row =
                                x_c_base + (ih_ok ? (unsigned)ih * in_w : 0u);
                            for (unsigned kwi = 0; kwi < pool_w; kwi++) {
                                #pragma HLS PIPELINE II=1
                                const int iw =
                                    (int)(ow * stride_w + kwi * dil_w) - (int)pad_left;
                                const bool valid =
                                    ih_ok && iw >= 0 && (unsigned)iw < in_w;
                                const Data_t pad_val = (pool_type == kPoolMax)
                                    ? Data_t(kDataMin)
                                    : Data_t(0);
                                win_buf[c_l][khi][kwi] =
                                    valid ? x[x_row + (unsigned)iw] : pad_val;
                            }
                        }
                    }

                    // ---------------------------------------------------
                    // Initialise accumulators.
                    //
                    // Fully unrolled → kTileC registers written in 1 cycle.
                    //   MAX: kAccMin sentinel (minimum representable AccData_t value;
                    //        any valid input beats it on the first comparison)
                    //   AVG / LP: 0 (sum starts at zero)
                    // ---------------------------------------------------
                    for (unsigned c1 = 0; c1 < kTileC; c1++) {
                        #pragma HLS UNROLL
                        acc[c1] = (pool_type == kPoolMax)
                            ? AccData_t(kAccMin)
                            : AccData_t(0);
                    }

                    // ---------------------------------------------------
                    // Reduce: II=1 pipelined flat counter.
                    //
                    // ri runs 0 .. pool_h*pool_w*kTileC - 1.
                    //
                    //   c1 = ri & (kTileC - 1)  — bitwise AND (kTileC is
                    //        a compile-time power of 2); selects channel lane.
                    //
                    // acc[c1] is written at ri and next at ri+kTileC, so
                    // the dependency distance = kTileC ≥ operation latency
                    // → II=1 achievable for all three pool types.
                    //
                    // Spatial counters (kwi_cnt, khi_cnt) advance every
                    // kTileC iterations via compare-increment chains.
                    // ---------------------------------------------------
                    unsigned kwi_cnt = 0, khi_cnt = 0;
                    const unsigned ri_bound = pool_h * pool_w * kTileC;
                    for (unsigned ri = 0; ri < ri_bound; ri++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned c1  = ri & (kTileC - 1);
                        const AccData_t val = AccData_t(win_buf[c1][khi_cnt][kwi_cnt]);

                        if (pool_type == kPoolMax) {
                            if (val > acc[c1]) acc[c1] = val;
                        } else if (pool_type == kPoolAvg) {
                            acc[c1] += val;
                        } else {
                            // LP: p=1 → |val|,  p=2 → val²
                            // Both branches cast to AccData_t: unary minus widens by 1 bit,
                            // multiplication doubles the bit width; explicit casts required.
                            const AccData_t contrib = (lp_order == 1u)
                                ? (val < AccData_t(0) ? AccData_t(-val) : val)
                                : AccData_t(val * val);
                            acc[c1] += contrib;
                        }

                        if ((ri & (kTileC - 1)) == kTileC - 1) {
                            if (++kwi_cnt == pool_w) {
                                kwi_cnt = 0;
                                ++khi_cnt;
                            }
                        }
                    }

                    // ---------------------------------------------------
                    // Finalise and write output.
                    //
                    // Post-reduction operations:
                    //   MAX: identity — acc is already the max value.
                    //   AVG: multiply by precomputed reciprocal denominator.
                    //   LP p=1: identity — acc is already Σ|x_i|.
                    //   LP p=2: sqrt(acc) via float (DSP-friendly in HLS).
                    //
                    // Output addresses are non-contiguous in C (stride =
                    // out_h*out_w per channel); y_addr is advanced by a
                    // counter to avoid a multiplier inside the pipeline.
                    // ---------------------------------------------------
                    const unsigned hw_stride = out_h * out_w;
                    unsigned y_addr =
                        (ni * channels + c_off) * out_h * out_w + oh * out_w + ow;

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
                        y[y_addr] = saturate_cast<Data_t>(result);
                        y_addr += hw_stride;
                    }

                } // c_tile loop

            } // ow loop
        } // oh loop

    } // batch loop
}

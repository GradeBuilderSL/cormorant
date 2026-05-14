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
//     bias_producer           ──bias_stream────►  process_conv_kernel_tile
//     input_patch_producer    ──patch_pipe─────►  broadcast_patches  ──patch_stream──►   │
//     stream_load_weights     ──weight_stream──►                                         │
//                                                                                        └──acc_stream──► write_output_tile
//                                                                                                                │
//     x/weight/bias (DDR gmem0/1/2)                                                                              ▼
//                                                                                                         y (DDR gmem3)
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
// oh-chunking helper.
//
// The Option-A consumer keeps a persistent per-output accumulator in
// partial_outputs[kMaxAccPersistEntries] that survives across the ic-tile
// (standard) or mt-tile (depthwise) reduction.  When out_h·out_w·out_ch
// exceeds kMaxAccPersistEntries, the full output no longer fits on-chip.
//
// Rather than degrading to a no-persistent-acc mode (which would re-read
// every input pixel ic_tiles · m_tiles times per (ni, oh, ow) — bandwidth
// catastrophe), the kernel splits the output along the oh axis into chunks
// whose footprint fits the buffer:
//
//     oh_per_chunk = max(1, kMaxAccPersistEntries / (out_w · out_ch))
//     num_chunks   = ceil(out_h / oh_per_chunk)
//
// Each chunk runs the full Option-A pipeline for its oh sub-range; between
// chunks the patch producer's line_buf is invalidated so the kh-row sliding
// window for the next chunk's first oh is reloaded from DDR (duplicate read
// of ~kh rows per (ni, c, chunk transition)).  Weights are re-streamed per
// chunk too — for the standard path this is just stream replay (the inline
// load loop already replays per (oh, ow), so per-chunk replay is the same
// bandwidth); for the depthwise path the small weight slice is reloaded
// num_chunks times per (ni, mt).
//
// Relaxed constraint: out_w · out_ch  ≤  kMaxAccPersistEntries
// (was: out_h · out_w · out_ch  ≤  kMaxAccPersistEntries).
//
// When out_w · out_ch > kMaxAccPersistEntries the buffer cannot hold even
// one output row.  oh_per_chunk is clamped to 1 in that case so the kernel
// still synthesises and (for small enough total) still works in C-sim, but
// large layers with that property will overflow partial_outputs.  Catching
// it remains the scheduler validator's responsibility.
// ---------------------------------------------------------------------------
static inline void compute_oh_chunking(
    unsigned out_h, unsigned out_w, unsigned out_ch,
    unsigned& oh_per_chunk, unsigned& num_chunks
) {
    const unsigned row_size = out_w * out_ch;
    unsigned per = (row_size > 0) ? (kMaxAccPersistEntries / row_size) : out_h;
    if (per == 0)        per = 1;
    if (per > out_h)     per = out_h;
    oh_per_chunk = per;
    num_chunks   = (out_h + per - 1) / per;
}

// ---------------------------------------------------------------------------
// M-grouping helper (standard path only).
//
// Eliminates per-(oh, ow) weight DDR replay by caching one (ict, M-group)
// weight slab on-chip and reusing it across the chunk's spatial sweep.
// When m_tiles ≤ kMaxMperGroup the cache holds ALL m_tiles' weights for
// the current ict and weights are read from DDR exactly once per
// (ni, chunk, ict).  When m_tiles > kMaxMperGroup the kernel processes
// M in groups of mt_per_group mt-tiles; weights are reloaded num_m_groups
// times per (ni, chunk, ict), still much cheaper than the pre-caching
// out_h·out_w replay.
//
// Patches are re-emitted by the input patch producer per m_group so the
// consumer can stream-read kTileIC·kh·kw patch values once per
// (m_group, oh, ow) and reuse across mt_in_group with cached weights.
// No DDR re-read for patches — line_buf is retained across the entire
// (m_group, oh, ow) sweep within one (chunk, ict).
// ---------------------------------------------------------------------------
static inline void compute_m_grouping(
    unsigned out_ch,
    unsigned& mt_per_group, unsigned& num_m_groups
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;
    unsigned mtg = kMaxMperGroup;
    if (mtg == 0)         mtg = 1;
    if (mtg > m_tiles)    mtg = m_tiles;
    mt_per_group = mtg;
    num_m_groups = (m_tiles + mtg - 1) / mtg;
}

// ---------------------------------------------------------------------------
// ow-tiling helper.
//
// Lifts the previous hard cap "in_w ≤ kMaxInW" by splitting the output
// column axis into tiles whose input-column window fits the compile-time
// kMaxLineBufCols bound on line_buf's column dim.  Within an ow-tile the
// patch producer uses circular column indexing (iw_slot = iw &
// (kMaxLineBufCols-1)) so the same line_buf slots are reused across rows.
// Between ow-tiles, line_buf is invalidated and the next tile's iw range
// is reloaded from DDR — the (kw-1)·dilation_w-col overlap is re-fetched.
//
// Tile geometry (per ow-tile spanning ows [ow_start, ow_start+ow_per_tile)):
//   first iw = ow_start * stride_w - pad_left
//   last  iw = (ow_start + ow_per_tile - 1) * stride_w + (kw-1)*dilation_w - pad_left
//   iw range width = (ow_per_tile-1)*stride_w + (kw-1)*dilation_w + 1
//
// We need iw_range_width ≤ kMaxLineBufCols, so:
//   ow_per_tile = (kMaxLineBufCols - ((kw-1)*dilation_w + 1)) / stride_w + 1
//
// When (kw-1)*dilation_w + 1 > kMaxLineBufCols the kernel-width window
// doesn't fit even one ow — falls back to ow_per_tile=1 (the
// kMaxLineBufCols static_assert in Config.h.in plus the per-kernel-size
// validation in the inference scheduler should make this unreachable for
// well-formed inputs).
// ---------------------------------------------------------------------------
static inline void compute_ow_tiling(
    unsigned out_w, unsigned kw,
    unsigned stride_w, unsigned dilation_w,
    unsigned& ow_per_tile, unsigned& num_ow_tiles
) {
    const unsigned window_w = (kw - 1) * dilation_w + 1;
    unsigned per;
    if (window_w >= kMaxLineBufCols) {
        // Window doesn't leave room for any additional output; clamp.
        per = 1;
    } else {
        per = (kMaxLineBufCols - window_w) / (stride_w > 0 ? stride_w : 1) + 1;
        if (per == 0) per = 1;
    }
    if (per > out_w) per = out_w;
    ow_per_tile = per;
    num_ow_tiles = (out_w + per - 1) / per;
}

// ---------------------------------------------------------------------------
// Standard: II=1 pipelined K-reduction with PN-wide input-channel parallelism.
//
// Each PIPELINE iteration fires kTileIC parallel MACs that share (khi_cnt,
// kwi_cnt, m1) and reduce across the input-channel axis through a PN-wide
// adder tree.  Outer counter ri = (khi, kwi, m1) with m1 cycling 0..kTileM-1
// via the bit mask; the lane rotation keeps the per-lane RAW distance on
// acc[m1] at kTileM cycles (≥ MAC pipeline depth incl. the log2(kTileIC)
// adder tree), avoiding any DEPENDENCE escape.
//
// X-propagation guard on partial IC tiles:
//   When ic_valid < kTileIC the stream_load_weights producer only writes
//   m_valid·ic_valid·kh·kw cells into w_buf — cells at ic_l >= ic_valid are
//   uninitialised BRAM, which is 'X' in RTL.  Multiplying patch·X yields X
//   (C-sim would have given 0 because patch is producer-padded with zeros,
//   so this is an RTL-only failure mode).  The ic_l < ic_valid guard MUXes
//   the BRAM read to 0 on invalid lanes, so the product is 0·0 = 0 and no
//   X reaches the adder tree.  HLS keeps the multiplier — the MUX adds a
//   single LUT per lane on the weight input, no DSP cost.
//
// Throughput: kTileIC MACs / cycle (vs 1 MAC / cycle previously);
// loop bound shrinks from ic_valid·kh·kw·kTileM to kh·kw·kTileM.
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

    unsigned kwi_cnt = 0, khi_cnt = 0;
    const unsigned ri_bound = kh * kw * kTileM;
    for (unsigned ri = 0; ri < ri_bound; ri++) {
        #pragma HLS PIPELINE II=1
        const unsigned m1 = ri & (kTileM - 1);

        AccData_t lane_sum = 0;
        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
            #pragma HLS UNROLL
            const Data_t w_val = (ic_l < ic_valid)
                ? w_buf[m1][ic_l][khi_cnt][kwi_cnt]
                : Data_t(0);
            lane_sum +=
                AccData_t(patch[ic_l][khi_cnt][kwi_cnt]) *
                AccData_t(w_val);
        }
        acc[m1] += lane_sum;

        if ((ri & (kTileM - 1)) == kTileM - 1) {
            if (++kwi_cnt == kw) {
                kwi_cnt = 0;
                if (++khi_cnt == kh) {
                    khi_cnt = 0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Depthwise: II=1 kH×kW reduction with PM-wide channel-parallel lanes.
//
// Each PIPELINE iteration fires kTileM independent MACs — one per output
// channel lane.  Depthwise has no input-channel reduction, so per-lane
// accumulators acc[m1] are pairwise independent.  Loop bound shrinks from
// kh·kw·kTileM (1 MAC / cycle) to kh·kw (kTileM MACs / cycle).
//
// Per-lane RAW distance on acc[m1] is 1 cycle (each iteration writes every
// lane); the ap_fixed<32,16> adder schedules single-cycle at 300 MHz so the
// recurrence closes without an II bump.  Pattern matches Bai et al. FPGA'18
// channel-parallel depthwise micro-architecture.
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
    const unsigned ri_bound_dw = kh * kw;
    for (unsigned ri = 0; ri < ri_bound_dw; ri++) {
        #pragma HLS PIPELINE II=1
        for (unsigned m1 = 0; m1 < kTileM; m1++) {
            #pragma HLS UNROLL
            acc[m1] +=
                AccData_t(patch[m1][khi_cnt][kwi_cnt]) *
                AccData_t(w_buf[m1][khi_cnt][kwi_cnt]);
        }

        if (++kwi_cnt == kw) {
            kwi_cnt = 0;
            ++khi_cnt;
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
// Tiled-IC (Option-A) + M-grouping + ow-tiling.  Loop nest:
//
//     for ni: for chunk: for ict: for ow_tile: for mg:
//         for oh_in_chunk: for ow_in_tile: for ic_l, kh, kw
//
// ow_tile is OUTER of m_group so line_buf's column window for the tile is
// loaded once per (ict, ow_tile) and reused across mg's patch re-emissions.
// Within the tile, line_buf indexes both row and column dims circularly:
//
//     row_slot = ih & (kMaxLineBufRows - 1)
//     col_slot = iw & (kMaxLineBufCols - 1)
//
// kMaxLineBufCols is a compile-time power-of-2 bound on the column window;
// in_w itself is NOT capped, only the per-tile iw extent.  Wider inputs
// produce more ow_tiles, with the (kw-1)·dilation_w-col overlap re-fetched
// from DDR at each tile transition.
//
// Per (ni, chunk, ict, ow_tile, mg, oh):
//   Phase 1 — load new rows × this ow_tile's iw range from DDR.
//             Only fires when mg=0 (subsequent m_groups re-stream the same
//             cached rows from line_buf).
// Per (ni, chunk, ict, ow_tile, mg, oh, ow):
//   Phase 2 — stream kTileIC × kh × kw patch values into patch_pipe.
//
// Constraints: (kh-1)*dilation_h + 1 <= kMaxLineBufRows,
//              (kw-1)*dilation_w + 1 <= kMaxLineBufCols.
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
    const unsigned ic_tiles = (in_ch + kTileIC - 1) / kTileIC;
    const unsigned in_hw    = in_h * in_w;

    unsigned oh_per_chunk, num_chunks;
    compute_oh_chunking(out_h, out_w, out_ch, oh_per_chunk, num_chunks);

    unsigned mt_per_group, num_m_groups;
    compute_m_grouping(out_ch, mt_per_group, num_m_groups);
    (void)mt_per_group;  // Producer only needs the group count (re-emission factor).

    unsigned ow_per_tile, num_ow_tiles;
    compute_ow_tiling(out_w, kw, stride_w, dilation_w,
                      ow_per_tile, num_ow_tiles);

    Data_t line_buf[kTileIC][kMaxLineBufRows][kMaxLineBufCols];

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start = chunk * oh_per_chunk;
        const unsigned oh_end   = std::min(out_h, oh_start + oh_per_chunk);

        for (unsigned ict = 0; ict < ic_tiles; ict++) {
            const unsigned ic_off   = ict * kTileIC;
            const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

          for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
            const unsigned ow_start = owt * ow_per_tile;
            const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile);

            // Input column range needed for this ow_tile.
            const int iw_load_start = (int)(ow_start * stride_w) - (int)pad_left;
            const int iw_load_last  = (int)((ow_end - 1) * stride_w
                                            + (kw - 1) * dilation_w)
                                      - (int)pad_left;

            // Highest absolute input row currently resident in line_buf for
            // THIS (chunk, ict, ow_tile).  Initialise so the first oh in
            // this chunk triggers a fresh kh-row load.  Per (chunk, ict)
            // line_buf is reloaded once per ow_tile (different iw range);
            // within an ow_tile, rows are retained across m_groups (mg>0
            // sees last_loaded_row already at the chunk's max and Phase 1
            // loads nothing).
            int last_loaded_row =
                (int)(oh_start * stride_h) - (int)pad_top - 1;

          for (unsigned mg = 0; mg < num_m_groups; mg++) {
            for (unsigned oh = oh_start; oh < oh_end; oh++) {
                const int ih_window_max = (int)(oh * stride_h)
                                        - (int)pad_top
                                        + (int)((kh - 1) * dilation_h);

                // -------------------------------------------------------
                // Phase 1: load any rows the current (oh, ict) window
                // needs that are not yet in line_buf.  Only kTileIC
                // channels are loaded here (ic_off..ic_off+ic_valid-1),
                // and only this ow_tile's iw range (clamped to [0,in_w)).
                // -------------------------------------------------------
                int load_start = last_loaded_row + 1;
                if (load_start < 0) load_start = 0;
                int load_end = ih_window_max;
                if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                // Clip the ow_tile's iw range to valid input columns.
                int iw_clipped_start = iw_load_start;
                if (iw_clipped_start < 0) iw_clipped_start = 0;
                int iw_clipped_last  = iw_load_last;
                if (iw_clipped_last >= (int)in_w)
                    iw_clipped_last = (int)in_w - 1;

                for (int ih = load_start; ih <= load_end; ih++) {
                    const unsigned slot =
                        (unsigned)ih & (kMaxLineBufRows - 1);
                    for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                        const unsigned c     = ic_off + ic_l;
                        const unsigned x_row = (ni * in_ch + c) * in_hw
                                             + (unsigned)ih * in_w;
                        for (int iw = iw_clipped_start;
                             iw <= iw_clipped_last; iw++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr = x_row + (unsigned)iw;
                            const unsigned col_slot =
                                (unsigned)iw & (kMaxLineBufCols - 1);
                            line_buf[ic_l][slot][col_slot] = x[addr];

#ifdef DEBUG_LOAD_DATA_CACHING
                            CycleCounters counters;
                            counters.mt   = 0;
                            counters.ni   = ni;
                            counters.ict  = ict;
                            counters.ic_l = ic_l;
                            counters.oh   = oh;
                            counters.ow   = owt;
                            counters.khi  = (unsigned)ih;
                            counters.kwi  = (unsigned)iw;
                            read_addresses[addr].push_back(counters);
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
                if (load_end > last_loaded_row) {
                    last_loaded_row = load_end;
                }

                for (unsigned ow = ow_start; ow < ow_end; ow++) {

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
                                const unsigned col_slot = iw_ok
                                    ? ((unsigned)iw & (kMaxLineBufCols - 1))
                                    : 0u;
                                patch_stream.write((ic_ok && ih_ok && iw_ok)
                                    ? line_buf[ic_l][slot][col_slot]
                                    : Data_t(0));
                            }
                        }
                    }
                } // ow loop
            } // oh loop
          } // m_group loop
          } // ow_tile loop
        } // ict loop
      } // chunk loop
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
// Tiled-M (Option-A) + ow-tiling.  Loop nest:
//
//     for ni: for chunk: for mt: for ow_tile: for oh_in_chunk:
//         for ow_in_tile: for m1, kh, kw
//
// line_buf is sized [kTileM][kMaxLineBufRows][kMaxLineBufCols] with
// circular indexing on BOTH dims (row_slot = ih & (kMaxLineBufRows-1),
// col_slot = iw & (kMaxLineBufCols-1)).  Within an ow_tile each x pixel
// in the tile's iw range is fetched from DDR exactly once per
// (ni, c, chunk); at ow_tile transitions the (kw-1)*dilation_w-col
// overlap is re-fetched, and at chunk transitions the
// (kh-1)*stride_h-row overlap is re-fetched.
//
// Per (ni, chunk, mt, ow_tile, oh):
//   Phase 1 — load new rows × this ow_tile's iw range from DDR for the
//             m_valid channels at offset m_off..m_off+m_valid-1.
// Per (ni, chunk, mt, ow_tile, oh, ow):
//   Phase 2 — stream a fixed kTileM × kh × kw block of patch values into
//             patch_pipe.  Lanes m1 >= m_valid are zero-padded so the
//             broadcaster operates with a compile-time-fixed
//             input_per_iter (= kTileM*kh*kw).
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

    unsigned oh_per_chunk, num_chunks;
    compute_oh_chunking(out_h, out_w, out_ch, oh_per_chunk, num_chunks);

    unsigned ow_per_tile, num_ow_tiles;
    compute_ow_tiling(out_w, kw, stride_w, dilation_w,
                      ow_per_tile, num_ow_tiles);

    Data_t line_buf[kTileM][kMaxLineBufRows][kMaxLineBufCols];

#ifdef DEBUG_LOAD_DATA_CACHING
    AddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start = chunk * oh_per_chunk;
        const unsigned oh_end   = std::min(out_h, oh_start + oh_per_chunk);

        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);

          for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
            const unsigned ow_start = owt * ow_per_tile;
            const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile);

            const int iw_load_start = (int)(ow_start * stride_w) - (int)pad_left;
            const int iw_load_last  = (int)((ow_end - 1) * stride_w
                                            + (kw - 1) * dilation_w)
                                      - (int)pad_left;

            // Highest absolute input row currently resident in line_buf for
            // THIS (chunk, mt, ow_tile).  Reset per ow_tile (different iw
            // range invalidates the cached row × tile_col data).
            int last_loaded_row =
                (int)(oh_start * stride_h) - (int)pad_top - 1;

            for (unsigned oh = oh_start; oh < oh_end; oh++) {
                const int ih_window_max = (int)(oh * stride_h)
                                        - (int)pad_top
                                        + (int)((kh - 1) * dilation_h);

                // ----- Phase 1: load new rows × tile iw range -----
                int load_start = last_loaded_row + 1;
                if (load_start < 0) load_start = 0;
                int load_end = ih_window_max;
                if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                int iw_clipped_start = iw_load_start;
                if (iw_clipped_start < 0) iw_clipped_start = 0;
                int iw_clipped_last  = iw_load_last;
                if (iw_clipped_last >= (int)in_w)
                    iw_clipped_last = (int)in_w - 1;

                for (int ih = load_start; ih <= load_end; ih++) {
                    const unsigned slot =
                        (unsigned)ih & (kMaxLineBufRows - 1);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        const unsigned c     = m_off + m1;
                        const unsigned x_row = (ni * in_ch + c) * in_hw
                                             + (unsigned)ih * in_w;
                        for (int iw = iw_clipped_start;
                             iw <= iw_clipped_last; iw++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr = x_row + (unsigned)iw;
                            const unsigned col_slot =
                                (unsigned)iw & (kMaxLineBufCols - 1);
                            line_buf[m1][slot][col_slot] = x[addr];

#ifdef DEBUG_LOAD_DATA_CACHING
                            CycleCounters counters;
                            counters.mt   = mt;
                            counters.ni   = ni;
                            counters.ict  = -1;
                            counters.ic_l = m1;
                            counters.oh   = oh;
                            counters.ow   = owt;
                            counters.khi  = (unsigned)ih;
                            counters.kwi  = (unsigned)iw;
                            read_addresses[addr].push_back(counters);
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
                if (load_end > last_loaded_row) {
                    last_loaded_row = load_end;
                }

                for (unsigned ow = ow_start; ow < ow_end; ow++) {
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
                                const unsigned col_slot = iw_ok
                                    ? ((unsigned)iw & (kMaxLineBufCols - 1))
                                    : 0u;
                                patch_stream.write((m_ok && ih_ok && iw_ok)
                                    ? line_buf[m1][slot][col_slot]
                                    : Data_t(0));
                            }
                        }
                    }
                }
            } // oh loop
          } // ow_tile loop
        } // mt loop
      } // chunk loop
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
// stream_load_weights — DATAFLOW source for the weight m_axi port.
//
// Streams weights from DDR (gmem1) to process_conv_kernel_tile via
// weight_stream in the exact order the consumer reads them.  The win is
// not fewer DDR transactions but overlap: the weight loads now run
// concurrently with bias_producer, input_patch_producer,
// broadcast_patches, and the consumer's accumulate / partial_outputs
// passes, instead of serialising inside the (oh, ow, mt) inner nest.
//
// Standard path (is_depthwise=0):
//   Iteration order matches process_conv_kernel_tile's Phase 2a nest with
//   M-grouping + ow-tiling — (ni, chunk, ict, ow_tile, mg, mt_in_group) —
//   so per (ict, ow_tile, mg) the producer emits the full
//   (mt_per_group_actual × m_valid × ic_valid × kh × kw) slab of weights
//   in (mt_in_group, m1, ic_l, khi, kwi) order (kwi fastest).  Each m1
//   stripe is read from a contiguous DDR region (m_axi infers bursts).
//   Weights are emitted ONCE per (ni, chunk, ict, ow_tile, mg) — no
//   spatial replay across (oh, ow_in_tile).  When out_w fits one ow_tile
//   AND m_tiles ≤ kMaxMperGroup each weight is read from DDR exactly once
//   per (ni, chunk); otherwise replay multiplies by num_ow_tiles ×
//   num_m_groups (still tiny compared to the pre-caching out_h·out_w
//   replay).
//
// Depthwise path (is_depthwise=1):
//   Iteration order matches Phase 2b's once-per-(chunk, mt) hoist —
//   (ni, chunk, mt) — so per (ni, chunk, mt) the producer emits m_valid *
//   kh * kw values in (m1, khi, kwi) order.  Weights are re-fetched from
//   DDR num_chunks times per (ni, mt); the per-mt slice is tiny
//   (kTileM·kh·kw values) so this overhead is negligible.
// ---------------------------------------------------------------------------
static void stream_load_weights(
    const Data_t*           weight,
    hls::stream<Data_t>&    weight_stream,
    unsigned ic_tiles,
    unsigned m_tiles,
    unsigned in_ch,
    unsigned out_ch,
    unsigned out_w,
    unsigned out_h,
    unsigned kw,
    unsigned kh,
    unsigned stride_w,
    unsigned dilation_w,
    unsigned batch,
    unsigned is_depthwise
)
{
    unsigned oh_per_chunk, num_chunks;
    compute_oh_chunking(out_h, out_w, out_ch, oh_per_chunk, num_chunks);
    (void)oh_per_chunk;  // Weight emission no longer depends on (oh, ow); only chunk count.

    unsigned mt_per_group, num_m_groups;
    compute_m_grouping(out_ch, mt_per_group, num_m_groups);

    unsigned ow_per_tile, num_ow_tiles;
    compute_ow_tiling(out_w, kw, stride_w, dilation_w,
                      ow_per_tile, num_ow_tiles);
    (void)ow_per_tile;

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        if (!is_depthwise) {
            // ---- Standard: once per (ni, chunk, ict, ow_tile, mg) ----
            for (unsigned ict = 0; ict < ic_tiles; ict++) {
                const unsigned ic_off   = ict * kTileIC;
                const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

              for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
                for (unsigned mg = 0; mg < num_m_groups; mg++) {
                    const unsigned mt_base = mg * mt_per_group;
                    const unsigned mt_in_group_count =
                        (mt_base + mt_per_group <= m_tiles)
                            ? mt_per_group
                            : (m_tiles - mt_base);

                    // Push mt_in_group_count tile-slices in
                    // (mt_in_group, m1, ic_l, khi, kwi) order — kwi fastest.
                    for (unsigned mt_in_group = 0;
                         mt_in_group < mt_in_group_count; mt_in_group++) {
                        const unsigned mt     = mt_base + mt_in_group;
                        const unsigned m_off  = mt * kTileM;
                        const unsigned m_valid =
                            std::min(kTileM, out_ch - m_off);

                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            const Data_t* w_ptr = weight
                                + (m_off + m1) * in_ch * kh * kw
                                + ic_off * kh * kw;
                            const unsigned wt_len = ic_valid * kh * kw;
                            for (unsigned r = 0; r < wt_len; r++) {
                                #pragma HLS PIPELINE II=1
                                weight_stream.write(w_ptr[r]);
                            }
                        }
                    }
                } // mg
              } // ow_tile
            } // ict
        } else {
            // -------- Depthwise: once per (ni, chunk, mt) --------
            for (unsigned mt = 0; mt < m_tiles; mt++) {
                const unsigned m_off   = mt * kTileM;
                const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                // Push m_valid * kh * kw values in (m1, khi, kwi) order.
                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                    const Data_t* w_ptr = weight + (m_off + m1) * kh * kw;
                    const unsigned wt_len = kh * kw;
                    for (unsigned r = 0; r < wt_len; r++) {
                        #pragma HLS PIPELINE II=1
                        weight_stream.write(w_ptr[r]);
                    }
                }
            } // mt
        } // depthwise
      } // chunk
    } // batch
}

// ---------------------------------------------------------------------------
// process_conv_kernel_tile — DATAFLOW consumer.
//
// Output is processed in chunks along the oh axis (see compute_oh_chunking).
// Each chunk holds chunk_oh_count·out_w·out_ch accumulators in
// partial_outputs[] and runs the full Option-A three-phase pipeline for its
// sub-range:
//
//   Per (ni, chunk):
//     Phase 1 (init):  drain chunk_oh_count*out_w*out_ch bias values into
//                      partial_outputs[] (BRAM-resident), indexed by
//                      oh_local = oh - oh_start.
//     Phase 2 (accum): standard   — for ict OUTER, (oh_in_chunk, ow, mt)
//                                   inner; load patch[kTileIC][kh][kw],
//                                   weight[kTileM][kTileIC][kh][kw],
//                                   reduce ic_valid*kh*kw*kTileM at II=1.
//                      depthwise  — for mt OUTER, (oh_in_chunk, ow) inner;
//                                   load patch[kTileM][kh][kw],
//                                   load w_buf[kTileM][kh][kw] ONCE per
//                                   (chunk, mt), reduce kh*kw*kTileM at II=1.
//     Phase 3 (drain): push partial_outputs to acc_stream in
//                      (oh_in_chunk, ow, mt, m1) order.  Concatenated across
//                      chunks this is (ni, oh, ow, mt, m1) — exactly what
//                      write_output_tile expects.
//
//   Both producers read each x pixel from DDR once per (ni, c, chunk):
//   line_buf is retained across oh WITHIN a (chunk, ict|mt) but reloads
//   the (kh-1)-row overlap at chunk boundaries.
//
// Memory constraint: out_w*out_ch <= kMaxAccPersistEntries  (one row fits).
// ---------------------------------------------------------------------------
static void process_conv_kernel_tile(
    hls::stream<Data_t>&    patch_stream,
    hls::stream<Data_t>&    weight_stream,
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

    unsigned oh_per_chunk, num_chunks;
    compute_oh_chunking(out_h, out_w, out_ch, oh_per_chunk, num_chunks);

    AccData_t partial_outputs[kMaxAccPersistEntries];

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start        = chunk * oh_per_chunk;
        const unsigned oh_end          = std::min(out_h,
                                                  oh_start + oh_per_chunk);
        const unsigned chunk_oh_count  = oh_end - oh_start;

        // -------- Phase 1: init partial_outputs from bias_stream --------
        for (unsigned oh_local = 0; oh_local < chunk_oh_count; oh_local++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = (oh_local * out_w + ow) * out_ch
                                             + m_off + m1;
                        partial_outputs[idx] = bias_stream.read();
                    }
                }
            }
        }

        // -------- Phase 2a: standard accumulate (ict OUTER, ow_tile, m_group) --------
        // Weight caching: per (ict, ow_tile, m_group) the mt_per_group_actual
        // mt-tiles' worth of weights are streamed into w_cache ONCE and reused
        // across the ow_tile's (oh_local, ow_in_tile) sweep — no per-(oh, ow)
        // replay.  Patches are streamed once per (ow_tile, m_group, oh_local,
        // ow_in_tile); within the m_group's mt_in_group loop the same patch
        // is reused with cached weights for each output-channel tile.
        if (!is_depthwise) {
            unsigned mt_per_group, num_m_groups;
            compute_m_grouping(out_ch, mt_per_group, num_m_groups);

            unsigned ow_per_tile, num_ow_tiles;
            compute_ow_tiling(out_w, kw, stride_w, dilation_w,
                              ow_per_tile, num_ow_tiles);

            for (unsigned ict = 0; ict < ic_tiles; ict++) {
                const unsigned ic_off   = ict * kTileIC;
                const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

              for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
                const unsigned ow_start = owt * ow_per_tile;
                const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile);

              for (unsigned mg = 0; mg < num_m_groups; mg++) {
                const unsigned mt_base = mg * mt_per_group;
                const unsigned mt_in_group_count =
                    (mt_base + mt_per_group <= m_tiles)
                        ? mt_per_group
                        : (m_tiles - mt_base);

                // ---- Load the m_group's weight slab from weight_stream ----
                // w_cache[mt_in_group][m1][ic_l][khi][kwi].  Partition dim=3
                // (ic_l) complete → kTileIC parallel banks so the inlined
                // accumulate_standard's PN-wide adder tree gets one read
                // per bank per cycle.
                Data_t w_cache[kMaxMperGroup][kTileM][kTileIC]
                              [kMaxKH][kMaxKW];
                #pragma HLS ARRAY_PARTITION variable=w_cache complete dim=3

                for (unsigned mt_in_group = 0;
                     mt_in_group < mt_in_group_count; mt_in_group++) {
                    const unsigned mt     = mt_base + mt_in_group;
                    const unsigned m_off  = mt * kTileM;
                    const unsigned m_valid =
                        std::min(kTileM, out_ch - m_off);

                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        for (unsigned ic_l = 0; ic_l < ic_valid; ic_l++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                for (unsigned kwi = 0; kwi < kw; kwi++) {
                                    #pragma HLS PIPELINE II=1
                                    w_cache[mt_in_group][m1][ic_l][khi][kwi]
                                        = weight_stream.read();
                                }
                            }
                        }
                    }
                }

                // ---- Spatial sweep: read patch ONCE per (oh, ow_in_tile),
                //      then iterate mt_in_group against the cached weights ----
                for (unsigned oh_local = 0; oh_local < chunk_oh_count;
                     oh_local++) {
                    for (unsigned ow = ow_start; ow < ow_end; ow++) {
                        Data_t patch[kTileIC][kMaxKH][kMaxKW];
                        #pragma HLS ARRAY_PARTITION variable=patch complete dim=0

                        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                for (unsigned kwi = 0; kwi < kw; kwi++) {
                                    #pragma HLS PIPELINE II=1
                                    patch_stream.read(patch[ic_l][khi][kwi]);
                                }
                            }
                        }

                        for (unsigned mt_in_group = 0;
                             mt_in_group < mt_in_group_count; mt_in_group++) {
                            const unsigned mt     = mt_base + mt_in_group;
                            const unsigned m_off  = mt * kTileM;
                            const unsigned m_valid =
                                std::min(kTileM, out_ch - m_off);

                            AccData_t acc[kTileM];
                            #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

                            const unsigned idx_base =
                                (oh_local * out_w + ow) * out_ch + m_off;
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[m1] = AccData_t(0);
                            }
                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                acc[m1] = partial_outputs[idx_base + m1];
                            }

                            accumulate_standard(
                                patch, w_cache[mt_in_group], acc,
                                ic_valid, kh, kw);

                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                partial_outputs[idx_base + m1] = acc[m1];
                            }
                        }
                    }
                }
              } // mg
              } // ow_tile
            } // ict
        } else {
            // -------- Phase 2b: depthwise accumulate (mt OUTER, ow_tile) --------
            unsigned ow_per_tile_dw, num_ow_tiles_dw;
            compute_ow_tiling(out_w, kw, stride_w, dilation_w,
                              ow_per_tile_dw, num_ow_tiles_dw);

            for (unsigned mt = 0; mt < m_tiles; mt++) {
                const unsigned m_off   = mt * kTileM;
                const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                // Load weights ONCE per (chunk, mt) (held in BRAM across all
                // ow_tiles and the (oh, ow_in_tile) sweep).  ow-tiling here
                // doesn't add weight DDR replay — weight slice is small and
                // shared across the full ow sweep.
                Data_t w_buf[kTileM][kMaxKH][kMaxKW];
                // PM-wide read: accumulate_depthwise unrolls
                // m1 = 0..kTileM-1 every cycle, so the channel dim of
                // w_buf must give kTileM parallel banks.
                #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                    for (unsigned khi = 0; khi < kh; khi++) {
                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                            #pragma HLS PIPELINE II=1
                            w_buf[m1][khi][kwi] = weight_stream.read();
                        }
                    }
                }

              for (unsigned owt = 0; owt < num_ow_tiles_dw; owt++) {
                const unsigned ow_start = owt * ow_per_tile_dw;
                const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile_dw);

                for (unsigned oh_local = 0; oh_local < chunk_oh_count;
                     oh_local++) {
                    for (unsigned ow = ow_start; ow < ow_end; ow++) {
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

                        const unsigned idx_base =
                            (oh_local * out_w + ow) * out_ch + m_off;
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
              } // ow_tile
            } // mt
        } // depthwise

        // -------- Phase 3: drain partial_outputs to acc_stream --------
        for (unsigned oh_local = 0; oh_local < chunk_oh_count; oh_local++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    const unsigned m_off   = mt * kTileM;
                    const unsigned m_valid = std::min(kTileM, out_ch - m_off);
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx = (oh_local * out_w + ow) * out_ch
                                             + m_off + m1;
                        acc_stream.write(partial_outputs[idx]);
                    }
                }
            }
        }
      } // chunk
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
    //
    // Per-port burst and outstanding-transaction tuning:
    //   max_{read,write}_burst_length=256  — maximum AXI burst beats (AMBA
    //                                        allows 256), so a fully widened
    //                                        bus moves up to one 4 KB page per
    //                                        burst, amortising address-channel
    //                                        latency.
    //   num_{read,write}_outstanding=8     — the m_axi adapter can have up to
    //                                        8 in-flight bursts before
    //                                        stalling, hiding DDR round-trip
    //                                        latency under DATAFLOW.
    //   m_axi_max_widen_bitwidth is set globally in scripts/Synthesis.tcl.in
    //   via AXI_BUS_WIDTH (lets users dial it back to match a 128-bit block
    //   design); not duplicated per-port so the global stays authoritative.
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

    #pragma HLS STABLE variable=x
    #pragma HLS STABLE variable=weight
    #pragma HLS STABLE variable=bias

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

    unsigned mt_per_group, num_m_groups;
    compute_m_grouping(out_ch, mt_per_group, num_m_groups);
    (void)mt_per_group;

    // Total patch emissions equal the consumer's read count.  ow-tiling
    // changes ordering but not totals — chunk_oh_count*ow_per_tile summed
    // across (chunk, ow_tile) = out_h*out_w.  So broadcast_iters formulae
    // are unchanged.
    //
    //   Standard:  kTileIC*kh*kw per (ni, ict, ow_tile, mg, oh, ow_in_tile);
    //              producer replays patches num_m_groups times per
    //              (ict, ow_tile, oh, ow_in_tile); passthrough (factor=1).
    //   Depthwise: kTileM *kh*kw per (ni, mt, ow_tile, oh, ow_in_tile);
    //              passthrough.
    const unsigned input_per_iter   = is_depthwise
        ? (kTileM  * kh * kw)
        : (kTileIC * kh * kw);
    const unsigned broadcast_iters  = is_depthwise
        ? (batch * m_tiles  * out_h * out_w)
        : (batch * ic_tiles * num_m_groups * out_h * out_w);
    const unsigned broadcast_factor = 1u;

    hls::stream<AccData_t> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    // patch_pipe carries the producer's emissions; patch_stream is the
    // broadcaster's passthrough to the consumer (broadcast_factor=1 now,
    // since the standard producer already replays per m_group).  patch_pipe
    // depth is one ic-tile's patch (kTileIC*kh*kw) — the broadcaster drains
    // it as the assembler fills it under DATAFLOW.
    hls::stream<Data_t> patch_pipe;
    #pragma HLS STREAM variable=patch_pipe depth=kTileIC*kMaxKH*kMaxKW

    hls::stream<Data_t> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=kTileIC

    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kTileM

    // weight_stream carries one Data_t per cycle from stream_load_weights
    // to process_conv_kernel_tile.  Depth is one full max-tile (kTileM *
    // kTileIC * kMaxKH * kMaxKW = 6272 at defaults) so the producer can
    // pre-fetch the next iteration's weight slice while the consumer is
    // still in accumulate — full producer/consumer overlap.
    hls::stream<Data_t> weight_stream;
    #pragma HLS STREAM variable=weight_stream depth=kTileM*kTileIC*kMaxKH*kMaxKW

    bias_producer(bias, bias_stream,
                  out_ch, bias_rep_count, has_bias);

    input_patch_producer(x, patch_pipe, batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise
    );

    broadcast_patches(patch_pipe, patch_stream,
                      broadcast_iters, input_per_iter, broadcast_factor);

    stream_load_weights(weight, weight_stream,
                        ic_tiles, m_tiles, in_ch, out_ch, out_w, out_h,
                        kw, kh, stride_w, dilation_w, batch, is_depthwise);

    process_conv_kernel_tile(
        patch_stream, weight_stream, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch);
}

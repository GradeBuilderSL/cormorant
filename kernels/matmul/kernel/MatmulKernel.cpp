// ---------------------------------------------------------------------------
// MatmulKernel.cpp — tiled matrix multiplication with persistent-A reuse
//                    and grouped K processing (cache-style A reload).
//
// Architecture:
//
//   batch loop                            — pointer offset only
//     n_block loop      (kBlockN rows)      — outer; cache-miss boundary on N
//       m_tile loop     (kTileM cols)
//         CLEAR acc[kBlockN][kTileM]
//         k_chunk loop  (kChunkK columns of A per chunk) — cache-miss on K
//           (re)LOAD a_buf if this chunk isn't already cached
//           k_tile loop (kTileK rows within the chunk)
//             LOAD b_tile (DDR; global K offset)
//             n_grp loop — II=1 lane-rotated K-reduce, a_buf indexed
//                          by chunk-local k offset
//         WRITE C        (after every k_chunk for this m_tile has accumulated)
//
// Caching model:
//
//   The on-chip a_buf[kBlockN][kChunkK] is treated as a cache for A.  Each
//   (n_block, k_chunk) pair names one cacheable working set.  The kernel
//   tracks the currently-resident chunk in `last_loaded_k_chunk`:
//
//     - Cache hit:  same chunk as last iteration → skip the DDR load.
//                   This is the common path: when K ≤ kChunkK the kernel
//                   has k_chunks = 1, so after m_tile 0 every later m_tile
//                   hits the cache and A is loaded exactly once per n_block
//                   (identical traffic to the previous persistent-A code).
//
//     - Cache miss: different chunk needed → reload from DDR.  Happens
//                   on every k_chunk boundary when K > kChunkK, and on
//                   every n_block boundary (cache is invalidated then
//                   since the resident rows belong to the previous block).
//
//   This is the same grouped-tile / duplicate-readings pattern already in
//   use on the N axis (the outer n_block loop has always reloaded A when
//   N > kBlockN).  Adding the K-axis chunk loop generalises the same model
//   in both directions, lifts the previous *hard* `K ≤ kChunkK` runtime
//   limit (which previously corrupted memory silently on overflow), and
//   leaves kBlockN / kChunkK as on-chip cache *sizes* rather than scheduler-
//   enforced workload bounds.  The price is m_tiles × k_chunks DDR loads
//   of A when k_chunks > 1, instead of one — accepted in exchange for
//   handling arbitrary K.
//
// II=1 strategy (unchanged):
//
//   Inside each n_grp the ki counter runs k_tile_valid·kTileN times.
//   n1 = ki % kTileN rotates the row lane; kk = ki / kTileN advances the
//   K index within the chunk.  The same acc[n_idx][m1] register is
//   written every kTileN cycles — distance enough for the ap_fixed MAC
//   latency (≈3) so the inner pipeline schedules at II=1.  Crossing
//   k_tile, k_chunk, and n_grp boundaries forces a short pipeline drain;
//   negligible vs the inner-loop body.
//
// On-chip buffers:
//
//   a_buf [kBlockN ][kChunkK ]  cyclic factor=kTileN on dim 1
//                            → kTileN banks, each (kBlockN/kTileN)·kChunkK deep.
//                              Holds the currently-cached chunk of A.
//   b_tile[kTileK][kTileM]  complete on dim 2 → kTileM column banks
//                            for the unrolled m1 inner loop.
//   acc   [kBlockN ][kTileM]  complete on dim 0 → kBlockN·kTileM registers.
//                            Cleared per m_tile, accumulates across every
//                            k_chunk and k_tile, written to C once all
//                            K contributions have been folded in.
// ---------------------------------------------------------------------------

#include <algorithm>
#include "MatmulKernel.h"

void MatmulKernel(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      n,
    unsigned      k,
    unsigned      m,
    unsigned      batch,
    unsigned      a_batch_stride,
    unsigned      b_batch_stride,
    unsigned      c_batch_stride
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    // -----------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 depth=MATMUL_COSIM_DEPTH_A
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 depth=MATMUL_COSIM_DEPTH_B
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 depth=MATMUL_COSIM_DEPTH_C
    #pragma HLS INTERFACE s_axilite port=a              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=n              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=k              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=m              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=a_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return         bundle=ctrl

    // -----------------------------------------------------------------------
    // On-chip buffers (see header comment for partitioning rationale).
    // -----------------------------------------------------------------------
    static Data_t    a_buf [kBlockN][kChunkK];
    static Data_t    b_tile[kTileK][kTileM];
    static AccData_t acc   [kBlockN][kTileM];

    #pragma HLS ARRAY_PARTITION variable=a_buf  cyclic factor=kTileN dim=1
    #pragma HLS ARRAY_PARTITION variable=b_tile complete            dim=2
    // acc storage: partition only on the m1 lane dim (dim 2) — each m1 lane
    // becomes its own dual-port BRAM bank of depth kBlockN.  The runtime
    // n_idx index becomes an address into that bank instead of a kBlockN-way
    // register MUX (which previously dominated the LUT cost, ≈5.7 k MUX LUTs
    // out of the kernel's 23 k total).  With ram_t2p (true dual port) HLS
    // schedules the R-M-W per cycle using one port for the read and the
    // other for the write, leaving each acc element's WAW distance at
    // kTileN cycles — the lane-rotation guarantee — so II=1 still holds.
    #pragma HLS ARRAY_PARTITION variable=acc    complete            dim=2
    #pragma HLS BIND_STORAGE variable=acc type=ram_t2p impl=bram

    // -----------------------------------------------------------------------
    // Batch loop — stride=0 on a or b means that pointer stays fixed.
    // -----------------------------------------------------------------------
    for (unsigned bi = 0; bi < batch; bi++) {
        const Data_t* a_ptr = a + bi * a_batch_stride;
        const Data_t* b_ptr = b + bi * b_batch_stride;
        Data_t*       c_ptr = c + bi * c_batch_stride;

        // -------------------------------------------------------------------
        // N-block loop (outer cache axis on N).  Cache is invalidated at
        // every block boundary because the resident rows of a_buf belong
        // to the previous block.
        // -------------------------------------------------------------------
        const unsigned n_blocks = (n + kBlockN - 1) / kBlockN;
        for (unsigned nb = 0; nb < n_blocks; nb++) {
            const unsigned n_block_off   = nb * kBlockN;
            const unsigned n_block_valid =
                std::min(unsigned(kBlockN), n - n_block_off);
            const unsigned n_grps =
                (n_block_valid + kTileN - 1) / kTileN;

            // K-axis chunking is enumerated by the inner k_chunk loop.
            const unsigned m_tiles  = (m + kTileM - 1) / kTileM;
            const unsigned k_chunks = (k + kChunkK - 1) / kChunkK;

            // last_loaded_k_chunk tracks which chunk of A is currently
            // resident in a_buf.  Initialised to UINT_MAX so the first
            // (m_tile=0, k_chunk=0) always misses.  Resets at every
            // n_block boundary (declared inside the n_block loop so the
            // value is re-initialised on entry).
            unsigned last_loaded_k_chunk = (unsigned)-1;

            // ---------------------------------------------------------------
            // M-tile loop.
            // ---------------------------------------------------------------
            for (unsigned m_tile = 0; m_tile < m_tiles; m_tile++) {
                const unsigned m_off   = m_tile * kTileM;
                const unsigned m_valid = std::min(unsigned(kTileM), m - m_off);

                // Clear acc — kBlockN rows × kTileM cols.  Now that acc is
                // a per-m1-bank BRAM (1 W port each), the unrolled clear of
                // kBlockN positions per bank in one cycle is no longer
                // possible.  Serialise across n1c (one row per cycle) and
                // unroll only the m1c dim, so each cycle issues exactly one
                // write per bank.  Cost: kBlockN cycles per m_tile vs the
                // prior 1 cycle — negligible (kBlockN=16, m_tiles ≤ 3 in
                // shipping tests, so ≤ 48 extra cycles total).
                for (unsigned n1c = 0; n1c < kBlockN; n1c++) {
                    #pragma HLS PIPELINE II=1
                    for (unsigned m1c = 0; m1c < kTileM; m1c++) {
                        #pragma HLS UNROLL
                        acc[n1c][m1c] = AccData_t(0);
                    }
                }

                // -----------------------------------------------------------
                // K-chunk loop (outer cache axis on K).
                // -----------------------------------------------------------
                for (unsigned k_chunk = 0; k_chunk < k_chunks; k_chunk++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    const unsigned k_chunk_off   = k_chunk * kChunkK;
                    const unsigned k_chunk_valid =
                        std::min(unsigned(kChunkK), k - k_chunk_off);

                    // -------------------------------------------------------
                    // Cache check.  Reload A only if the resident chunk
                    // differs from the one we need.
                    //
                    //   k_chunks = 1  (K ≤ kChunkK): m_tile 0 loads, every
                    //                              later m_tile hits — one
                    //                              load per n_block, same
                    //                              traffic as the previous
                    //                              persistent-A code.
                    //   k_chunks > 1 (K > kChunkK):  each m_tile reloads
                    //                              every chunk in turn —
                    //                              m_tiles × k_chunks
                    //                              loads per n_block.
                    // -------------------------------------------------------
                    if (last_loaded_k_chunk != k_chunk) {
                        for (unsigned n1 = 0; n1 < n_block_valid; n1++) {
                            for (unsigned ki = 0; ki < k_chunk_valid; ki++) {
                                #pragma HLS PIPELINE II=1
                                a_buf[n1][ki] =
                                    a_ptr[(n_block_off + n1) * k +
                                          (k_chunk_off + ki)];
                            }
                        }
                        last_loaded_k_chunk = k_chunk;
                    }

                    // -------------------------------------------------------
                    // K-tile loop within this chunk.  k_tile_off is chunk-
                    // local (used to index a_buf); k_off_global combines
                    // chunk and tile to address B in DDR.
                    // -------------------------------------------------------
                    const unsigned k_tiles_in_chunk =
                        (k_chunk_valid + kTileK - 1) / kTileK;
                    for (unsigned k_tile = 0;
                         k_tile < k_tiles_in_chunk; k_tile++) {
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=(kChunkK/kTileK)
                        const unsigned k_tile_off   = k_tile * kTileK;
                        const unsigned k_tile_valid =
                            std::min(unsigned(kTileK),
                                     k_chunk_valid - k_tile_off);
                        const unsigned k_off_global =
                            k_chunk_off + k_tile_off;

                        // ---------------------------------------------------
                        // Load b_tile (DDR, global K offset).
                        // ---------------------------------------------------
                        for (unsigned k1 = 0; k1 < k_tile_valid; k1++) {
                            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                                #pragma HLS PIPELINE II=1
                                b_tile[k1][m1] =
                                    b_ptr[(k_off_global + k1) * m +
                                          (m_off + m1)];
                            }
                        }

                        // ---------------------------------------------------
                        // K-reduce, n_grp-wrapped.  a_buf is indexed by
                        // (n_idx, k_tile_off + kk) — k_tile_off is the
                        // chunk-local tile offset, kk is the within-tile
                        // K index.  acc[n_idx][m1] persists across every
                        // k_tile, k_chunk and n_grp within this m_tile.
                        // ---------------------------------------------------
                        for (unsigned n_grp = 0; n_grp < n_grps; n_grp++) {
                            #pragma HLS LOOP_TRIPCOUNT min=1 max=(kBlockN/kTileN)
                            const unsigned ki_bound = k_tile_valid * kTileN;
                            for (unsigned ki = 0; ki < ki_bound; ki++) {
                                #pragma HLS PIPELINE II=1
                                const unsigned n1    = ki % kTileN;
                                const unsigned kk    = ki / kTileN;
                                const unsigned n_idx = n_grp * kTileN + n1;
                                const Data_t   a_val =
                                    a_buf[n_idx][k_tile_off + kk];
                                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                    #pragma HLS UNROLL
                                    acc[n_idx][m1] +=
                                        AccData_t(a_val) *
                                        AccData_t(b_tile[kk][m1]);
                                }
                            }
                        }
                    }
                }

                // -----------------------------------------------------------
                // Write C — acc has now folded in every (k_chunk, k_tile)
                // contribution for this m_tile.
                // -----------------------------------------------------------
                for (unsigned n1 = 0; n1 < n_block_valid; n1++) {
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        c_ptr[(n_block_off + n1) * m + (m_off + m1)] =
                            saturate_cast<Data_t>(acc[n1][m1]);
                    }
                }
            }
        }
    }
}

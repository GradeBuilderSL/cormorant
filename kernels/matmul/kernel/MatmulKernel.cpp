// ---------------------------------------------------------------------------
// MatmulKernel.cpp — tiled matrix multiplication with persistent-A reuse.
//
// Architecture (post-loop-swap, persistent-A):
//
//   batch loop                          — pointer offset only; no buffer reset
//     n_block loop      (kMaxN rows)    — outer; A loaded ONCE per block
//       LOAD a_buf      (n_block_valid burst reads on gmem0)
//       m_tile loop     (kTileM cols)
//         CLEAR acc[kMaxN][kTileM]
//         k_tile loop   (kTileK rows)
//           LOAD b_tile (k_valid burst reads on gmem1) — ONCE, reused across N
//           n_grp loop  (kTileN rows per group, n_grps = ceil(n_block/kTileN))
//             K-REDUCE  (II=1, lane-rotated ki sweep)
//         WRITE C       (n_block_valid burst writes on gmem2)
//
// Why this is faster than the previous shipped version (single sequential
// nest with n_tile OUTSIDE the m_tile loop):
//
//   Prior layout reloaded B[k_tile_size × m_tile_size] from DDR once per
//   (n_tile, m_tile, k_tile) — n_tile_count × duplicate reads.  This version
//   issues the B-tile DDR read once per (m_tile, k_tile) and broadcasts the
//   tile across all rows in the n-block via the n_grp wrapper around the
//   II=1 K-reduction.  Compute cycles are identical (same total MAC count,
//   same kTileN-cycle lane rotation), the saving is purely DDR traffic on
//   tests whose N spans more than one kTileN block (N > kTileN).
//
//   For tests with N ≤ kTileN the kernel issues exactly the same DDR
//   accesses as before (n_grps = 1; outer n_block iterates once) — no
//   regression on small-N geometries.
//
// II=1 strategy (unchanged from the prior version, just wrapped):
//
//   Inside each n_grp iteration, the ki counter runs k_valid·kTileN times.
//   n1 = ki % kTileN rotates the row lane; kk = ki / kTileN advances the
//   K index.  The same acc[n_idx][m1] register is therefore written every
//   kTileN cycles — distance enough to cover the ap_fixed multiply latency
//   (≈3) so HLS schedules the inner pipeline at II=1.  Crossing n_grp
//   boundaries forces a small pipeline drain (≈ pipeline depth) per group;
//   negligible vs the k_valid·kTileN body.
//
// On-chip buffers (declared `static` so HLS infers BRAM):
//
//   a_buf [kMaxN ][kMaxK ]  cyclic factor=kTileN on dim 1
//                            → kTileN banks, each (kMaxN/kTileN)·kMaxK deep.
//                              Bank = n_idx % kTileN = n1 (compile-time in
//                              the inner ki sweep); within-bank address =
//                              (n_idx / kTileN, k_off+kk) = (n_grp, k_off+kk).
//                              Single-port BRAM is sufficient: each cycle
//                              touches one bank at a runtime-but-constant
//                              within-bank position.
//   b_tile[kTileK][kTileM]  complete on dim 2
//                            → kTileM column banks for the unrolled m1 loop.
//   acc   [kMaxN ][kTileM]  complete on dim 0
//                            → kMaxN·kTileM individual registers.  Runtime
//                              n_idx becomes a kMaxN-way read MUX + decoder
//                              per (n_grp, n1) write.  The cyclic partition
//                              tried first synthesised as depth-(kMaxN/kTileN)
//                              register-array banks whose R-M-W port arbitration
//                              forced II=2 (HLS 200-885 at line 227); complete
//                              partitioning leaves the lane-rotation WAW
//                              distance at kTileN cycles per register — enough
//                              for the ap_fixed MAC pipeline, so II=1 schedules.
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
    //
    // Three m_axi ports keep A, B, and C reads/writes on separate AXI buses
    // so the tool can issue them concurrently.  All scalar arguments go into
    // the s_axilite ctrl register file accessed by the PS driver.
    //
    // depth=<N> is a C/RTL co-simulation hint only — it sizes the cosim
    // verification adapter FIFO per m_axi port and does NOT constrain the
    // synthesised AXI master or the exported IP.  The MATMUL_COSIM_DEPTH_*
    // macros (MatmulKernel.h) are the single source of truth; cosim of an
    // m_axi kernel aborts without a depth specification.
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
    // On-chip buffers (see header comment for the partitioning rationale).
    // -----------------------------------------------------------------------
    static Data_t    a_buf [kMaxN][kMaxK];
    static Data_t    b_tile[kTileK][kTileM];
    static AccData_t acc   [kMaxN][kTileM];

    #pragma HLS ARRAY_PARTITION variable=a_buf  cyclic factor=kTileN dim=1
    #pragma HLS ARRAY_PARTITION variable=b_tile complete            dim=2
    #pragma HLS ARRAY_PARTITION variable=acc    complete            dim=0

    // -----------------------------------------------------------------------
    // Batch loop — stride=0 on a or b means that pointer stays fixed (broadcasts).
    // -----------------------------------------------------------------------
    for (unsigned bi = 0; bi < batch; bi++) {
        const Data_t* a_ptr = a + bi * a_batch_stride;
        const Data_t* b_ptr = b + bi * b_batch_stride;
        Data_t*       c_ptr = c + bi * c_batch_stride;

        // -------------------------------------------------------------------
        // N-block loop — process up to kMaxN rows per outer iteration.
        //
        // For all current behavior-test geometries (N ≤ 12) this loop runs
        // exactly once: A is loaded into BRAM once per kernel invocation and
        // the whole (m_tile × k_tile × n_grp) inner nest reuses it.  When
        // N > kMaxN the outer loop reloads A for each block — equivalent to
        // the prior code's n_tile reload, just at coarser granularity.
        // -------------------------------------------------------------------
        const unsigned n_blocks = (n + kMaxN - 1) / kMaxN;
        for (unsigned nb = 0; nb < n_blocks; nb++) {
            const unsigned n_block_off   = nb * kMaxN;
            const unsigned n_block_valid =
                std::min(unsigned(kMaxN), n - n_block_off);
            const unsigned n_grps =
                (n_block_valid + kTileN - 1) / kTileN;

            // ---------------------------------------------------------------
            // Load a_buf: n_block_valid rows × k columns from A.
            // One burst per row; the inner ki loop pipelines at II=1 for
            // back-to-back AXI beats.  Rows ≥ n_block_valid retain stale
            // data — they accumulate into acc lanes that the C-write loop
            // skips, so the staleness is invisible.
            // ---------------------------------------------------------------
            for (unsigned n1 = 0; n1 < n_block_valid; n1++) {
                for (unsigned ki = 0; ki < k; ki++) {
                    #pragma HLS PIPELINE II=1
                    a_buf[n1][ki] = a_ptr[(n_block_off + n1) * k + ki];
                }
            }

            // ---------------------------------------------------------------
            // M-tile loop — process kTileM output columns per iteration.
            // ---------------------------------------------------------------
            const unsigned m_tiles = (m + kTileM - 1) / kTileM;
            for (unsigned m_tile = 0; m_tile < m_tiles; m_tile++) {
                const unsigned m_off   = m_tile * kTileM;
                const unsigned m_valid = std::min(unsigned(kTileM), m - m_off);

                // Clear acc[*][*] unconditionally — kMaxN × kTileM registers
                // wiped in one cycle by the fully-unrolled loop.  Clearing
                // the full kMaxN range (not just n_block_valid) costs the
                // same and immunises against stale data from a previous
                // invocation in case kMaxN shrinks between calls.
                for (unsigned n1c = 0; n1c < kMaxN; n1c++) {
                    #pragma HLS UNROLL
                    for (unsigned m1c = 0; m1c < kTileM; m1c++) {
                        #pragma HLS UNROLL
                        acc[n1c][m1c] = AccData_t(0);
                    }
                }

                // -----------------------------------------------------------
                // K-tile loop — accumulate one TILE_K slice of K per pass.
                // -----------------------------------------------------------
                const unsigned k_tiles = (k + kTileK - 1) / kTileK;
                for (unsigned k_tile = 0; k_tile < k_tiles; k_tile++) {
                    const unsigned k_off   = k_tile * kTileK;
                    const unsigned k_valid = std::min(unsigned(kTileK), k - k_off);

                    // -------------------------------------------------------
                    // Load b_tile: k_valid rows × m_valid columns from B.
                    // Issued ONCE per (m_tile, k_tile); the broadcast across
                    // n_grps below is purely on-chip.  This is the central
                    // change from the prior code, which reissued the same
                    // burst once per n_tile (n_grps ≡ n_tiles).
                    // -------------------------------------------------------
                    for (unsigned k1 = 0; k1 < k_valid; k1++) {
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            b_tile[k1][m1] =
                                b_ptr[(k_off + k1) * m + (m_off + m1)];
                        }
                    }

                    // -------------------------------------------------------
                    // K-reduction, n_grp-wrapped.
                    //
                    // Each n_grp processes kTileN rows (n_idx = n_grp·kTileN
                    // .. n_grp·kTileN + kTileN - 1).  The ki sweep iterates
                    // k_valid·kTileN times and rotates n1 = ki%kTileN so the
                    // same acc[n_idx][m1] register is written every kTileN
                    // cycles — distance covers the ap_fixed MAC latency, so
                    // the inner pipeline schedules at II=1.
                    //
                    // For the partial last n_grp (n_block_valid not a
                    // multiple of kTileN), trailing lanes n_idx ≥
                    // n_block_valid still execute; they update acc lanes
                    // that the C-write loop skips, costing some MAC cycles
                    // but no correctness hazard.
                    //
                    // Power-of-two kTileN: ki%kTileN is a bitwise AND and
                    // ki/kTileN a right shift — no dividers in RTL.
                    // -------------------------------------------------------
                    for (unsigned n_grp = 0; n_grp < n_grps; n_grp++) {
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=(kMaxN/kTileN)
                        const unsigned ki_bound = k_valid * kTileN;
                        for (unsigned ki = 0; ki < ki_bound; ki++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned n1    = ki % kTileN;
                            const unsigned kk    = ki / kTileN;
                            const unsigned n_idx = n_grp * kTileN + n1;
                            const Data_t   a_val = a_buf[n_idx][k_off + kk];
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[n_idx][m1] +=
                                    AccData_t(a_val) *
                                    AccData_t(b_tile[kk][m1]);
                            }
                        }
                    }
                }

                // -----------------------------------------------------------
                // Write output block: saturate_cast acc → C.
                // n_block_valid sequential burst writes of m_valid elements
                // each; the inner m1 loop pipelines at II=1 for burst AXI
                // writes.
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

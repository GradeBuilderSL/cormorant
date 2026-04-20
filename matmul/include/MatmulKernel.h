#pragma once

#include "Config.h"

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts v to type T with saturation when T is an ap_fixed type; passes
// through unchanged for float / double / integer types.
//
// Identical in semantics to the version in include/VectorOP.h; duplicated
// here so the matmul/ subdirectory is self-contained and does not depend on
// the VectorOPKernel headers.
// ---------------------------------------------------------------------------

// Primary template: no saturation (float, double, integer types).
template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef MATMUL_HAVE_APFIXED
// Partial specialisation for ap_fixed<W,I,Q,O,N>:
// rounds the value through AP_TRN, AP_SAT before narrowing to the target type.
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct saturate_to<ap_fixed<W, I, Q, O, N>> {
    template<typename From>
    static ap_fixed<W, I, Q, O, N> cast(From v) {
        return ap_fixed<W, I, Q, O, N>(ap_fixed<W, I, AP_TRN, AP_SAT>(v));
    }
};
#endif

template<typename T, typename From>
inline T saturate_cast(From v) {
    return saturate_to<T>::template cast<From>(v);
}

// ---------------------------------------------------------------------------
// MatmulKernel — tiled matrix multiplication.
//
// Computes C = A × B for a batch of 2-D matrix products.  Batch broadcasting
// is encoded via stride=0: set a_batch_stride=0 to reuse A for every batch
// iteration (A broadcasts), b_batch_stride=0 to reuse B.
//
// Parameters:
//   a, b, c         Pointers to row-major matrices in DDR.
//   n               Rows of A (rows of C).
//   k               Inner dimension (cols of A = rows of B).
//                   Must be ≤ kMaxK (compile-time limit, enforced by caller).
//   m               Cols of B (cols of C).
//   batch           Total number of 2-D products to compute.
//   a_batch_stride  Elements to advance 'a' per batch step (0 = broadcasts).
//   b_batch_stride  Elements to advance 'b' per batch step (0 = broadcasts).
//   c_batch_stride  Elements to advance 'c' per batch step.
//
// Memory layout (row-major):
//   A[n][k] : a[row*k + col]
//   B[k][m] : b[row*m + col]
//   C[n][m] : c[row*m + col]
//
// AXI interface (added in HLS kernel — see MatmulKernel.cpp pragma comments):
//   a, b  → m_axi, gmem0 / gmem1  (read ports)
//   c     → m_axi, gmem2          (write port)
//   all scalars → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
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
);

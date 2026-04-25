#pragma once

#include "Config.h"

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts v to type T with saturation when T is an ap_fixed type; passes
// through unchanged for float / double / integer types.
//
// Identical in semantics to the version in conv/include/ConvKernel.h;
// duplicated here so the pool/ subdirectory is self-contained.
// ---------------------------------------------------------------------------

template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef POOL_HAVE_APFIXED
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
// PoolingKernel — 2-D pooling following ONNX semantics.
//
// Supports MaxPool, AveragePool, LpPool and their Global variants.  Global
// variants are handled by the caller passing pool_h=in_h, pool_w=in_w,
// stride_h=stride_w=1, pad_top=pad_left=0.
//
// pool_type         : 0=kPoolMax  1=kPoolAvg  2=kPoolLp
// lp_order          : 1 or 2 (used when pool_type=kPoolLp)
// count_include_pad : 0 = exclude padding pixels from average denominator
//                     1 = include  (used when pool_type=kPoolAvg only)
//
// Output dimensions are precomputed by the caller:
//   out_h = floor((in_h + pad_top + pad_bottom - dil_h*(pool_h-1) - 1) / stride_h) + 1
//   out_w = floor((in_w + pad_left + pad_right - dil_w*(pool_w-1) - 1) / stride_w) + 1
// Symmetric padding is assumed: pad_bottom = pad_top, pad_right = pad_left.
// Asymmetric trailing padding is handled implicitly by the bounds check.
//
// Memory layout (NCHW, row-major):
//   x[batch][channels][in_h ][in_w ]
//   y[batch][channels][out_h][out_w]
//
// AXI interface (in PoolingKernel.cpp):
//   x → m_axi gmem0  (read)
//   y → m_axi gmem1  (write)
//   all scalars → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
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
);

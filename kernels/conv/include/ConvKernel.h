#pragma once

#include "Config.h"

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts v to type T with saturation when T is an ap_fixed type; passes
// through unchanged for float / double / integer types.
//
// Identical in semantics to the version in matmul/include/MatmulKernel.h;
// duplicated here so the conv/ subdirectory is self-contained.
// ---------------------------------------------------------------------------

template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef CONV_HAVE_APFIXED
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
// ConvKernel — 2-D convolution following ONNX Conv semantics.
//
// Supports standard convolution (group=1) and depthwise convolution
// (group=in_ch, i.e. is_depthwise=1).
//
// Computes Y = conv(X, weight) + bias for a batch of 2-D feature maps in
// NCHW layout.  Padding is applied implicitly: any input index outside
// [0, in_h) × [0, in_w) is treated as zero.
//
// Parameters:
//   x, weight, bias, y   DDR pointers (NCHW row-major; see layout below).
//   batch                N: number of input images.
//   in_ch                C: input channels.
//   in_h, in_w           H, W: input spatial dimensions.
//   out_ch               M: output channels.
//   out_h, out_w         oH, oW: output spatial dimensions (precomputed by caller).
//   kh, kw               Kernel height/width.  Must satisfy kh ≤ kMaxKH and
//                        kw ≤ kMaxKW (compile-time limits); enforced by caller.
//   stride_h, stride_w   Convolution stride (≥ 1).
//   dilation_h, dilation_w  Kernel dilation (1 = standard convolution).
//   pad_top, pad_left    Zero-padding rows/columns before the input.
//                        pad_bottom and pad_right are implicit: out-of-bounds
//                        input accesses are zero-padded by the bounds check.
//   has_bias             0 = do not read bias pointer; 1 = add bias[m] to y.
//   is_depthwise         0 = standard conv (group=1);
//                        1 = depthwise conv (group=in_ch).
//                            weight layout changes to [out_ch][1][kh][kw].
//
// Memory layout (row-major, NCHW):
//   x     [batch][in_ch][in_h ][in_w ]
//   Standard (is_depthwise=0):
//     weight[out_ch][in_ch][kh][kw]  → (m*in_ch + c)*kh*kw + khi*kw + kwi
//   Depthwise (is_depthwise=1):
//     weight[out_ch][1][kh][kw]      → m*kh*kw + khi*kw + kwi
//   bias  [out_ch]
//   y     [batch][out_ch][out_h][out_w]
//
// AXI interface (in ConvKernel.cpp):
//   x, weight, bias → m_axi gmem0/1/2  (read ports)
//   y               → m_axi gmem3      (write port)
//   all scalars     → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
void ConvKernel(
    const Data_t *x,
    const Data_t *weight,
    const Data_t *bias,
    Data_t *y,
    unsigned batch,
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
    unsigned has_bias,
    unsigned is_depthwise);

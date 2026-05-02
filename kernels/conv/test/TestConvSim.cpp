// ---------------------------------------------------------------------------
// TestConvSim.cpp — reference tests for ConvKernel.
//
// Each test computes the same convolution with two independent implementations:
//
//   ref_conv()          — naive 7-nested-loop ground-truth oracle.
//   ref_depthwise_conv()— ground truth for depthwise (group=in_ch).
//   ConvKernel()        — tiled reference that mirrors the HLS kernel structure.
//
// Outputs are compared element-by-element with zero tolerance for fixed-point
// types (both share the same accumulator type and saturation policy), and
// relative 1e-5 tolerance for float.
//
// Test matrix (standard conv):
//   1×1 kernel, 3×3 various pads/strides, 5×5 kernel, large spatial,
//   partial TILE_IC, partial TILE_M, dilation=2, batch>1, bias,
//   exact tile multiples, asymmetric stride, asymmetric dilation,
//   1×1 output, horizontal filter, ResNet-style strided block,
//   saturation (ap_fixed only).
//
// Test matrix (depthwise conv, is_depthwise=1):
//   3×3 depthwise, 3×3 depthwise+bias, partial TILE_M, dilation=2,
//   stride=2, batch>1, exact tile multiple, 5×5 kernel,
//   asymmetric stride, saturation (ap_fixed only).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ConvKernel.h"

// ---------------------------------------------------------------------------
// Scalar limits derived via saturate_cast — works for both ap_fixed and float.
// ---------------------------------------------------------------------------
static const double kSatMax = static_cast<double>(saturate_cast<Data_t>( 1e30));
static const double kSatMin = static_cast<double>(saturate_cast<Data_t>(-1e30));

// ---------------------------------------------------------------------------
// Naive reference — standard conv ground truth (group=1).
// ---------------------------------------------------------------------------
static void ref_conv(
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
    unsigned      has_bias)
{
    for (unsigned n = 0; n < batch; n++) {
        for (unsigned m = 0; m < out_ch; m++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
                    AccData_t sum = has_bias ? AccData_t(bias[m]) : AccData_t(0);
                    for (unsigned c = 0; c < in_ch; c++) {
                        for (unsigned khi = 0; khi < kh; khi++) {
                            const int ih = (int)(oh * stride_h + khi * dilation_h)
                                         - (int)pad_top;
                            for (unsigned kwi = 0; kwi < kw; kwi++) {
                                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                             - (int)pad_left;
                                if (ih >= 0 && (unsigned)ih < in_h &&
                                    iw >= 0 && (unsigned)iw < in_w)
                                {
                                    sum += AccData_t(x[(n*in_ch+c)*in_h*in_w + ih*in_w + iw])
                                         * AccData_t(weight[(m*in_ch+c)*kh*kw + khi*kw + kwi]);
                                }
                            }
                        }
                    }
                    y[(n*out_ch+m)*out_h*out_w + oh*out_w + ow] = saturate_cast<Data_t>(sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Naive reference — depthwise conv ground truth (group=in_ch).
//
// Weight layout: [ch][1][kh][kw] → offset m*kh*kw + khi*kw + kwi
// out_ch == in_ch == ch.
// ---------------------------------------------------------------------------
static void ref_depthwise_conv(
    const Data_t* x,
    const Data_t* weight,
    const Data_t* bias,
    Data_t*       y,
    unsigned      batch,
    unsigned      ch,      // in_ch == out_ch
    unsigned      in_h,
    unsigned      in_w,
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
    unsigned      has_bias)
{
    for (unsigned n = 0; n < batch; n++) {
        for (unsigned m = 0; m < ch; m++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
                    AccData_t sum = has_bias ? AccData_t(bias[m]) : AccData_t(0);
                    for (unsigned khi = 0; khi < kh; khi++) {
                        const int ih = (int)(oh * stride_h + khi * dilation_h)
                                     - (int)pad_top;
                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                            const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                         - (int)pad_left;
                            if (ih >= 0 && (unsigned)ih < in_h &&
                                iw >= 0 && (unsigned)iw < in_w)
                            {
                                sum += AccData_t(x[(n*ch+m)*in_h*in_w + ih*in_w + iw])
                                     * AccData_t(weight[m*kh*kw + khi*kw + kwi]);
                            }
                        }
                    }
                    y[(n*ch+m)*out_h*out_w + oh*out_w + ow] = saturate_cast<Data_t>(sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output spatial size helper.
// ---------------------------------------------------------------------------
static unsigned out_size(unsigned in_sz, unsigned k, unsigned stride,
                         unsigned dilation, unsigned pad_begin, unsigned pad_end)
{
    const unsigned eff_k = dilation * (k - 1) + 1;
    return (in_sz + pad_begin + pad_end - eff_k) / stride + 1;
}

// ---------------------------------------------------------------------------
// Per-element comparison.
// ---------------------------------------------------------------------------
static bool vals_close(double ref, double got)
{
#ifdef CONV_HAVE_APFIXED
    return ref == got;  // fixed-point: exact match required
#else
    if (ref == got) return true;
    const double scale = std::max({std::abs(ref), std::abs(got), 1e-6});
    return std::abs(ref - got) <= 1e-5 * scale;
#endif
}

// ---------------------------------------------------------------------------
// Run one test case.  Returns number of mismatches.
// ---------------------------------------------------------------------------
struct ConvParams {
    unsigned batch, in_ch, in_h, in_w;
    unsigned out_ch;
    unsigned kh, kw;
    unsigned stride_h, stride_w;
    unsigned dilation_h, dilation_w;
    unsigned pad_top, pad_left, pad_bottom, pad_right;
    bool     has_bias;
    bool     is_depthwise;
};

static int run_test(const char* name, const ConvParams& p,
                    const std::vector<Data_t>& x_data,
                    const std::vector<Data_t>& w_data,
                    const std::vector<Data_t>& b_data)
{
    const unsigned out_h = out_size(p.in_h, p.kh, p.stride_h, p.dilation_h,
                                    p.pad_top,  p.pad_bottom);
    const unsigned out_w = out_size(p.in_w, p.kw, p.stride_w, p.dilation_w,
                                    p.pad_left, p.pad_right);

    const unsigned y_size = p.batch * p.out_ch * out_h * out_w;
    std::vector<Data_t> y_ref(y_size, Data_t(0));
    std::vector<Data_t> y_got(y_size, Data_t(0));

    if (p.is_depthwise) {
        ref_depthwise_conv(x_data.data(), w_data.data(),
                           p.has_bias ? b_data.data() : nullptr,
                           y_ref.data(),
                           p.batch, p.in_ch, p.in_h, p.in_w,
                           out_h, out_w,
                           p.kh, p.kw,
                           p.stride_h, p.stride_w,
                           p.dilation_h, p.dilation_w,
                           p.pad_top, p.pad_left,
                           p.has_bias ? 1u : 0u);
    } else {
        ref_conv(x_data.data(), w_data.data(),
                 p.has_bias ? b_data.data() : nullptr,
                 y_ref.data(),
                 p.batch, p.in_ch, p.in_h, p.in_w,
                 p.out_ch, out_h, out_w,
                 p.kh, p.kw,
                 p.stride_h, p.stride_w,
                 p.dilation_h, p.dilation_w,
                 p.pad_top, p.pad_left,
                 p.has_bias ? 1u : 0u);
    }

    ConvKernel(x_data.data(), w_data.data(),
               b_data.data(),  // always a valid pointer (kernel guards by has_bias)
               y_got.data(),
               p.batch, p.in_ch, p.in_h, p.in_w,
               p.out_ch, out_h, out_w,
               p.kh, p.kw,
               p.stride_h, p.stride_w,
               p.dilation_h, p.dilation_w,
               p.pad_top, p.pad_left,
               p.has_bias ? 1u : 0u,
               p.is_depthwise ? 1u : 0u);

    int mismatches = 0;
    for (unsigned i = 0; i < y_size; i++) {
        const double r = static_cast<double>(y_ref[i]);
        const double g = static_cast<double>(y_got[i]);
        if (!vals_close(r, g)) {
            if (mismatches < 5) {
                printf("  [%u] ref=%.6f got=%.6f\n", i, r, g);
            }
            mismatches++;
        }
    }

    const char* status = (mismatches == 0) ? "PASS" : "FAIL";
    printf("%-55s %s", name, status);
    if (mismatches > 0) printf("  (%d mismatches)", mismatches);
    printf("  [%s batch=%u C=%u H=%u W=%u M=%u kH=%u kW=%u s=%u,%u d=%u,%u p=%u,%u out=%ux%u]\n",
           p.is_depthwise ? "DW" : "STD",
           p.batch, p.in_ch, p.in_h, p.in_w, p.out_ch,
           p.kh, p.kw, p.stride_h, p.stride_w,
           p.dilation_h, p.dilation_w, p.pad_top, p.pad_left,
           out_h, out_w);
    return mismatches;
}

// ---------------------------------------------------------------------------
// Fill vector with random values in [-scale, +scale].
// ---------------------------------------------------------------------------
template<typename T>
static std::vector<T> rand_vec(unsigned n, float scale, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<T> v(n);
    for (auto& e : v) e = T(dist(rng));
    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::mt19937 rng(kSeed);
    int total_failures = 0;

    printf("ConvKernel simulation tests\n");
    printf("Data_t    = %s\n", sizeof(Data_t) == 2 ? "ap_fixed<16,8>" : "float");
    printf("TILE_M=%u  TILE_IC=%u  MAX_KH=%u  MAX_KW=%u\n",
           kTileM, kTileIC, kMaxKH, kMaxKW);
    printf("------------------------------------------------------------------\n");

    // -----------------------------------------------------------------------
    // Test 1: 1×1 kernel, single channel, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 2.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    2.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x1 kernel, 1ch, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 1: 1×1 kernel, single channel, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 2.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    2.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x1 kernel, 1ch, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 2: 3×3 kernel, no padding, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, no pad, no bias → 3x3 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 3: 3×3 kernel, pad=1 (same-size output)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, pad=1 (same) → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 4: 3×3 kernel, stride=2, no padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=6; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, stride=2 → 2x2 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 5: 3×3 kernel with bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=2;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("3x3, pad=1, has_bias, 2 out_ch", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 6: Batch = 2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=2; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("batch=2, 3x3, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 7: Partial input channel tile (in_ch = TILE_IC + 5)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC+5; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("partial IC tile (in_ch=TILE_IC+5)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 8: Partial output channel tile (out_ch = TILE_M + 3)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=kTileM+3;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("partial M tile (out_ch=TILE_M+3)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 9: Dilation = 2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=9; p.in_w=9; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=2; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3 dilation=2 → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 10: 5×5 kernel
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=7; p.out_ch=1;
        p.kh=5; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("5x5 kernel → 3x3 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 11: Multiple tiles in all dimensions (14×14 feature map)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC*2+3; p.in_h=14; p.in_w=14; p.out_ch=kTileM*2+1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.25f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.25f, rng);
        total_failures += run_test("14x14, multi-tile M and IC, bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 12: Non-square input and kernel
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=8; p.out_ch=2;
        p.kh=3; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("non-square: 6x8 input, 3x5 kernel", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 13: Mixed stride and padding (asymmetric)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=7; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("7x7, stride=2, asymmetric pad [1,1,0,0]", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 21: 1×1 kernel, exact IC and M tile multiples (no partial tile)
    // Pure channel projection; exercises the IC reduction with two full tiles.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC*2; p.in_h=5; p.in_w=5; p.out_ch=kTileM*2;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.1f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("1x1, IC=TILE_IC*2 M=TILE_M*2 bias (exact tiles)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 22: Asymmetric strides (stride_h=2, stride_w=1) → 2×4 output
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=6; p.out_ch=2;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, asymmetric stride h=2 w=1 → 2x4 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 23: Asymmetric dilation (dilation_h=1, dilation_w=2) → 5×5 output
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=9; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, asymmetric dilation h=1 w=2 → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 24: Single-pixel output (3×3 input, 3×3 kernel, no padding)
    // Exercises the oh/ow loops iterating exactly once.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC; p.in_h=3; p.in_w=3; p.out_ch=kTileM;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f,  rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("3x3 input 3x3 kernel → 1x1 out, C=TILE_IC M=TILE_M", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 25: 1×5 horizontal filter, same-width padding (pad_left=pad_right=2)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=8; p.out_ch=2;
        p.kh=1; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=2; p.pad_bottom=0; p.pad_right=2;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x5 horizontal filter, pad_left=pad_right=2", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 26: batch=3, C=TILE_IC, M=TILE_M, stride=2, pad=1, bias
    // Models a ResNet-style strided block; verifies batch iteration across tiles.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=3; p.in_ch=kTileIC; p.in_h=7; p.in_w=7; p.out_ch=kTileM;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f,  rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("batch=3, C=TILE_IC M=TILE_M stride=2 (ResNet-style)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Depthwise tests (is_depthwise=1).
    // Weight layout: [ch][1][kh][kw]  (no in_ch dimension in weight)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Test 14 (DW): 3×3 depthwise, 4 channels, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=8; p.in_w=8; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3, 4ch, no pad, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 15 (DW): 3×3 depthwise + bias, same padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=8; p.in_w=8; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3, 4ch, pad=1, has_bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 16 (DW): Partial TILE_M — channels = TILE_M + 3
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileM+3; p.in_h=6; p.in_w=6; p.out_ch=kTileM+3;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW partial M tile (ch=TILE_M+3)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 17 (DW): Depthwise with dilation=2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=9; p.in_w=9; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=2; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3 dilation=2, 4ch", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 18 (DW): Stride=2 depthwise
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=8; p.in_w=8; p.out_ch=8;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW stride=2, 8ch, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 27 (DW): batch=2, 3×3 depthwise, same padding, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=2; p.in_ch=4; p.in_h=6; p.in_w=6; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW batch=2, 4ch, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28 (DW): ch=TILE_M*2 — exact tile multiple, no partial last tile
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileM*2; p.in_h=6; p.in_w=6; p.out_ch=kTileM*2;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW ch=TILE_M*2 (exact tile), bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 29 (DW): 5×5 depthwise kernel, 4 channels, no padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=9; p.in_w=9; p.out_ch=4;
        p.kh=5; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.25f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.25f, rng);
        total_failures += run_test("DW 5x5 kernel, 4ch → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 30 (DW): Asymmetric stride (stride_h=2, stride_w=1), 8 channels
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=8; p.in_w=8; p.out_ch=8;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW asymmetric stride h=2 w=1, 8ch → 4x8 out", p, x, w, b);
    }

#ifdef CONV_HAVE_APFIXED
    // -----------------------------------------------------------------------
    // Test 19: Saturation — positive overflow
    // All weights = +1, all inputs = kSatMax; kernel_size=1, C=1
    // Bias = kSatMax; output must saturate at kSatMax.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        const unsigned n_w = p.out_ch*p.in_ch*p.kh*p.kw;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(n_w, Data_t(1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMax));
        total_failures += run_test("saturation: positive overflow → AP_MAX", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 20: Saturation — negative overflow
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        const unsigned n_w = p.out_ch*p.in_ch*p.kh*p.kw;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(n_w, Data_t(-1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMin));
        total_failures += run_test("saturation: negative overflow → AP_MIN", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 31: Depthwise saturation — positive overflow
    // weight=+1, input=kSatMax, bias=kSatMax → output must clamp at kSatMax.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=true;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(p.out_ch*1*p.kh*p.kw, Data_t(1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMax));
        total_failures += run_test("DW saturation: positive overflow → AP_MAX", p, x, w, b);
    }
#endif

    printf("------------------------------------------------------------------\n");
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILED: %d element mismatch(es) across all tests\n", total_failures);
    }
    return total_failures == 0 ? 0 : 1;
}

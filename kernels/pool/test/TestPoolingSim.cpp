// ---------------------------------------------------------------------------
// TestPoolingSim.cpp — C++ simulation tests for PoolingKernel.
//
// Verifies all three pool types (MAX, AVG, LP) against a pure-float reference
// implementation for a range of geometries, including padding, dilation,
// batched inputs, multi-tile channel counts, and Global* variants.
//
// Build and run via CMake:
//   cmake pool/   &&  make TestPoolingSim  &&  ./TestPoolingSim
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "PoolingKernel.h"

// Tolerance: ~5 LSBs for ap_fixed<16,8> (1 LSB = 1/256 ≈ 0.0039).
// LP p=2 uses sqrtf internally, which adds at most 1 ULP of additional error.
static constexpr float kTol = 0.02f;

static float to_float(Data_t v)    { return float(v); }
static Data_t from_float(float v)  { return Data_t(v); }

// ---------------------------------------------------------------------------
// Reference pooling — pure float / double, no quantisation inside.
// ---------------------------------------------------------------------------
static float ref_pool_elem(const std::vector<Data_t>& x,
                            int N, int C, int H, int W,
                            int n, int c, int oh, int ow,
                            int pool_h, int pool_w,
                            int stride_h, int stride_w,
                            int pad_top,  int pad_left,
                            int dil_h,    int dil_w,
                            int pool_type, int lp_order,
                            int count_include_pad)
{
    double acc = 0.0;
    int valid_count = 0;

    if (pool_type == 0) acc = -1.0e30; // MAX sentinel

    for (int khi = 0; khi < pool_h; khi++) {
        for (int kwi = 0; kwi < pool_w; kwi++) {
            int ih = oh * stride_h + khi * dil_h - pad_top;
            int iw = ow * stride_w + kwi * dil_w - pad_left;
            bool valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
            if (valid) {
                double v = to_float(x[(n * C + c) * H * W + ih * W + iw]);
                if (pool_type == 0) {          // MAX
                    acc = std::max(acc, v);
                } else if (pool_type == 1) {   // AVG
                    acc += v;
                } else {                        // LP
                    acc += (lp_order == 1) ? std::abs(v) : v * v;
                }
                valid_count++;
            }
        }
    }

    double result;
    if (pool_type == 0) {
        result = acc;
    } else if (pool_type == 1) {
        double denom = count_include_pad ? (pool_h * pool_w) : valid_count;
        result = (denom > 0) ? acc / denom : 0.0;
    } else {
        result = (lp_order == 1) ? acc : std::sqrt(acc);
    }

    // Clamp to ap_fixed<16,8> range [-128, 127.996] to match saturate_cast.
    return (float)std::max(-128.0, std::min(127.996, result));
}

// ---------------------------------------------------------------------------
// Run one test case.
// ---------------------------------------------------------------------------
struct TC {
    const char* name;
    int N, C, H, W;
    int out_h, out_w;
    int pool_h, pool_w;
    int stride_h, stride_w;
    int pad_top,  pad_left;
    int dil_h,    dil_w;
    int pool_type;        // 0=MAX  1=AVG  2=LP
    int lp_order;         // 1 or 2 (LP only)
    int count_include_pad; // 0 or 1 (AVG only)
};

static bool run_test(const TC& tc)
{
    const int in_size  = tc.N * tc.C * tc.H * tc.W;
    const int out_size = tc.N * tc.C * tc.out_h * tc.out_w;

    std::vector<Data_t> x(in_size);
    std::vector<Data_t> y(out_size, Data_t(0));

    // Deterministic fill: values in [-4, 4] in steps of 0.1
    for (int i = 0; i < in_size; i++) {
        int v = (int)((unsigned)(kSeed * 1103515245u + (unsigned)i * 12345u) >> 16u) % 81;
        x[i] = from_float((v - 40) * 0.1f);
    }

    PoolingKernel(
        x.data(), y.data(),
        (unsigned)tc.N,    (unsigned)tc.C,
        (unsigned)tc.H,    (unsigned)tc.W,
        (unsigned)tc.out_h,(unsigned)tc.out_w,
        (unsigned)tc.pool_h,(unsigned)tc.pool_w,
        (unsigned)tc.stride_h,(unsigned)tc.stride_w,
        (unsigned)tc.pad_top, (unsigned)tc.pad_left,
        (unsigned)tc.dil_h,   (unsigned)tc.dil_w,
        (unsigned)tc.pool_type,
        (unsigned)tc.lp_order,
        (unsigned)tc.count_include_pad
    );

    int failures = 0;
    for (int n = 0; n < tc.N; n++) {
        for (int c = 0; c < tc.C; c++) {
            for (int oh = 0; oh < tc.out_h; oh++) {
                for (int ow = 0; ow < tc.out_w; ow++) {
                    float ref = ref_pool_elem(
                        x, tc.N, tc.C, tc.H, tc.W,
                        n, c, oh, ow,
                        tc.pool_h, tc.pool_w,
                        tc.stride_h, tc.stride_w,
                        tc.pad_top, tc.pad_left,
                        tc.dil_h, tc.dil_w,
                        tc.pool_type, tc.lp_order,
                        tc.count_include_pad);
                    float got = to_float(y[(n * tc.C + c) * tc.out_h * tc.out_w
                                           + oh * tc.out_w + ow]);
                    if (std::abs(ref - got) > kTol) {
                        if (failures < 4) {
                            printf("    FAIL [n=%d,c=%d,oh=%d,ow=%d]: "
                                   "ref=%.4f  got=%.4f  diff=%.4f\n",
                                   n, c, oh, ow, ref, got, std::abs(ref - got));
                        }
                        failures++;
                    }
                }
            }
        }
    }

    const char* status = (failures == 0) ? "PASS" : "FAIL";
    printf("  [%s] %-45s  failures=%d/%d\n", status, tc.name, failures, out_size);
    return failures == 0;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
int main()
{
    const TC tests[] = {
        // --- MaxPool ---
        // {name, N,C,H,W, out_h,out_w, pool_h,pool_w, stride_h,stride_w,
        //  pad_top,pad_left, dil_h,dil_w, pool_type, lp_order, count_include_pad}
        {"MaxPool 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool 3x3 stride1 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 0,0,0},
        {"MaxPool 2x2 stride2 rect 6x10",
                1,8,6,10, 3,5, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool batch=3 2x2 stride2",
                3,4,6,6, 3,3, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool channels>kTileC (C=16)",
                1,16,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool dilation=2 pool2x2",
                1,4,8,8, 6,6, 2,2, 1,1, 0,0, 2,2, 0,0,0},
        // Global MaxPool: pool_h=in_h, pool_w=in_w, stride=1, pad=0
        {"GlobalMaxPool 4x4",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 0,0,0},
        {"GlobalMaxPool batch=2 C=12 6x6",
                2,12,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 0,0,0},

        // --- AveragePool ---
        {"AvgPool 2x2 stride2 no_pad",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 1,0,0},
        {"AvgPool 3x3 stride1 pad1 no_include",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        {"AvgPool 3x3 stride1 pad1 include_pad",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 1,1,0},
        {"AvgPool channels>kTileC (C=12)",
                1,12,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 1,0,0},
        // Global AveragePool
        {"GlobalAvgPool 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 1,0,0},
        {"GlobalAvgPool batch=2 C=16 6x6",
                2,16,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 1,0,0},
        // AvgPool asymmetric spatial (H≠W)
        {"AvgPool rect 6x10 2x2 stride2",
                1,4,6,10, 3,5, 2,2, 2,2, 0,0, 1,1, 1,0,0},

        // --- LpPool ---
        {"LpPool p=1 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 2,1,0},
        {"LpPool p=2 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 2,2,0},
        {"LpPool p=1 3x3 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 2,1,0},
        {"LpPool p=2 3x3 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 2,2,0},
        // Global LpPool
        {"GlobalLpPool p=1 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 2,1,0},
        {"GlobalLpPool p=2 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 2,2,0},
        {"GlobalLpPool p=2 batch=2 C=16 6x6",
                2,16,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 2,2,0},

        // --- Edge cases ---
        // 1x1 output (global-like, single output per channel)
        {"MaxPool 1x1 pool full 5x5",
                1,4,5,5, 1,1, 5,5, 1,1, 0,0, 1,1, 0,0,0},
        // All-padded corner: 3x3 pool, pad=1, on a 2x2 input → corner pixel
        // sees only 1 valid neighbour
        {"AvgPool corner padding 3x3 pad1 on 2x2",
                1,2,2,2, 2,2, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        // Large channel count to stress tiling (C=32 = 4 tiles of kTileC=8)
        {"MaxPool C=32 2x2 stride2",
                1,32,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
    };

    const int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    printf("PoolingKernel simulation tests\n");
    printf("  Data_t    = %s\n", sizeof(Data_t) == 4 ? "float" : "ap_fixed<16,8>");
    printf("  kTileC    = %u\n", kTileC);
    printf("  kMaxPoolH = %u\n", kMaxPoolH);
    printf("  kMaxPoolW = %u\n", kMaxPoolW);
    printf("  tolerance = %.4f\n\n", kTol);

    for (int i = 0; i < n_tests; i++) {
        if (run_test(tests[i])) passed++;
    }

    printf("\n%d / %d tests passed.\n", passed, n_tests);
    return (passed == n_tests) ? 0 : 1;
}

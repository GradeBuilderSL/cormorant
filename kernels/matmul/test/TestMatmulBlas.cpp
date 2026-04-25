// ---------------------------------------------------------------------------
// TestMatmulBlas.cpp — validates MatmulKernel (float build) against cblas_sgemm.
//
// Both implementations receive identical float input arrays.  Outputs are
// compared with an element-wise absolute tolerance derived from floating-point
// error analysis for a length-K dot product:
//
//   atol(i) = k * 8 * FLT_EPSILON * max(|ref[i]|, |got[i]|, 1e-6f)
//
// The factor of 8 gives headroom for BLAS implementations that may reorder
// operations or use FMA instructions, producing slightly different rounding.
// For inputs drawn from [-1,1] and typical K values this bound is never
// exceeded in practice; failures indicate real disagreement.
//
// cblas_sgemm call convention (row-major, no transpose):
//   C[n×m] = alpha * A[n×k] * B[k×m] + beta * C[n×m]
//   α=1, β=0  →  C = A × B
//   lda = k,  ldb = m,  ldc = m
//
// Test matrix:
//   2D shapes  : 1×1×1, tile-exact, partial-tile (N/M/K independently and
//                all together), arbitrary small, unit N, unit M, multi-tile
//   Large K    : N=8, K=512, M=32  (stress-tests accumulation precision)
//   Batch      : no broadcast, A broadcasts, B broadcasts
// ---------------------------------------------------------------------------

#include <cblas.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "MatmulKernel.h"

// Tolerance multiplier: k * kFactor * FLT_EPSILON * |value|
static constexpr float kFactor = 8.0f;

// ---------------------------------------------------------------------------
// Element comparison with a k-dependent absolute tolerance.
// ---------------------------------------------------------------------------
static bool is_close(float ref_val, float got_val, unsigned k)
{
    const float scale = std::max({std::abs(ref_val), std::abs(got_val), 1e-6f});
    const float atol  = static_cast<float>(k) * kFactor * FLT_EPSILON * scale;
    return std::abs(ref_val - got_val) <= atol;
}

// ---------------------------------------------------------------------------
// Compare two float arrays and report the first mismatch.
// ---------------------------------------------------------------------------
static bool compare_outputs(
    const float* ref,
    const float* got,
    unsigned     count,
    unsigned     k,
    const char*  label)
{
    unsigned mismatches = 0;
    for (unsigned i = 0; i < count; i++) {
        if (!is_close(ref[i], got[i], k)) {
            if (mismatches == 0) {
                printf("  FAIL  [%u] blas=%.8f  kernel=%.8f  diff=%.3e\n",
                       i,
                       static_cast<double>(ref[i]),
                       static_cast<double>(got[i]),
                       static_cast<double>(std::abs(ref[i] - got[i])));
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("  PASS  %s\n", label);
        return true;
    }
    printf("  FAIL  %s  (%u/%u elements exceed tolerance)\n",
           label, mismatches, count);
    return false;
}

// ---------------------------------------------------------------------------
// RunTest2D — 2D matrix product via cblas_sgemm vs MatmulKernel (float).
// ---------------------------------------------------------------------------
static bool RunTest2D(const char* label, unsigned n, unsigned k, unsigned m,
                      unsigned seed = kSeed)
{
    std::vector<float> A(n * k), B(k * m);
    std::vector<float> C_blas(n * m, 0.0f);
    std::vector<float> C_kern(n * m, 0.0f);

    std::default_random_engine rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // BLAS reference: C = 1·A·B + 0·C  (row-major, no transpose)
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                static_cast<int>(n),
                static_cast<int>(m),
                static_cast<int>(k),
                1.0f,
                A.data(), static_cast<int>(k),
                B.data(), static_cast<int>(m),
                0.0f,
                C_blas.data(), static_cast<int>(m));

    // Tiled kernel
    MatmulKernel(A.data(), B.data(), C_kern.data(),
                 n, k, m,
                 /*batch=*/1,
                 /*a_stride=*/n * k, /*b_stride=*/k * m, /*c_stride=*/n * m);

    return compare_outputs(C_blas.data(), C_kern.data(), n * m, k, label);
}

// ---------------------------------------------------------------------------
// RunTestBatch — batched products via cblas_sgemm (looped) vs MatmulKernel.
// ---------------------------------------------------------------------------
static bool RunTestBatch(const char* label,
                         unsigned n, unsigned k, unsigned m,
                         unsigned batch,
                         unsigned a_stride, unsigned b_stride,
                         unsigned seed = kSeed)
{
    const unsigned a_total  = (a_stride == 0) ? n * k : batch * a_stride;
    const unsigned b_total  = (b_stride == 0) ? k * m : batch * b_stride;
    const unsigned c_stride = n * m;

    std::vector<float> A(a_total), B(b_total);
    std::vector<float> C_blas(batch * c_stride, 0.0f);
    std::vector<float> C_kern(batch * c_stride, 0.0f);

    std::default_random_engine rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // BLAS reference: one call per batch element (cblas_sgemm is not batched).
    for (unsigned bi = 0; bi < batch; bi++) {
        const float* a_ptr = A.data() + bi * a_stride;
        const float* b_ptr = B.data() + bi * b_stride;
        float*       c_ptr = C_blas.data() + bi * c_stride;

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    static_cast<int>(n),
                    static_cast<int>(m),
                    static_cast<int>(k),
                    1.0f,
                    a_ptr, static_cast<int>(k),
                    b_ptr, static_cast<int>(m),
                    0.0f,
                    c_ptr, static_cast<int>(m));
    }

    MatmulKernel(A.data(), B.data(), C_kern.data(),
                 n, k, m, batch, a_stride, b_stride, c_stride);

    return compare_outputs(C_blas.data(), C_kern.data(), batch * c_stride, k, label);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    bool all_ok = true;
    int  total  = 0;
    int  passed = 0;

    printf("MatmulKernel BLAS comparison tests (float)\n");
    printf("  Data_t=float  AccData_t=float\n");
    printf("  kTileN=%u  kTileM=%u  kTileK=%u  kMaxK=%u\n\n",
           kTileN, kTileM, kTileK, kMaxK);

    auto run = [&](bool ok) { total++; if (ok) passed++; else all_ok = false; };

    // -----------------------------------------------------------------------
    // 2-D shape tests
    // -----------------------------------------------------------------------
    printf("--- 2D shapes ---\n");

    run(RunTest2D("1x1x1",           1,           1,      1));
    run(RunTest2D("TileN x TileK x TileM",
                  kTileN,       kTileK,      kTileM));
    run(RunTest2D("(TileN+2) x TileK x TileM  [partial N]",
                  kTileN + 2,   kTileK,      kTileM));
    run(RunTest2D("TileN x (TileK+5) x TileM  [partial K]",
                  kTileN,       kTileK + 5,  kTileM));
    run(RunTest2D("TileN x TileK x (TileM+3)  [partial M]",
                  kTileN,       kTileK,      kTileM + 3));
    run(RunTest2D("(TileN+2) x (TileK+5) x (TileM+3)  [all partial]",
                  kTileN + 2,   kTileK + 5,  kTileM + 3));
    run(RunTest2D("7 x 13 x 5  [arbitrary small]",
                  7,            13,          5));
    run(RunTest2D("1 x K x M  [N=1 row vector]",
                  1,            kTileK,      kTileM));
    run(RunTest2D("N x K x 1  [M=1 column vector]",
                  kTileN,       kTileK,      1));
    run(RunTest2D("3*TileN x (2*TileK+7) x (2*TileM+1)  [multi-tile all]",
                  kTileN * 3,   kTileK * 2 + 7,  kTileM * 2 + 1));

    // -----------------------------------------------------------------------
    // Large-K test — stresses accumulation order differences.
    //
    // K=512 means 512 multiply-add operations per output element.  Reordering
    // these (as BLAS may do with SIMD) produces FP rounding differences
    // proportional to K * FLT_EPSILON.  The test tolerance is set to cover this.
    // -----------------------------------------------------------------------
    printf("\n--- Large K (precision stress) ---\n");

    run(RunTest2D("8 x 512 x 32  [large K]",    8,  512, 32));
    run(RunTest2D("16 x 1024 x 16  [K=kMaxK/2]", 16, kMaxK / 2, 16));

    // -----------------------------------------------------------------------
    // Batch tests
    // -----------------------------------------------------------------------
    printf("\n--- Batch ---\n");

    const unsigned BN = kTileN + 1, BK = kTileK / 4, BM = kTileM + 3;

    run(RunTestBatch("batch=3, no broadcast",
                     BN, BK, BM,
                     /*batch=*/3,
                     /*a_stride=*/BN * BK, /*b_stride=*/BK * BM));

    run(RunTestBatch("batch=4, A broadcasts (a_stride=0)",
                     BN, BK, BM,
                     /*batch=*/4,
                     /*a_stride=*/0, /*b_stride=*/BK * BM));

    run(RunTestBatch("batch=4, B broadcasts (b_stride=0)",
                     BN, BK, BM,
                     /*batch=*/4,
                     /*a_stride=*/BN * BK, /*b_stride=*/0));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%d/%d tests passed\n", passed, total);
    if (!all_ok) {
        printf("TestMatmulBlas FAILED\n");
        return 1;
    }
    printf("TestMatmulBlas PASSED\n");
    return 0;
}

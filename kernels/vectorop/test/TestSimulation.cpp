#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "VectorOP.h"

// ---------------------------------------------------------------------------
// --dump-data <dir> mode: instead of running VectorOPKernel and comparing,
// dump the per-test a/b/c_ref tensors to hex files (one 16-bit value per
// line, suitable for $readmemh).  A manifest.txt indexes every test with
// its geometry so an HDL testbench can load the same fixtures.  RNG state
// is shared with verify mode (same seed, same draw order).
// ---------------------------------------------------------------------------
static std::string g_dump_dir;     // empty → verify mode (default)
static int         g_test_idx = 0;
static FILE*       g_manifest = nullptr;

// ---------------------------------------------------------------------------
// Reference model
//
// ref_sat() derives its clamp range from std::numeric_limits<Data_t> so that
// it matches the kernel's saturate_cast<Data_t> for every supported type:
//
//   ap_fixed<W,I>  → limits reflect the fixed-point range; saturation clips.
//   float / double → limits are FLT_MAX / DBL_MAX; no practical saturation
//                    occurs for the inputs we use, matching the kernel's
//                    pass-through behaviour.
//
// This removes the hardcoded ap_fixed<16,8> constants from the reference and
// ensures the random-value tests are correct for any configured Data_t.
// ---------------------------------------------------------------------------

// Saturation limits for Data_t, derived via saturate_cast rather than
// std::numeric_limits.  The Vitis ap_fixed numeric_limits specialisation
// references an internal 'Type' alias that only resolves inside the HLS
// synthesis frontend; it fails to compile under regular GCC (C-sim build).
// saturate_cast<Data_t>(±1e38) gives the same result without that dependency:
//   ap_fixed<W,I,...>  →  AP_SAT clips to the representable extreme
//   float / double     →  identity cast, value stays ~1e38 (no practical clamp)
static const double kSatMax = static_cast<double>(saturate_cast<Data_t>( 1e38));
static const double kSatMin = static_cast<double>(saturate_cast<Data_t>(-1e38));

static double ref_sat(double v) {
    return std::max(kSatMin, std::min(kSatMax, v));
}

static double ref_op(Op op, double a, double b) {
    switch (op) {
        case OP_ADD:   return ref_sat(a + b);
        case OP_SUB:   return ref_sat(a - b);
        case OP_MUL:   return ref_sat(a * b);
        case OP_DIV:   return (b == 0.0) ? 0.0 : ref_sat(a / b);
        case OP_RELU:  return std::max(0.0, a);
        case OP_RELU6: return std::min(std::max(0.0, a), 6.0);
        default:       return a;
    }
}

// ---------------------------------------------------------------------------
// Dump-mode helpers (only meaningful for fixed-point builds — the HDL
// testbench reads 16-bit hex values one per line).
// ---------------------------------------------------------------------------
#ifdef VA_HAVE_APFIXED
static uint16_t data_to_raw16(const Data_t& v) {
    return static_cast<uint16_t>(v.range().to_uint());
}

static void write_hex_file(const std::string& path,
                           const std::vector<Data_t>& vec) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        std::exit(1);
    }
    for (const auto& v : vec)
        std::fprintf(f, "%04x\n", data_to_raw16(v));
    std::fclose(f);
}
#endif

static std::string sanitize_label(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-') out.push_back(c);
        else                                          out.push_back('_');
    }
    return out;
}

// Materialise c_ref (as Data_t) and write a/b/c hex files plus a manifest
// row.  Caller has already computed the reference output as doubles.
static void dump_one_case(const char* label,
                          unsigned size, unsigned op_code,
                          unsigned outer, unsigned a_inc, unsigned b_inc,
                          const std::vector<Data_t>& a,
                          const std::vector<Data_t>& b,
                          const std::vector<double>& c_ref_d) {
#ifndef VA_HAVE_APFIXED
    (void)label; (void)size; (void)op_code;
    (void)outer; (void)a_inc; (void)b_inc;
    (void)a; (void)b; (void)c_ref_d;
    std::fprintf(stderr, "--dump-data requires VA_HAVE_APFIXED build\n");
    std::exit(1);
#else
    const int idx = g_test_idx++;
    char idx_buf[16];
    std::snprintf(idx_buf, sizeof(idx_buf), "%02d", idx);
    const std::string prefix = g_dump_dir + "/test_" + idx_buf + "_";

    std::vector<Data_t> c_ref(c_ref_d.size());
    for (size_t i = 0; i < c_ref_d.size(); ++i)
        c_ref[i] = saturate_cast<Data_t>(c_ref_d[i]);

    write_hex_file(prefix + "a.hex", a);
    write_hex_file(prefix + "b.hex", b);
    write_hex_file(prefix + "c.hex", c_ref);

    std::fprintf(g_manifest, "%d %u %u %u %u %u %s\n",
                 idx, size, op_code, outer, a_inc, b_inc,
                 sanitize_label(label).c_str());
    std::printf("[DUMP] test_%02d  %-30s  size=%u op=%u outer=%u a_inc=%u b_inc=%u\n",
                idx, label, size, op_code, outer, a_inc, b_inc);
#endif
}

// ---------------------------------------------------------------------------
// RunTest — random values across a range that includes overflow.
// Tolerance: 1 LSB for ap_fixed types; relative 1e-5 for float/double.
// ---------------------------------------------------------------------------

struct Range { double lo, hi; };

static Range input_range(Op op) {
    switch (op) {
        case OP_ADD:
        case OP_SUB:  return { -80.0,  80.0 };   // sums/diffs can exceed ±128
        case OP_MUL:  return { -15.0,  15.0 };   // products can exceed ±128
        case OP_DIV:  return {   1.0,  10.0 };   // b always positive, non-zero
        default:      return { -10.0,  10.0 };
    }
}

static bool RunTest(Op op, const char* opName, unsigned size, unsigned seed) {
    std::default_random_engine rng(seed);
    Range ra = input_range(op);
    Range rb = (op == OP_DIV) ? Range{1.0, 10.0} : input_range(op);
    std::uniform_real_distribution<double> distA(ra.lo, ra.hi);
    std::uniform_real_distribution<double> distB(rb.lo, rb.hi);

    std::vector<Data_t> a(size), b(size), c(size);
    std::vector<double> c_ref(size);
    for (unsigned i = 0; i < size; ++i) {
        a[i]     = Data_t(distA(rng));
        b[i]     = Data_t(distB(rng));
        c[i]     = Data_t(0);
        c_ref[i] = ref_op(op, static_cast<double>(a[i]),
                               static_cast<double>(b[i]));
    }

    if (!g_dump_dir.empty()) {
        dump_one_case(opName, size, static_cast<unsigned>(op),
                      /*outer*/ 1u, /*a_inc*/ 0u, /*b_inc*/ 0u,
                      a, b, c_ref);
        return true;
    }

    VectorOPKernel(a.data(), b.data(), c.data(), size, static_cast<unsigned>(op), 1u, 0u, 0u);

    const bool isFloat = std::is_floating_point<Data_t>::value;
    const double absTol = 1.0 / 256.0;
    const double relTol = 1e-5;

    unsigned mismatches = 0;
    for (unsigned i = 0; i < size; ++i) {
        const double got  = static_cast<double>(c[i]);
        const double ref  = c_ref[i];
        const double diff = std::abs(got - ref);
        bool bad;
        if (isFloat) {
            const double absRef = std::abs(ref);
            bad = (absRef > 1e-9) ? (diff / absRef > relTol) : (diff > relTol);
        } else {
            bad = (diff > absTol);
        }
        if (bad) {
            ++mismatches;
            if (mismatches <= 3)
                std::cerr << "  [" << opName << "] MISMATCH at [" << i << "]: "
                          << "got=" << got << "  ref=" << ref
                          << "  a=" << static_cast<double>(a[i])
                          << "  b=" << static_cast<double>(b[i]) << "\n";
        }
    }
    return mismatches == 0;
}

// ---------------------------------------------------------------------------
// RunSatTest — saturation boundary verification with known inputs.
//
// Uses a fixed-size vector so every element receives the same constant values.
// Comparison tolerance is zero for fixed-point types (ap_fixed results are
// exact when converted to double) and 1 LSB for float (float rounding can
// introduce sub-LSB error when the result happens to land on a boundary).
//
// Two kinds of cases are tested:
//   OVERFLOW  — inputs that arithmetically exceed the representable range;
//               the output must be exactly sat_max or sat_min, not wrapped.
//   BOUNDARY  — inputs whose result lands exactly at the extreme value or
//               one LSB inside it; the output must match precisely.
// ---------------------------------------------------------------------------
static bool RunSatTest(Op op, double a_val, double b_val, const char* desc) {
    static const unsigned kSize = 8;
    static const double   kTol  = std::is_floating_point<Data_t>::value
                                  ? 1.0 / 256.0
                                  : 0.0;

    std::vector<Data_t> a(kSize, Data_t(a_val));
    std::vector<Data_t> b(kSize, Data_t(b_val));
    std::vector<Data_t> c(kSize, Data_t(0));

    // Reference uses the same saturate-cast semantics as the kernel.
    const double expected = ref_op(op,
                                   static_cast<double>(Data_t(a_val)),
                                   static_cast<double>(Data_t(b_val)));

    if (!g_dump_dir.empty()) {
        std::vector<double> c_ref(kSize, expected);
        dump_one_case(desc, kSize, static_cast<unsigned>(op),
                      /*outer*/ 1u, /*a_inc*/ 0u, /*b_inc*/ 0u,
                      a, b, c_ref);
        return true;
    }

    VectorOPKernel(a.data(), b.data(), c.data(), kSize, static_cast<unsigned>(op), 1u, 0u, 0u);

    bool ok = true;
    for (unsigned i = 0; i < kSize; ++i) {
        const double got  = static_cast<double>(c[i]);
        const double diff = std::abs(got - expected);
        if (diff > kTol) {
            if (ok)  // print header only once
                std::cerr << "  FAIL — " << desc << "\n"
                          << "    a=" << a_val << "  b=" << b_val
                          << "  expected=" << expected
                          << "  (sat_max=" << kSatMax
                          << "  sat_min=" << kSatMin << ")\n";
            std::cerr << "    c[" << i << "]=" << got
                      << "  diff=" << diff << "\n";
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Saturation test table
//
// One LSB of ap_fixed<16,8> = 1/256 = 0.00390625.
// MAX =  127.99609375 = 0x7FFF.
// MIN = -128.0        = 0x8000.
//
// Each row:  op, a, b, description
//
// OVERFLOW tests:  arithmetic result is well outside the representable range;
//                  the kernel must clamp, not wrap.
//
// BOUNDARY tests:  result lands exactly at MAX/MIN or one LSB away, probing
//                  the precise edge of the saturation logic.
//
// NO-SAT tests:    inputs that produce a result within range; confirm that
//                  clamping does NOT fire when it shouldn't.
//
// RELU/RELU6:      clipping at 0 and 6 — probed at ±1 LSB around each fence.
// ---------------------------------------------------------------------------
struct SatEntry {
    Op          op;
    double      a, b;
    const char* desc;
};

static const SatEntry kSatTests[] = {
    // ── ADD ─────────────────────────────────────────────────────────────────
    // overflow →  sat_max
    { OP_ADD,  100.0,          100.0,         "ADD  100+100=200    → sat_max" },
    { OP_ADD,   64.0,           64.0,         "ADD   64+64=128     → sat_max" },
    { OP_ADD,  127.99609375,     0.00390625,  "ADD  max+1LSB=128   → sat_max" },
    // overflow →  sat_min
    { OP_ADD, -100.0,          -100.0,        "ADD -100-100=-200   → sat_min" },
    { OP_ADD, -128.0,            -0.00390625, "ADD  min-1LSB       → sat_min" },
    // result exactly at boundary (no saturation should fire)
    { OP_ADD,   64.0,           63.99609375,  "ADD   64+63.996=max (no clip)" },
    { OP_ADD,  -64.0,          -64.0,         "ADD  -64-64=-128=min (no clip)" },

    // ── SUB ─────────────────────────────────────────────────────────────────
    // overflow →  sat_max
    { OP_SUB,  100.0,          -100.0,        "SUB  100-(-100)=200 → sat_max" },
    { OP_SUB,  127.99609375,    -0.00390625,  "SUB  max-(-1LSB)    → sat_max" },
    // overflow →  sat_min
    { OP_SUB, -100.0,           100.0,        "SUB -100-100=-200   → sat_min" },
    { OP_SUB, -128.0,             0.00390625, "SUB  min-1LSB       → sat_min" },

    // ── MUL ─────────────────────────────────────────────────────────────────
    // overflow →  sat_max  (positive × positive)
    { OP_MUL,   16.0,           16.0,         "MUL  16×16=256      → sat_max" },
    { OP_MUL,   12.0,           12.0,         "MUL  12×12=144      → sat_max" },
    // overflow →  sat_max  (negative × negative)
    { OP_MUL,  -16.0,          -16.0,         "MUL -16×-16=256     → sat_max" },
    // overflow →  sat_min  (positive × negative)
    { OP_MUL,  -16.0,           16.0,         "MUL -16×16=-256     → sat_min" },
    // result within range (no saturation should fire)
    { OP_MUL,   11.0,           11.0,         "MUL  11×11=121 (no clip)"      },

    // ── RELU  (clipping at 0) ────────────────────────────────────────────────
    // just below 0 → 0
    { OP_RELU,  -0.00390625,     0.0,         "RELU -1LSB → 0"                },
    { OP_RELU,  -1.0,            0.0,         "RELU -1.0  → 0"                },
    // exactly 0 → 0
    { OP_RELU,   0.0,            0.0,         "RELU  0    → 0"                },
    // just above 0 → pass through
    { OP_RELU,   0.00390625,     0.0,         "RELU +1LSB → +1LSB (no clip)"  },
    { OP_RELU,   3.5,            0.0,         "RELU  3.5  → 3.5 (no clip)"    },

    // ── RELU6 (clipping at 0 and 6) ─────────────────────────────────────────
    // below 0 → 0
    { OP_RELU6, -1.0,            0.0,         "RELU6 -1    → 0"               },
    { OP_RELU6, -0.00390625,     0.0,         "RELU6 -1LSB → 0"               },
    // in range [0, 6] → pass through
    { OP_RELU6,  0.0,            0.0,         "RELU6  0    → 0 (no clip)"     },
    { OP_RELU6,  0.00390625,     0.0,         "RELU6 +1LSB → +1LSB (no clip)" },
    { OP_RELU6,  3.0,            0.0,         "RELU6  3.0  → 3.0 (no clip)"   },
    { OP_RELU6,  5.99609375,     0.0,         "RELU6  6-1LSB → 6-1LSB (no clip)" },
    { OP_RELU6,  6.0,            0.0,         "RELU6  6.0  → 6.0 (no clip)"   },
    // above 6 → 6
    { OP_RELU6,  6.00390625,     0.0,         "RELU6  6+1LSB → 6"             },
    { OP_RELU6,  8.0,            0.0,         "RELU6  8    → 6"               },
    { OP_RELU6, 20.0,            0.0,         "RELU6 20    → 6"               },
};

static bool RunSatTests() {
    const unsigned n = sizeof(kSatTests) / sizeof(kSatTests[0]);
    bool allPassed = true;

    std::cout << "\n--- Saturation boundary tests (" << n << ") ---\n";
    for (unsigned i = 0; i < n; ++i) {
        const SatEntry& t = kSatTests[i];
        std::cout << "[" << (i + 1) << "/" << n << "] " << t.desc
                  << " ... " << std::flush;
        const bool ok = RunSatTest(t.op, t.a, t.b, t.desc);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        allPassed &= ok;
    }
    return allPassed;
}

// ---------------------------------------------------------------------------
// RunBroadcastTests — verify the outer/a_inc/b_inc broadcasting path.
//
// Two cases are tested:
//   a_advances: a strides through the output by a_inc per outer step;
//               b repeats at offset 0 each step (b_inc=0).
//   b_advances: b strides through the output; a repeats (a_inc=0).
// ---------------------------------------------------------------------------
static bool RunBroadcastTests() {
    bool allPassed = true;
    std::cout << "\n--- Broadcast tests (2) ---\n";

    // Case 1: a advances (a_inc=size), b repeats (b_inc=0)
    // outer=2, size=4 → c[8]; a[8] strides, b[4] repeats, OP_ADD
    {
        const unsigned outer = 2, size = 4;
        std::vector<Data_t> a = {
            Data_t(1), Data_t(2), Data_t(3), Data_t(4),
            Data_t(5), Data_t(6), Data_t(7), Data_t(8)
        };
        std::vector<Data_t> b = { Data_t(1), Data_t(1), Data_t(1), Data_t(1) };
        std::vector<Data_t> c(8, Data_t(0));
        VectorOPKernel(a.data(), b.data(), c.data(), size, OP_ADD, outer, size, 0u);

        bool ok = true;
        for (unsigned o = 0; o < outer; ++o) {
            for (unsigned i = 0; i < size; ++i) {
                const double got = static_cast<double>(c[o * size + i]);
                const double exp = ref_op(OP_ADD,
                                          static_cast<double>(a[o * size + i]),
                                          static_cast<double>(b[i]));
                if (std::abs(got - exp) > 1.0 / 256.0) {
                    std::cerr << "  a_advances FAIL c[" << o*size+i << "]: "
                              << "got=" << got << " exp=" << exp << "\n";
                    ok = false;
                }
            }
        }
        std::cout << "[1/2] a_advances outer=2 size=4 OP_ADD ... "
                  << (ok ? "PASS" : "FAIL") << "\n";
        allPassed &= ok;
    }

    // Case 2: b advances (b_inc=size), a repeats (a_inc=0)
    // outer=3, size=4 → c[12]; a[4] repeats, b[12] strides, OP_MUL
    {
        const unsigned outer = 3, size = 4;
        std::vector<Data_t> a = { Data_t(2), Data_t(2), Data_t(2), Data_t(2) };
        std::vector<Data_t> b;
        for (int i = 0; i < 12; ++i) b.push_back(Data_t(i + 1));
        std::vector<Data_t> c(12, Data_t(0));
        VectorOPKernel(a.data(), b.data(), c.data(), size, OP_MUL, outer, 0u, size);

        bool ok = true;
        for (unsigned o = 0; o < outer; ++o) {
            for (unsigned i = 0; i < size; ++i) {
                const double got = static_cast<double>(c[o * size + i]);
                const double exp = ref_op(OP_MUL,
                                          static_cast<double>(a[i]),
                                          static_cast<double>(b[o * size + i]));
                if (std::abs(got - exp) > 1.0 / 256.0) {
                    std::cerr << "  b_advances FAIL c[" << o*size+i << "]: "
                              << "got=" << got << " exp=" << exp << "\n";
                    ok = false;
                }
            }
        }
        std::cout << "[2/2] b_advances outer=3 size=4 OP_MUL ... "
                  << (ok ? "PASS" : "FAIL") << "\n";
        allPassed &= ok;
    }

    return allPassed;
}

// ---------------------------------------------------------------------------
// Ops table (used by RunAllTests and single-run mode)
// ---------------------------------------------------------------------------
struct OpEntry { Op op; const char* name; };

static const OpEntry kOps[] = {
    { OP_ADD,   "ADD"   },
    { OP_SUB,   "SUB"   },
    { OP_MUL,   "MUL"   },
    { OP_DIV,   "DIV"   },
    { OP_RELU,  "RELU"  },
    { OP_RELU6, "RELU6" },
};

// ---------------------------------------------------------------------------
// Full test suite
// ---------------------------------------------------------------------------
static bool RunAllTests() {
    const unsigned sizes[] = { 1, 8, 64, 256, 1024 };
    const unsigned nSizes  = sizeof(sizes) / sizeof(sizes[0]);
    const unsigned nOps    = sizeof(kOps)  / sizeof(kOps[0]);
    const unsigned nRandom = nSizes * nOps;

    bool allPassed = true;
    unsigned idx = 0;

    std::cout << "--- Random-value tests (" << nRandom << ") ---\n";
    for (unsigned o = 0; o < nOps; ++o) {
        for (unsigned s = 0; s < nSizes; ++s) {
            ++idx;
            std::cout << "[" << idx << "/" << nRandom << "] "
                      << kOps[o].name << "  size=" << sizes[s]
                      << " ... " << std::flush;
            const bool ok = RunTest(kOps[o].op, kOps[o].name,
                                    sizes[s], kSeed + idx);
            std::cout << (ok ? "PASS" : "FAIL") << "\n";
            allPassed &= ok;
        }
    }

    allPassed &= RunSatTests();
    if (g_dump_dir.empty()) {
        // Broadcast tests stage their own constants and call the kernel
        // inline; they're skipped in dump mode.  The HDL testbench can
        // exercise broadcasting via the random/saturation fixtures.
        allPassed &= RunBroadcastTests();
    }

    const unsigned nTotal = nRandom + sizeof(kSatTests) / sizeof(kSatTests[0]) + 2u;
    std::cout << "\n"
              << (allPassed ? "All " : "FAILED — ")
              << nTotal << " tests"
              << (allPassed ? " passed.\n" : " had errors.\n");
    return allPassed;
}

int main(int argc, char** argv) {
    // Optional --dump-data <dir>: emit per-test a/b/c_ref hex files plus a
    // manifest, then exit (no kernel run).  Otherwise: original verify mode.
    std::vector<const char*> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if ((a == "--dump-data" || a == "-d") && i + 1 < argc) {
            g_dump_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--dump-data <dir>] [op size]\n"
                      << "  No arguments: full verify suite.\n"
                      << "  --dump-data <dir>: write hex fixtures + manifest "
                         "and exit.\n"
                      << "  op size: single random test "
                         "(op: 0=ADD 1=SUB 2=MUL 3=DIV 4=RELU 5=RELU6).\n";
            return 0;
        } else {
            rest.push_back(argv[i]);
        }
    }

    if (!g_dump_dir.empty()) {
        const std::string manifest_path = g_dump_dir + "/manifest.txt";
        g_manifest = std::fopen(manifest_path.c_str(), "w");
        if (!g_manifest) {
            std::cerr << "Failed to open " << manifest_path << " for writing\n";
            return 1;
        }
        std::fprintf(g_manifest,
            "# VectorOPKernel test fixture manifest\n"
            "# idx size op outer a_inc b_inc label\n");

        std::cout << "VectorOP test data dump → " << g_dump_dir << "\n";
        const bool ok = RunAllTests();
        std::fclose(g_manifest);
        g_manifest = nullptr;
        std::cout << "Dumped " << g_test_idx << " test(s) to " << g_dump_dir << "\n";
        return ok ? 0 : 1;
    }

    if (rest.size() == 2) {
        const unsigned opCode = std::stoul(rest[0]);
        const unsigned size   = std::stoul(rest[1]);
        const unsigned nOps   = sizeof(kOps) / sizeof(kOps[0]);
        if (opCode >= nOps) {
            std::cerr << "op must be 0–" << (nOps - 1) << "\n";
            return 1;
        }
        std::cout << "Single test: op=" << kOps[opCode].name
                  << "  size=" << size << "\n";
        const bool ok = RunTest(kOps[opCode].op, kOps[opCode].name, size, kSeed);
        std::cout << (ok ? "PASS\n" : "FAIL\n");
        return ok ? 0 : 1;
    }
    if (!rest.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " [--dump-data <dir>] [op size]\n";
        return 1;
    }
    return RunAllTests() ? 0 : 1;
}

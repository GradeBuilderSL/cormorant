#include "VectorOP.h"
#include "hls_stream.h"
#ifdef __SYNTHESIS__
#include "hls_math.h"
#else
#include <cmath>
#endif

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

// hls::exp is available only under the HLS compiler; std::exp is used for
// C simulation.  Wrapping in a static inline keeps the softmax body clean.
static inline float sm_exp(float x) {
#ifdef __SYNTHESIS__
    return hls::exp(x);
#else
    return std::exp(x);
#endif
}

// ---------------------------------------------------------------------------
// Scalar compute helpers (inlined into the compute stage)
// ---------------------------------------------------------------------------

static inline Data_t sub_add(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a + b);
}

static inline Data_t sub_sub(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a - b);
}

// Full-precision multiply; saturate_cast clips the 2W-bit product to Data_t.
static inline Data_t sub_mul(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a * b);
}

// Iterative fixed-point divider; II > 1 expected for OP_DIV.
static inline Data_t sub_div(Data_t a, Data_t b) {
    if (b == Data_t(0)) return Data_t(0);
    return saturate_cast<Data_t>(a / b);
}

static inline Data_t sub_relu(Data_t a) {
    return (a < Data_t(0)) ? Data_t(0) : a;
}

static inline Data_t sub_relu6(Data_t a) {
    if (a < Data_t(0)) return Data_t(0);
    if (a > Data_t(6)) return Data_t(6);
    return a;
}

// ---------------------------------------------------------------------------
// Dataflow stages
//
// Three functions run concurrently via #pragma HLS dataflow in the top-level
// kernel.  Load stages push Data_t items into hls::streams; the compute stage
// drains both streams and pushes results; the store stage writes to DDR.
//
// Broadcast semantics are preserved: a stride==0 causes load_a to re-read the
// same 'size' elements on every outer iteration; similarly for load_b.
// For unary ops load_b never issues AXI reads — it feeds zeros into b_s so the
// compute stage always receives a balanced item count.
// ---------------------------------------------------------------------------

static void load_a(
    const Data_t*        src,
    hls::stream<Data_t>& dst,
    unsigned             outer,
    unsigned             size,
    unsigned             stride
) {
    #pragma HLS INLINE off
    for (unsigned o = 0; o < outer; ++o) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        for (unsigned i = 0; i < size; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            dst.write(src[o * stride + i]);
        }
    }
}

static void load_b(
    const Data_t*        src,
    hls::stream<Data_t>& dst,
    unsigned             outer,
    unsigned             size,
    unsigned             stride,
    unsigned             op
) {
    #pragma HLS INLINE off
    for (unsigned o = 0; o < outer; ++o) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        for (unsigned i = 0; i < size; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            // For unary ops the runtime branch prevents AXI reads on gmem1.
            dst.write((op < OP_RELU) ? src[o * stride + i] : Data_t(0));
        }
    }
}

static void compute(
    hls::stream<Data_t>& a_s,
    hls::stream<Data_t>& b_s,
    hls::stream<Data_t>& c_s,
    unsigned             outer,
    unsigned             size,
    unsigned             op
) {
    #pragma HLS INLINE off
    for (unsigned o = 0; o < outer; ++o) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        for (unsigned i = 0; i < size; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            Data_t ai = a_s.read();
            Data_t bi = b_s.read();
            Data_t ci;
            switch (op) {
                case OP_ADD:   ci = sub_add (ai, bi); break;
                case OP_SUB:   ci = sub_sub (ai, bi); break;
                case OP_MUL:   ci = sub_mul (ai, bi); break;
                case OP_DIV:   ci = sub_div (ai, bi); break;
                case OP_RELU:  ci = sub_relu (ai);    break;
                case OP_RELU6: ci = sub_relu6(ai);    break;
                default:       ci = ai;               break;
            }
            c_s.write(ci);
        }
    }
}

static void store_c(
    hls::stream<Data_t>& src,
    Data_t*              dst,
    unsigned             outer,
    unsigned             size,
    unsigned             stride
) {
    #pragma HLS INLINE off
    for (unsigned o = 0; o < outer; ++o) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        for (unsigned i = 0; i < size; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            dst[o * stride + i] = src.read();
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax kernel (axis = -1 only)
//
// Three sequential DDR passes per outer row:
//   1. find max(a[i])       — prevents exp() overflow for wide fixed-point types
//   2. sum exp(a[i]-max)    — loop-carried add; II > 1 for float, accepted
//   3. write exp(a[i]-max) / sum — II=1, no loop-carried dependency
//
// All intermediate arithmetic is done in float so hls::exp can be inlined
// efficiently regardless of Data_t (ap_fixed, half, float, …).
// ---------------------------------------------------------------------------
static void softmax_kernel(
    const Data_t* a,
    Data_t*       c,
    unsigned      outer,
    unsigned      size,
    unsigned      a_inc,
    unsigned      c_inc
) {
#pragma HLS INLINE off
    for (unsigned o = 0; o < outer; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        // Pass 1: find max for numerical stability
        float max_val = static_cast<float>(a[o * a_inc]);
        for (unsigned i = 1; i < size; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=0 max=65535
            const float v = static_cast<float>(a[o * a_inc + i]);
            if (v > max_val) max_val = v;
        }

        // Pass 2: accumulate sum of exp(x - max)
        float sum = 0.0f;
        for (unsigned i = 0; i < size; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            sum += sm_exp(static_cast<float>(a[o * a_inc + i]) - max_val);
        }
        const float inv_sum = 1.0f / sum;

        // Pass 3: write normalised output
        for (unsigned i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            const float val =
                sm_exp(static_cast<float>(a[o * a_inc + i]) - max_val) * inv_sum;
            c[o * c_inc + i] = saturate_cast<Data_t>(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Top kernel
// ---------------------------------------------------------------------------
void VectorOPKernel(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      size,
    unsigned      op,
    unsigned      outer,
    unsigned      a_inc,
    unsigned      b_inc
) {
    // max_widen_bitwidth: HLS coalesces sequential Data_t accesses into bursts
    // up to 512 bits wide (32 × ap_fixed<16,8> per beat); on platforms with a
    // narrower physical bus (e.g. KV260 128-bit HPC) the cap is applied by HLS.
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 \
        num_read_outstanding=16  max_read_burst_length=256  max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 \
        num_read_outstanding=16  max_read_burst_length=256  max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 \
        num_write_outstanding=16 max_write_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE s_axilite port=a      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=size   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=op     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=outer  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=a_inc  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b_inc  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // c_inc == 0 when outer==1 (a_inc==b_inc==0) — writes c[i] directly.
    const unsigned c_inc = a_inc + b_inc;

    // Softmax bypasses the element-wise dataflow pipeline: it requires two
    // extra DDR passes (max-find + exp-sum) per row before normalisation.
    if (op == OP_SOFTMAX) {
        softmax_kernel(a, c, outer, size, a_inc, c_inc);
        return;
    }

    // Static streams persist across synthesis elaboration; depth=32 decouples
    // load and compute so that DDR bursts can run ahead of the compute stage.
    static hls::stream<Data_t> a_s("a_s");
    static hls::stream<Data_t> b_s("b_s");
    static hls::stream<Data_t> c_s("c_s");
    #pragma HLS stream variable=a_s depth=32
    #pragma HLS stream variable=b_s depth=32
    #pragma HLS stream variable=c_s depth=32

    // Four concurrent pipeline stages; load stages overlap with compute and
    // store so that DDR read latency is hidden behind active computation.
    #pragma HLS dataflow
    load_a(a, a_s, outer, size, a_inc);
    load_b(b, b_s, outer, size, b_inc, op);
    compute(a_s, b_s, c_s, outer, size, op);
    store_c(c_s, c, outer, size, c_inc);
}

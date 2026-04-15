#include "VectorOP.h"

// ---------------------------------------------------------------------------
// Subkernels
//
// Each function operates directly on Data_t values.
// saturate_cast<Data_t> handles the overflow policy at compile time:
//   ap_fixed<W,I,...>  — clips the arithmetic result to [-2^(I-1), 2^(I-1)-2^-(W-I)]
//   float / double     — passes the result through unchanged
//
// HLS inlines these functions into the loop body and instantiates all paths
// as concurrent hardware; the 'op' register drives the output mux.
// ---------------------------------------------------------------------------

// add: ap_fixed addition produces ap_fixed<W+1,I+1> (one guard bit);
//      saturate_cast narrows back to Data_t with saturation.
static inline Data_t sub_add(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a + b);
}

// sub: same width rules as add.
static inline Data_t sub_sub(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a - b);
}

// mul: ap_fixed multiplication produces ap_fixed<2W,2I> (full precision);
//      saturate_cast clips the product back to Data_t width.
static inline Data_t sub_mul(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a * b);
}

// div: HLS infers an iterative fixed-point divider; II > 1 is expected for
//      OP_DIV.  Zero denominator returns 0.
static inline Data_t sub_div(Data_t a, Data_t b) {
    if (b == Data_t(0)) return Data_t(0);
    return saturate_cast<Data_t>(a / b);
}

// relu / relu6: purely combinational comparisons; no intermediate overflow
//               is possible, so saturate_cast is not needed.
static inline Data_t sub_relu(Data_t a) {
    return (a < Data_t(0)) ? Data_t(0) : a;
}

static inline Data_t sub_relu6(Data_t a) {
    if (a < Data_t(0)) return Data_t(0);
    if (a > Data_t(6)) return Data_t(6);
    return a;
}

// ---------------------------------------------------------------------------
// Top kernel
// ---------------------------------------------------------------------------
void VectorOPKernel(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      size,
    unsigned      op
) {
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=a      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=size   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=op     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    for (unsigned i = 0; i < size; ++i) {
        #pragma HLS PIPELINE II=1
        Data_t ai = a[i];
        Data_t ci;
        // b[i] is read only for binary operations.  OP_RELU and OP_RELU6 are
        // unary; keeping the read out of those cases avoids unnecessary AXI
        // transactions on the gmem1 port when a unary op is selected.
        switch (op) {
            case OP_ADD:   ci = sub_add (ai, b[i]); break;
            case OP_SUB:   ci = sub_sub (ai, b[i]); break;
            case OP_MUL:   ci = sub_mul (ai, b[i]); break;
            case OP_DIV:   ci = sub_div (ai, b[i]); break;
            case OP_RELU:  ci = sub_relu (ai);       break;
            case OP_RELU6: ci = sub_relu6(ai);       break;
            default:       ci = ai;                  break;
        }
        c[i] = ci;
    }
}

#pragma once

#include "Config.h"
#include "ap_fixed.h"  // always present in the Vitis HLS environment

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts value v to type T, applying saturation if T is an ap_fixed type.
//
//   ap_fixed<W,I,Q,O,N>  — converts v through ap_fixed<W,I,AP_TRN,AP_SAT>,
//                           which clips to the representable range
//                           [-2^(I-1), 2^(I-1) - 2^-(W-I)] before narrowing.
//   float / double / int  — returns T(v) unchanged (pass-through).
//
// Works identically in C simulation and HLS synthesis; no #ifdef needed.
// The partial specialisation is never instantiated when Data_t is float,
// so there is no runtime cost for non-fixed-point types.
// ---------------------------------------------------------------------------

// Primary template: no saturation (float, double, integer types, …)
template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

// Partial specialisation for ap_fixed<W,I,Q,O,N>:
// clips via AP_SAT, then re-wraps in the original overflow mode.
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct saturate_to<ap_fixed<W, I, Q, O, N>> {
    template<typename From>
    static ap_fixed<W, I, Q, O, N> cast(From v) {
        // ap_fixed<W,I,AP_TRN,AP_SAT> applies saturation on the narrowing
        // step; the bits it produces are always within the valid range of the
        // destination type, so the outer ap_fixed<W,I,Q,O,N> cast is lossless.
        return ap_fixed<W, I, Q, O, N>(ap_fixed<W, I, AP_TRN, AP_SAT>(v));
    }
};

template<typename T, typename From>
inline T saturate_cast(From v) {
    return saturate_to<T>::template cast<From>(v);
}

// ---------------------------------------------------------------------------
// Operation codes for VectorOPKernel.
// Passed at runtime via the AXI-Lite 'op' register.
// ---------------------------------------------------------------------------
enum Op : unsigned {
    OP_ADD   = 0,  // c[i] = saturate_cast<Data_t>(a[i] + b[i])
    OP_SUB   = 1,  // c[i] = saturate_cast<Data_t>(a[i] - b[i])
    OP_MUL   = 2,  // c[i] = saturate_cast<Data_t>(a[i] * b[i])
    OP_DIV   = 3,  // c[i] = saturate_cast<Data_t>(a[i] / b[i])  (b[i] ≠ 0)
    OP_RELU  = 4,  // c[i] = max(a[i], 0)          — unary, b[] not read
    OP_RELU6 = 5,  // c[i] = min(max(a[i], 0), 6)  — unary, b[] not read
};

// ---------------------------------------------------------------------------
// VectorOPKernel — element-wise vector operation with saturating output.
//
// Interface:
//   a, b    — AXI4 master (m_axi) read ports; base addresses via AXI-Lite
//   c       — AXI4 master (m_axi) write port;  base address  via AXI-Lite
//   size    — AXI-Lite register: number of elements to process
//   op      — AXI-Lite register: operation selector (Op enum)
//   return  — AXI-Lite control: ap_ctrl_hs (start/done/idle/ready)
//
// For unary operations (OP_RELU, OP_RELU6) only a[] is read; no AXI
// transactions are issued on the gmem1 port and b_addr is ignored.
// ---------------------------------------------------------------------------
void VectorOPKernel(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      size,
    unsigned      op
);

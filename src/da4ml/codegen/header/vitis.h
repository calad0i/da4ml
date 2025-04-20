#pragma once
#include "ap_fixed.h"

template <int s, int b, int i, ap_q_mode Q, ap_o_mode O, int N> ap_fixed<b, i + s> bit_shift(ap_fixed<b, i, Q, O, N> x) {
#pragma HLS INLINE
    ap_fixed<b, i + s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b, int i, ap_q_mode Q, ap_o_mode O, int N> ap_ufixed<b, i + s> bit_shift(ap_ufixed<b, i, Q, O, N> x) {
#pragma HLS INLINE
    ap_ufixed<b, i + s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ap_fixed<b, s> bit_shift(ap_int<b> x) {
#pragma HLS INLINE
    ap_fixed<b, s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ap_ufixed<b, s> bit_shift(ap_uint<b> x) {
#pragma HLS INLINE
    ap_ufixed<b, s> r;
    r.range() = x.range();
    return r;
};

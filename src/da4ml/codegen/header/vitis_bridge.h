#pragma once
#include "ap_fixed.h"

template <typename inp_t, typename out_t, size_t SIZE_IN, size_t SIZE_OUT, typename F>
void vitis_bridge(F f, double *inp, double *out, int size) {
    inp_t in_fixed_buf[SIZE_IN];
    out_t out_fixed_buf[SIZE_OUT];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < SIZE_IN; j++) {
            in_fixed_buf[j] = inp_t(inp[i * SIZE_IN + j]);
        }
        f(in_fixed_buf, out_fixed_buf);
        for (int j = 0; j < SIZE_OUT; j++) {
            out[i * SIZE_OUT + j] = double(out_fixed_buf[j]);
        }
    }
}

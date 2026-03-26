#pragma once

#include "types.hh"

Pair idx_mc(const DAState &state);
Pair idx_mc_dc(const DAState &state, bool absolute = false);
std::tuple<int8_t, int8_t, int8_t>
overlap_counts(const QInterval &q0, const QInterval &q1, const int8_t shift1);
Pair idx_wmc(const DAState &state);
Pair idx_wmc_dc(const DAState &state, bool absolute = false);

inline int8_t iceil_log2(float x) {
    // s1, m24, e7
    uint32_t bits = std::bit_cast<uint32_t>(x);
    uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    return static_cast<int8_t>(exp - 127 + (mant != 0));
}

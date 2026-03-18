#include "indexers.hh"
#include "bit_decompose.hh"
#include <cmath>
#include <algorithm>
#include <limits>

Pair idx_mc(const DAState &state) {
    Pair best_pair = {-1, -1, false, 0};
    size_t max_freq = 0;
    for (const auto &kv : state.freq_stat) {
        if (kv.second >= max_freq) {
            max_freq = kv.second;
            best_pair = kv.first;
        }
    }
    return best_pair;
}

Pair idx_mc_dc(const DAState &state, bool absolute) {
    Pair best_pair = {-1, -1, false, 0};
    float factor = 1e9; // Large factor to prioritize latency difference

    float max_score = absolute ? 0 : -std::numeric_limits<float>::infinity();
    for (const auto &kv : state.freq_stat) {
        auto &k = kv.first;
        float lat0 = state.ops[k.id0].latency;
        float lat1 = state.ops[k.id1].latency;
        float score = kv.second - factor * std::abs(lat0 - lat1);
        if (score >= max_score) {
            max_score = score;
            best_pair = k;
        }
    }
    return best_pair;
}

static inline int8_t qint_to_bw(const QInterval &q) {
    if (q.min == q.max && q.min == 0) {
        return 0;
    }
    int8_t keep_negative = q.min < 0 ? 1 : 0;
    float _min = q.min / q.step, _max = q.max / q.step;
    int8_t mbits = std::max(iceil_log2(_min), iceil_log2(_max + 1));
    return mbits + keep_negative;
}

std::tuple<int8_t, int8_t, int8_t>
overlap_counts(const QInterval &q0, const QInterval &q1, const int8_t shift1) {
    int8_t r0 = -get_lsb_loc(q0.step), r1 = -get_lsb_loc(q1.step) - shift1;
    int8_t b0 = qint_to_bw(q0), b1 = qint_to_bw(q1);
    int8_t l0 = r0 - b0, l1 = r1 - b1;
    std::array<int8_t, 4> sorted = {l0, r0, l1, r1};
    std::sort(sorted.begin(), sorted.end());
    if (r0 < l0 || r1 < l0) {
        std::swap(sorted[1], sorted[2]);
    }

    int8_t overlap = sorted[2] - sorted[1];
    int8_t accum = sorted[1] - sorted[0];
    int8_t tail = sorted[3] - sorted[2];
    return {accum, overlap, tail};
}

Pair idx_wmc(const DAState &state) {
    int64_t max_score = 0;
    Pair best_pair = {-1, -1, false, 0};
    for (const auto &kv : state.freq_stat) {
        auto &k = kv.first;
        auto [_accum, n_overlap, _tail] = overlap_counts(
            state.ops[k.id0].qint, state.ops[k.id1].qint, state.ops[k.id1].data
        );
        int64_t score = int64_t(kv.second) * n_overlap;
        if (score >= max_score) {
            max_score = score;
            best_pair = k;
        }
    }
    return best_pair;
}

Pair idx_wmc_dc(const DAState &state, bool absolute) {
    float max_score = absolute ? 0 : -std::numeric_limits<float>::infinity();
    Pair best_pair = {-1, -1, false, 0};
    for (const auto &kv : state.freq_stat) {
        auto &k = kv.first;
        auto [_accum, n_overlap, _tail] = overlap_counts(
            state.ops[k.id0].qint, state.ops[k.id1].qint, state.ops[k.id1].data
        );
        float lat0 = state.ops[k.id0].latency;
        float lat1 = state.ops[k.id1].latency;
        float score = kv.second * n_overlap - 256 * std::abs(lat0 - lat1);
        if (score >= max_score) {
            max_score = score;
            best_pair = k;
        }
    }
    return best_pair;
}

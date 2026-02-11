#include "indexers.hh"
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
    size_t max_freq = 0;
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

std::pair<double, double> overlap_and_accum(const QInterval &q0, const QInterval &q1) {
    double min0 = q0.min, max0 = q0.max, step0 = q0.step;
    double min1 = q1.min, max1 = q1.max, step1 = q1.step;
    max0 += step0;
    max1 += step1;

    double f = -std::log2(std::max(step0, step1));
    double i_high = std::ceil(
        std::log2(
            std::max({std::abs(min0), std::abs(min1), std::abs(max0), std::abs(max1)})
        )
    );
    double i_low = std::ceil(
        std::log2(
            std::min(
                std::max(std::abs(min0), std::abs(max0)),
                std::max(std::abs(min1), std::abs(max1))
            )
        )
    );
    int k = (q0.min < 0 || q1.min < 0) ? 1 : 0;
    double n_accum = k + i_high + f;
    double n_overlap = k + i_low + f;
    return {n_overlap, n_accum};
}

Pair idx_wmc(const DAState &state) {
    float max_score = -std::numeric_limits<float>::infinity();
    Pair best_pair = {-1, -1, false, 0};
    for (const auto &kv : state.freq_stat) {
        auto &k = kv.first;
        auto [n_overlap, _] =
            overlap_and_accum(state.ops[k.id0].qint, state.ops[k.id1].qint);
        float score = kv.second * n_overlap;
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
        auto [n_overlap, n_accum] =
            overlap_and_accum(state.ops[k.id0].qint, state.ops[k.id1].qint);
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

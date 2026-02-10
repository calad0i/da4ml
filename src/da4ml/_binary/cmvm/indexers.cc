#include "indexers.hh"
#include <cmath>
#include <algorithm>
#include <limits>

int64_t idx_mc(const DAState &state) {
    auto freqs = state.freq_stat.values();
    int64_t idx = -1, max_freq_count = 0;
    for (size_t i = 0; i < freqs.size(); ++i) {
        if (freqs[i] > max_freq_count) {
            max_freq_count = freqs[i];
            idx = static_cast<int64_t>(i);
        }
    }
    return idx;
}

int64_t idx_mc_dc(const DAState &state, bool absolute) {
    auto freqs = state.freq_stat.values();
    auto keys = state.freq_stat.keys();
    float factor = 1e9; // Large factor to prioritize latency difference

    double max_score = -std::numeric_limits<double>::infinity();
    int64_t best_idx = 0;
    for (size_t i = 0; i < freqs.size(); ++i) {
        auto lat0 = state.ops[keys[i].id0].latency;
        auto lat1 = state.ops[keys[i].id1].latency;
        auto score = freqs[i] - std::abs(lat1 - lat0) * factor;
        if (score > max_score) {
            max_score = score;
            best_idx = static_cast<int64_t>(i);
        }
    }
    if (absolute && max_score < 0)
        return -1;
    return best_idx;
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

int64_t idx_wmc(const DAState &state) {
    auto freqs = state.freq_stat.values();
    auto keys = state.freq_stat.keys();

    double max_score = -std::numeric_limits<double>::infinity();
    int64_t best_idx = -1;
    for (size_t i = 0; i < freqs.size(); ++i) {
        auto [n_overlap, _] =
            overlap_and_accum(state.ops[keys[i].id0].qint, state.ops[keys[i].id1].qint);
        double score = freqs[i] * n_overlap;
        if (score > max_score) {
            max_score = score;
            best_idx = static_cast<int64_t>(i);
        }
    }
    if (max_score < 0)
        return -1;
    return best_idx;
}

int64_t idx_wmc_dc(const DAState &state, bool absolute) {
    auto freqs = state.freq_stat.values();
    auto keys = state.freq_stat.keys();

    double max_score = -std::numeric_limits<double>::infinity();
    int64_t best_idx = -1;
    for (size_t i = 0; i < freqs.size(); ++i) {
        auto &k = keys[i];
        auto [n_overlap, _] =
            overlap_and_accum(state.ops[k.id0].qint, state.ops[k.id1].qint);
        double lat0 = state.ops[k.id0].latency;
        double lat1 = state.ops[k.id1].latency;
        double score = freqs[i] * n_overlap - 256 * std::abs(lat0 - lat1);
        if (score > max_score) {
            max_score = score;
            best_idx = static_cast<int64_t>(i);
        }
    }
    if (absolute && max_score < 0)
        return -1;
    return best_idx;
}

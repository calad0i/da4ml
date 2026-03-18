#pragma once

#include "types.hh"
#include <vector>
#include <tuple>

QInterval qint_add(
    const QInterval &q0,
    const QInterval &q1,
    int64_t shift,
    bool sub0 = false,
    bool sub1 = false
);

std::pair<float, float> cost_add(const QInterval &q0, const QInterval &q1, int64_t shift);

DAState create_state(
    const xt::xarray<float> &kernel_in,
    const std::vector<QInterval> &qintervals,
    const std::vector<float> &inp_latencies,
    bool no_stat_init = false,
    bool partial = false
);
void update_stats(DAState &state, const Pair &pair);

std::vector<std::tuple<int64_t, int64_t, int64_t>>
gather_matching_idxs(const DAState &state, const Pair &pair);

Op pair_to_op(const Pair &pair, const DAState &state);

void update_expr(DAState &state, const Pair &pair);

void update_state(DAState &state, const Pair &pair, bool partial = false);

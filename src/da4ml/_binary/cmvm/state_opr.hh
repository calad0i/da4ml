#pragma once

#include "types.hh"
#include <vector>
#include <tuple>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

QInterval qint_add(
    const QInterval &q0,
    const QInterval &q1,
    int64_t shift,
    bool sub0 = false,
    bool sub1 = false
);

std::pair<double, double> cost_add(
    const QInterval &q0,
    const QInterval &q1,
    int64_t shift,
    bool sub = false,
    int adder_size = -1,
    int carry_size = -1
);

nb::tuple cost_add_numpy(
    const nb::tuple &q0_obj,
    const nb::tuple &q1_obj,
    int64_t shift,
    bool sub,
    int adder_size,
    int carry_size
);

DAState create_state(
    const xt::xarray<std::float32_t> &kernel,
    const std::vector<QInterval> &qintervals,
    const std::vector<double> &inp_latencies,
    bool no_stat_init = false
);

void update_stats(DAState &state, const Pair &pair);

std::vector<std::tuple<int64_t, int64_t, int64_t>>
gather_matching_idxs(const DAState &state, const Pair &pair);

Op pair_to_op(
    const Pair &pair,
    const DAState &state,
    int adder_size = -1,
    int carry_size = -1
);

void update_expr(DAState &state, const Pair &pair, int adder_size, int carry_size);

void update_state(DAState &state, const Pair &pair, int adder_size, int carry_size);

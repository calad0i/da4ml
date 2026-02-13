#pragma once

#include "types.hh"
#include <string>
#include <vector>

DAState cmvm(
    const xt::xarray<float> &kernel,
    const std::string &method = "wmc",
    const std::vector<QInterval> &qintervals = {},
    const std::vector<float> &inp_latencies = {},
    int adder_size = -1,
    int carry_size = -1
);

CombLogicResult to_solution(const DAState &state, int adder_size, int carry_size);

CombLogicResult solve_single(
    const xt::xarray<float> &kernel,
    const std::string &method,
    const std::vector<QInterval> &qintervals,
    const std::vector<float> &latencies,
    int adder_size,
    int carry_size
);

#pragma once

#include "types.hh"
#include <string>
#include <vector>

DAState cmvm(
    const xt::xarray<float> &kernel,
    const std::string &method = "wmc",
    const std::vector<QInterval> &qintervals = {},
    const std::vector<float> &inp_latencies = {},
    bool partial = false
);

CombLogicResult to_solution(const DAState &state, bool partial = false);

CombLogicResult solve_single(
    const xt::xarray<float> &kernel,
    const std::string &method,
    const std::vector<QInterval> &qintervals,
    const std::vector<float> &latencies,
    bool partial = false
);

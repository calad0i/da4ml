#pragma once

#include "types.hh"
#include <string>
#include <vector>

float minimal_latency(
    const xt::xarray<float> &kernel,
    const std::vector<QInterval> &qintervals,
    const std::vector<float> &latencies,
    int carry_size = -1,
    int adder_size = -1
);

PipelineResult _solve(
    const xt::xarray<float> &kernel,
    std::string method0 = "wmc",
    std::string method1 = "auto",
    int hard_dc = -1,
    int decompose_dc = -2,
    const std::vector<QInterval> &qintervals = {},
    const std::vector<float> &latencies = {},
    int adder_size = -1,
    int carry_size = -1
);

PipelineResult solve(
    const xt::xarray<float> &kernel,
    const std::string &method0 = "wmc",
    const std::string &method1 = "auto",
    int hard_dc = -1,
    int decompose_dc = -2,
    const std::vector<QInterval> &qintervals = {},
    const std::vector<float> &latencies = {},
    int adder_size = -1,
    int carry_size = -1,
    bool search_all_decompose_dc = true
);

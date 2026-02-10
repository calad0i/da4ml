#pragma once

#include "types.hh"
#include <string>
#include <vector>

double minimal_latency(
    const xt::xarray<std::float32_t> &kernel,
    const std::vector<QInterval> &qintervals,
    const std::vector<double> &latencies,
    int carry_size = -1,
    int adder_size = -1
);

PipelineResult single_solve(
    const xt::xarray<std::float32_t> &kernel,
    std::string method0 = "wmc",
    std::string method1 = "auto",
    int hard_dc = -1,
    int decompose_dc = -2,
    const std::vector<QInterval> &qintervals = {},
    const std::vector<double> &latencies = {},
    int adder_size = -1,
    int carry_size = -1
);

PipelineResult solve(
    const xt::xarray<std::float32_t> &kernel,
    const std::string &method0 = "wmc",
    const std::string &method1 = "auto",
    int hard_dc = -1,
    int decompose_dc = -2,
    const std::vector<QInterval> &qintervals = {},
    const std::vector<double> &latencies = {},
    int adder_size = -1,
    int carry_size = -1,
    bool search_all_decompose_dc = true
);

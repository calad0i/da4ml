#include "api.hh"
#include "cmvm_core.hh"
#include "mat_decompose.hh"
#include "src/da4ml/_binary/cmvm/types.hh"
#include "state_opr.hh"
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

float minimal_latency(
    const xt::xarray<float> &kernel,
    const std::vector<QInterval> &qintervals,
    const std::vector<float> &latencies,
    int carry_size,
    int adder_size
) {
    DAState state = create_state(kernel, qintervals, latencies, true);
    CombLogicResult sol = to_solution(state, adder_size, carry_size);
    float max_lat = 0.0;
    for (auto idx : sol.out_idxs) {
        float lat = (idx >= 0) ? sol.ops[idx].latency : 0.0;
        max_lat = std::max(max_lat, lat);
    }
    return max_lat;
}

PipelineResult _solve(
    const xt::xarray<float> &kernel,
    std::string method0,
    std::string method1,
    int hard_dc,
    int decompose_dc,
    const std::vector<QInterval> &qintervals_in,
    const std::vector<float> &latencies_in,
    int adder_size,
    int carry_size
) {
    size_t n_in = kernel.shape(0);

    if (method1 == "auto") {
        if (hard_dc >= 6 || method0.ends_with("dc")) {
            method1 = method0;
        }
        else {
            method1 = method0 + "-dc";
        }
    }
    if (hard_dc == 0 && !method0.ends_with("dc")) {
        method0 = method0 + "-dc";
    }

    std::vector<QInterval> qintervals;
    if (qintervals_in.empty()) {
        qintervals.assign(n_in, QInterval{-128.0, 127.0, 1.0});
    }
    else {
        qintervals = qintervals_in;
    }
    std::vector<float> inp_latencies;
    if (latencies_in.empty()) {
        inp_latencies.assign(n_in, 0.0);
    }
    else {
        inp_latencies = latencies_in;
    }

    float min_lat = std::numeric_limits<float>::infinity();
    if (hard_dc >= 0)
        min_lat =
            minimal_latency(kernel, qintervals, inp_latencies, carry_size, adder_size);
    float latency_allowed = hard_dc + min_lat;

    int log2_n = static_cast<int>(std::ceil(std::log2(static_cast<float>(n_in))));
    if (decompose_dc == -2) {
        decompose_dc = std::min(hard_dc, log2_n);
    }
    else {
        decompose_dc = std::min({hard_dc, decompose_dc, log2_n});
    }

    CombLogicResult sol0, sol1;
    while (true) {
        if (decompose_dc < 0 && hard_dc >= 0) {
            if (method0 != "dummy") {
                method0 = "wmc-dc";
                method1 = "wmc-dc";
            }
            else {
                method0 = "dummy";
                method1 = "dummy";
            }
        }

        auto [mat0, mat1] = kernel_decompose(xt::xarray<float>(kernel), decompose_dc);
        sol0 = solve_single(
            mat0, method0, qintervals, inp_latencies, adder_size, carry_size
        );

        std::vector<float> latencies0;
        std::vector<QInterval> qintervals0;
        float max_lat0 = 0.0;
        for (auto idx : sol0.out_idxs) {
            float lat = (idx >= 0) ? sol0.ops[idx].latency : 0.0;
            latencies0.push_back(lat);
            max_lat0 = std::max(max_lat0, lat);
            if (idx >= 0) {
                qintervals0.push_back(sol0.ops[idx].qint);
            }
            else {
                qintervals0.push_back(
                    QInterval{0.0, 0.0, std::numeric_limits<float>::infinity()}
                );
            }
        }

        if (max_lat0 > latency_allowed) {
            if (!(method0 == "wmc-dc" && method1 == "wmc-dc") || decompose_dc >= 0) {
                decompose_dc--;
                continue;
            }
        }

        sol1 =
            solve_single(mat1, method1, qintervals0, latencies0, adder_size, carry_size);

        float max_lat1 = 0.0;
        for (auto idx : sol1.out_idxs) {
            float lat = (idx >= 0) ? sol1.ops[idx].latency : 0.0;
            max_lat1 = std::max(max_lat1, lat);
        }

        if (max_lat1 > latency_allowed) {
            if (!(method0 == "wmc-dc" && method1 == "wmc-dc") || decompose_dc >= 0) {
                decompose_dc--;
                continue;
            }
        }
        break;
    }

    PipelineResult result;
    result.solutions = {std::move(sol0), std::move(sol1)};
    return result;
}

PipelineResult solve(
    const xt::xarray<float> &kernel,
    const std::string &method0,
    const std::string &method1,
    int hard_dc,
    int decompose_dc,
    const std::vector<QInterval> &qintervals_in,
    const std::vector<float> &latencies_in,
    int adder_size,
    int carry_size,
    bool search_all_decompose_dc
) {
    size_t n_in = kernel.shape(0);

    std::vector<QInterval> qintervals;
    if (qintervals_in.empty()) {
        qintervals.assign(n_in, QInterval{-128.0, 127.0, 1.0});
    }
    else {
        qintervals = qintervals_in;
    }
    std::vector<float> latencies;
    if (latencies_in.empty()) {
        latencies.assign(n_in, 0.0);
    }
    else {
        latencies = latencies_in;
    }

    if (!search_all_decompose_dc) {
        return _solve(
            kernel,
            method0,
            method1,
            hard_dc,
            decompose_dc,
            qintervals,
            latencies,
            adder_size,
            carry_size
        );
    }

    int _hard_dc = hard_dc;
    if (_hard_dc < 0)
        _hard_dc = 1000000000;

    int max_decompose_dc = std::min(
        _hard_dc, static_cast<int>(std::ceil(std::log2(static_cast<float>(n_in))))
    );

    std::vector<int> try_dcs;
    for (int d = -1; d <= max_decompose_dc; ++d) {
        try_dcs.push_back(d);
    }

    size_t n_tries = try_dcs.size();
    std::vector<PipelineResult> solution_candidates(n_tries);
    std::vector<float> costs(n_tries);

    std::exception_ptr eptr = nullptr;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_tries; ++i) {
        try {
            auto _csol = _solve(
                kernel,
                method0,
                method1,
                _hard_dc,
                try_dcs[i],
                qintervals,
                latencies,
                adder_size,
                carry_size
            );
            float _cost = 0.0;
            for (auto &sol : _csol.solutions) {
                for (auto &op : sol.ops) {
                    _cost += op.cost;
                }
            }
            solution_candidates[i] = _csol;
            costs[i] = _cost;
        }
        catch (...) {
#pragma omp critical
            {
                if (!eptr)
                    eptr = std::current_exception();
            }
        }
    }
    if (eptr)
        std::rethrow_exception(eptr);

    // Find argmin
    size_t best = 0;
    for (size_t i = 1; i < n_tries; ++i) {
        if (costs[i] < costs[best])
            best = i;
    }

    return solution_candidates[best];
}

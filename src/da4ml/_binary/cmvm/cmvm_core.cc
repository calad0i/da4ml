#include "cmvm_core.hh"
#include "state_opr.hh"
#include "indexers.hh"
#include <cmath>
#include <algorithm>
#include <queue>
#include <tuple>
#include <functional>

DAState cmvm(
    const xt::xarray<float> &kernel,
    const std::string &method,
    const std::vector<QInterval> &qintervals_in,
    const std::vector<double> &inp_latencies_in,
    int adder_size,
    int carry_size
) {
    size_t n_in = kernel.shape(0);
    std::vector<QInterval> qintervals;
    if (qintervals_in.empty()) {
        qintervals.assign(n_in, QInterval{-128.0, 127.0, 1.0});
    }
    else {
        qintervals = qintervals_in;
    }
    std::vector<double> inp_latencies;
    if (inp_latencies_in.empty()) {
        inp_latencies.assign(n_in, 0.0);
    }
    else {
        inp_latencies = inp_latencies_in;
    }

    DAState state = create_state(kernel, qintervals, inp_latencies);

    while (true) {
        if (state.freq_stat.empty())
            break;

        Pair pair_chosen;
        if (method == "mc") {
            pair_chosen = idx_mc(state);
        }
        else if (method == "mc-dc") {
            pair_chosen = idx_mc_dc(state, true);
        }
        else if (method == "mc-pdc") {
            pair_chosen = idx_mc_dc(state, false);
        }
        else if (method == "wmc") {
            pair_chosen = idx_wmc(state);
        }
        else if (method == "wmc-dc") {
            pair_chosen = idx_wmc_dc(state, true);
        }
        else if (method == "wmc-pdc") {
            pair_chosen = idx_wmc_dc(state, false);
        }
        else if (method == "dummy") {
            break;
        }
        else {
            throw std::runtime_error("Unknown method: " + method);
        }

        if (pair_chosen.id0 == -1 || pair_chosen.id1 == -1)
            break;

        update_state(state, pair_chosen, adder_size, carry_size);
    }
    return state;
}

// Heap entry for to_solution — matches Python's tuple comparison order
struct HeapEntry {
    double lat;
    int64_t sub;
    int64_t left_align;
    double qmin, qmax, qstep; // QInterval fields for comparison
    int64_t id;
    int64_t shift;

    auto as_tuple() const {
        return std::tie(lat, sub, left_align, qmin, qmax, qstep, id, shift);
    }
    bool operator>(const HeapEntry &o) const { return as_tuple() > o.as_tuple(); }
};

CombLogicResult to_solution(const DAState &state, int adder_size, int carry_size) {
    auto ops = state.ops; // copy
    int64_t n_out = static_cast<int64_t>(state.kernel.shape(1));
    size_t n_expr = state.expr.size();
    size_t n_bits = state.n_bits;

    // Build 3D expr array for indexing
    // expr[i_in][i_out, j_bit]

    std::vector<int64_t> out_idxs, out_shifts_vec, out_negs;
    std::vector<int64_t> inp_shifts(state.shift0.begin(), state.shift0.end());
    std::vector<int64_t> out_shifts_base(state.shift1.begin(), state.shift1.end());

    int64_t _global_id = static_cast<int64_t>(ops.size());

    for (int64_t i_out = 0; i_out < n_out; ++i_out) {
        // Find all nonzero entries in expr[:, i_out, :]
        std::vector<int64_t> idx, shifts;
        std::vector<int64_t> sub_vals;
        for (size_t i_in = 0; i_in < n_expr; ++i_in) {
            for (int16_t v : state.expr[i_in].rows[i_out]) {
                idx.push_back(static_cast<int64_t>(i_in));
                shifts.push_back(static_cast<int64_t>(SparseExpr::to_shift(v)));
                sub_vals.push_back(SparseExpr::to_sign(v) == -1 ? 1 : 0);
            }
        }

        // No reduction required
        if (idx.size() == 1) {
            out_shifts_vec.push_back(out_shifts_base[i_out] + shifts[0]);
            out_idxs.push_back(idx[0]);
            out_negs.push_back(sub_vals[0]);
            continue;
        }
        // Output is zero
        if (idx.empty()) {
            out_idxs.push_back(-1);
            out_shifts_vec.push_back(out_shifts_base[i_out]);
            out_negs.push_back(0);
            continue;
        }

        // Build heap entries
        // Min-heap using std::greater
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>>
            heap;
        for (size_t k = 0; k < idx.size(); ++k) {
            auto &qint = ops[idx[k]].qint;
            double lat = ops[idx[k]].latency;
            int64_t n_int = static_cast<int64_t>(
                std::log2(std::max(std::abs(qint.max + qint.step), std::abs(qint.min)))
            );
            int64_t la = n_int + shifts[k];
            heap.push(
                {lat, sub_vals[k], la, qint.min, qint.max, qint.step, idx[k], shifts[k]}
            );
        }

        while (heap.size() > 1) {
            auto e0 = heap.top();
            heap.pop();
            auto e1 = heap.top();
            heap.pop();

            QInterval qint0 = {e0.qmin, e0.qmax, e0.qstep};
            QInterval qint1 = {e1.qmin, e1.qmax, e1.qstep};
            double lat0 = e0.lat, lat1 = e1.lat;
            int64_t sub0 = e0.sub, sub1 = e1.sub;
            int64_t id0 = e0.id, id1 = e1.id;
            int64_t shift0 = e0.shift, shift1 = e1.shift;

            QInterval qint;
            double dlat, dcost;
            Op op;
            int64_t result_shift;

            if (sub0) {
                int64_t s = shift0 - shift1;
                qint = qint_add(qint1, qint0, s, sub1 != 0, sub0 != 0);
                auto [dl, dc] =
                    cost_add(qint1, qint0, s, (1 ^ sub1) != 0, adder_size, carry_size);
                dlat = dl;
                dcost = dc;
                double lat = std::max(lat0, lat1) + dlat;
                op = Op{id1, id0, 1 ^ sub1, s, qint, lat, dcost};
                result_shift = shift1;
            }
            else {
                int64_t s = shift1 - shift0;
                qint = qint_add(qint0, qint1, s, sub0 != 0, sub1 != 0);
                auto [dl, dc] =
                    cost_add(qint0, qint1, s, sub1 != 0, adder_size, carry_size);
                dlat = dl;
                dcost = dc;
                double lat = std::max(lat0, lat1) + dlat;
                op = Op{id0, id1, sub1, s, qint, lat, dcost};
                result_shift = shift0;
            }

            int64_t la = static_cast<int64_t>(std::log2(
                             std::max(std::abs(qint.max + qint.step), std::abs(qint.min))
                         )) +
                         result_shift;
            double lat = op.latency;
            heap.push(
                {lat,
                 sub0 & sub1,
                 la,
                 qint.min,
                 qint.max,
                 qint.step,
                 _global_id,
                 result_shift}
            );
            ops.push_back(op);
            _global_id++;
        }

        auto final_entry = heap.top();
        out_idxs.push_back(_global_id - 1);
        out_negs.push_back(final_entry.sub);
        out_shifts_vec.push_back(out_shifts_base[i_out] + final_entry.shift);
    }

    CombLogicResult result;
    result.shape = {
        static_cast<int64_t>(state.kernel.shape(0)),
        static_cast<int64_t>(state.kernel.shape(1))
    };
    result.inp_shifts = std::move(inp_shifts);
    result.out_idxs = std::move(out_idxs);
    result.out_shifts = std::move(out_shifts_vec);
    result.out_negs = std::move(out_negs);
    result.ops = std::move(ops);
    result.carry_size = carry_size;
    result.adder_size = adder_size;
    return result;
}

CombLogicResult solve_single(
    const xt::xarray<float> &kernel,
    const std::string &method,
    const std::vector<QInterval> &qintervals,
    const std::vector<double> &latencies,
    int adder_size,
    int carry_size
) {
    DAState state = cmvm(kernel, method, qintervals, latencies, adder_size, carry_size);
    return to_solution(state, adder_size, carry_size);
}

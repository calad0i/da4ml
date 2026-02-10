#include "state_opr.hh"
#include "bit_decompose.hh"
#include <cmath>
#include <algorithm>
#include <unordered_map>

QInterval
qint_add(const QInterval &q0, const QInterval &q1, int64_t shift, bool sub0, bool sub1) {
    double min0 = q0.min, max0 = q0.max, step0 = q0.step;
    double min1 = q1.min, max1 = q1.max, step1 = q1.step;
    if (sub0) {
        std::swap(min0, max0);
        min0 = -min0;
        max0 = -max0;
    }
    if (sub1) {
        std::swap(min1, max1);
        min1 = -min1;
        max1 = -max1;
    }

    double s = std::pow(2.0, shift);
    min1 *= s;
    max1 *= s;
    step1 *= s;

    return QInterval{min0 + min1, max0 + max1, std::min(step0, step1)};
}

std::pair<double, double> cost_add(
    const QInterval &q0,
    const QInterval &q1,
    int64_t shift,
    bool sub,
    int adder_size,
    int carry_size
) {
    if (adder_size < 0 && carry_size < 0)
        return {1.0, 1.0};
    if (adder_size < 0)
        adder_size = 65535;
    if (carry_size < 0)
        carry_size = 65535;

    double min0 = q0.min, max0 = q0.max, step0 = q0.step;
    double min1 = q1.min, max1 = q1.max, step1 = q1.step;
    if (sub)
        std::swap(min1, max1);
    double sf = std::pow(2.0, shift);
    min1 *= sf;
    max1 *= sf;
    step1 *= sf;
    max0 += step0;
    max1 += step1;

    double f = -std::log2(std::max(step0, step1));
    double i = std::ceil(
        std::log2(
            std::max({std::abs(min0), std::abs(min1), std::abs(max0), std::abs(max1)})
        )
    );
    int k = (q0.min < 0 || q1.min < 0) ? 1 : 0;
    double n_accum = k + i + f;

    return {std::ceil(n_accum / carry_size), std::ceil(n_accum / adder_size)};
}

DAState create_state(
    const xt::xarray<std::float32_t> &kernel_in,
    const std::vector<QInterval> &qintervals,
    const std::vector<double> &inp_latencies,
    bool no_stat_init
) {
    size_t n_in = kernel_in.shape(0);
    size_t n_out = kernel_in.shape(1);
    DAState state;

    xt::xarray<std::float32_t> kernel_f32(kernel_in);
    auto [csd, shift0, shift1] = csd_decompose(kernel_f32);

    // Zero out CSD rows for zero-range inputs
    for (size_t i = 0; i < n_in; ++i) {
        if (qintervals[i].min == 0.0 && qintervals[i].max == 0.0) {
            xt::view(csd, i) = 0;
        }
    }

    size_t n_bits = csd.shape(2);
    std::vector<xt::xarray<int8_t>> expr;
    for (size_t i = 0; i < n_in; ++i) {
        expr.push_back(xt::view(csd, i));
    }

    auto &stat = state.freq_stat;
    if (!no_stat_init) {
        for (size_t i_out = 0; i_out < n_out; ++i_out) {
            for (size_t i0 = 0; i0 < n_in; ++i0) {
                for (size_t j0 = 0; j0 < n_bits; ++j0) {
                    int8_t bit0 = csd(i0, i_out, j0);
                    if (!bit0)
                        continue;
                    for (size_t i1 = i0; i1 < n_in; ++i1) {
                        for (size_t j1 = 0; j1 < n_bits; ++j1) {
                            int8_t bit1 = csd(i1, i_out, j1);
                            if (!bit1)
                                continue;
                            if (i0 == i1 && j0 <= j1)
                                continue;
                            Pair pair{
                                (int64_t)i0,
                                (int64_t)i1,
                                bit0 != bit1,
                                (int64_t)j1 - (int64_t)j0
                            };
                            stat[pair] = stat[pair] + 1;
                        }
                    }
                }
            }
        }
        // Remove pairs with count < 2
        for (auto it = stat.begin(); it != stat.end();) {
            if (it->second < 2) {
                it = stat.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    std::vector<Op> ops;
    for (size_t i = 0; i < n_in; ++i) {
        ops.push_back(Op{(int64_t)i, -1, -1, 0, qintervals[i], inp_latencies[i], 0.0});
    }

    state.shift0 = shift0;
    state.shift1 = shift1;
    state.expr = std::move(expr);
    state.ops = std::move(ops);
    // state.freq_stat = std::move(stat);
    state.kernel = kernel_in;
    return state;
}

std::vector<std::tuple<int64_t, int64_t, int64_t>>
gather_matching_idxs(const DAState &state, const Pair &pair) {
    int64_t id0 = pair.id0, id1 = pair.id1;
    int64_t shift = pair.shift;
    bool sub = pair.sub;
    size_t n_out = state.kernel.shape(1);
    size_t n_bits = state.expr[0].shape(1);

    bool flip = false;
    if (shift < 0) {
        std::swap(id0, id1);
        shift = -shift;
        flip = true;
    }

    int sign = sub ? -1 : 1;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> result;

    for (int64_t j0 = 0; j0 < (int64_t)n_bits - shift; ++j0) {
        for (size_t i_out = 0; i_out < n_out; ++i_out) {
            int8_t bit0 = state.expr[id0](i_out, j0);
            int64_t j1 = j0 + shift;
            int8_t bit1 = state.expr[id1](i_out, j1);
            if (sign * bit1 * bit0 != 1)
                continue;

            if (flip) {
                result.emplace_back(i_out, j1, j0);
            }
            else {
                result.emplace_back(i_out, j0, j1);
            }
        }
    }
    return result;
}

Op pair_to_op(const Pair &pair, const DAState &state, int adder_size, int carry_size) {
    auto [dlat, cost] = cost_add(
        state.ops[pair.id0].qint,
        state.ops[pair.id1].qint,
        pair.shift,
        pair.sub,
        adder_size,
        carry_size
    );
    double lat =
        std::max(state.ops[pair.id0].latency, state.ops[pair.id1].latency) + dlat;
    QInterval qint = qint_add(
        state.ops[pair.id0].qint, state.ops[pair.id1].qint, pair.shift, false, pair.sub
    );
    return Op{pair.id0, pair.id1, (int64_t)pair.sub, pair.shift, qint, lat, cost};
}

void update_expr(DAState &state, const Pair &pair, int adder_size, int carry_size) {
    int64_t id0 = pair.id0, id1 = pair.id1;
    Op op = pair_to_op(pair, state, adder_size, carry_size);
    size_t n_out = state.kernel.shape(1);
    size_t n_bits = state.expr[0].shape(1);

    state.ops.push_back(op);

    xt::xarray<int8_t> new_slice = xt::zeros<int8_t>({n_out, n_bits});

    // Inline gather_matching_idxs with interleaved zeroing to match
    // Python generator behavior: zeroing bits from one match must be
    // visible to subsequent match checks (critical when id0 == id1).
    int64_t gid0 = pair.id0, gid1 = pair.id1;
    int64_t shift = pair.shift;
    bool sub = pair.sub;

    bool flip = false;
    if (shift < 0) {
        std::swap(gid0, gid1);
        shift = -shift;
        flip = true;
    }

    int sign = sub ? -1 : 1;

    for (int64_t j0 = 0; j0 < (int64_t)n_bits - shift; ++j0) {
        for (size_t i_out = 0; i_out < n_out; ++i_out) {
            int8_t bit0 = state.expr[gid0](i_out, j0);
            int64_t j1 = j0 + shift;
            int8_t bit1 = state.expr[gid1](i_out, j1);
            if (sign * bit1 * bit0 != 1)
                continue;

            int64_t rj0, rj1;
            if (flip) {
                rj0 = j1;
                rj1 = j0;
            }
            else {
                rj0 = j0;
                rj1 = j1;
            }

            new_slice(i_out, rj0) = state.expr[id0](i_out, rj0);
            state.expr[id0](i_out, rj0) = 0;
            state.expr[id1](i_out, rj1) = 0;
        }
    }

    state.expr.push_back(std::move(new_slice));
}

void update_stats(DAState &state, const Pair &pair) {
    int64_t id0 = pair.id0, id1 = pair.id1;

    // Delete all entries involving id0 or id1
    for (auto it = state.freq_stat.begin(); it != state.freq_stat.end();) {
        const Pair &p = it->first;
        if (p.id0 == id0 || p.id0 == id1 || p.id1 == id0 || p.id1 == id1) {
            it = state.freq_stat.erase(it);
        }
        else {
            ++it;
        }
    }

    int64_t n_constructed = static_cast<int64_t>(state.expr.size());
    std::vector<int64_t> modified = {n_constructed - 1, id0};
    if (id1 != id0)
        modified.push_back(id1);

    size_t n_bits = state.expr[0].shape(1);
    size_t n_out = state.kernel.shape(1);

    for (size_t i_out = 0; i_out < n_out; ++i_out) {
        for (int64_t _in0 : modified) {
            for (int64_t _in1 = 0; _in1 < n_constructed; ++_in1) {
                // Check if _in1 is in modified and _in0 > _in1
                bool in1_modified = false;
                for (int64_t m : modified) {
                    if (m == _in1) {
                        in1_modified = true;
                        break;
                    }
                }
                if (in1_modified && _in0 > _in1)
                    continue;

                int64_t lo = std::min(_in0, _in1);
                int64_t hi = std::max(_in0, _in1);

                for (size_t j0 = 0; j0 < n_bits; ++j0) {
                    int8_t bit0 = state.expr[lo](i_out, j0);
                    if (!bit0)
                        continue;
                    for (size_t j1 = 0; j1 < n_bits; ++j1) {
                        int8_t bit1 = state.expr[hi](i_out, j1);
                        if (!bit1)
                            continue;
                        if (lo == hi && j0 <= (int64_t)j1)
                            continue;
                        Pair p{lo, hi, bit0 != bit1, (int64_t)j1 - (int64_t)j0};
                        state.freq_stat[p] = state.freq_stat[p] + 1;
                    }
                }
            }
        }
    }

    // Remove pairs with count < 2
    for (auto it = state.freq_stat.begin(); it != state.freq_stat.end();) {
        if (it->second < 2) {
            it = state.freq_stat.erase(it);
        }
        else {
            ++it;
        }
    }
}

void update_state(DAState &state, const Pair &pair, int adder_size, int carry_size) {
    update_expr(state, pair, adder_size, carry_size);
    update_stats(state, pair);
}

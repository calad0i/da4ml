#include "state_opr.hh"
#include "bit_decompose.hh"
#include "src/da4ml/_binary/cmvm/types.hh"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <array>

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

inline Pair _make_pair(int64_t id0, int64_t id1, int8_t v0, int8_t v1) {
    if (id0 > id1) {
        throw std::invalid_argument("id0 should be <= id1");
    }
    bool sub = SparseExpr::to_sign(v0) != SparseExpr::to_sign(v1);
    int8_t shift =
        static_cast<int8_t>(SparseExpr::to_shift(v1) - SparseExpr::to_shift(v0));
    return Pair{id0, id1, shift, sub};
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
    std::vector<SparseExpr> expr;
    for (size_t i = 0; i < n_in; ++i) {
        SparseExpr se;
        se.rows.resize(n_out);
        for (size_t io = 0; io < n_out; ++io) {
            for (size_t j = 0; j < n_bits; ++j) {
                int8_t v = csd(i, io, j);
                if (v != 0)
                    se.set_bit(io, j, v);
            }
        }
        expr.push_back(std::move(se));
    }

    auto &stat = state.freq_stat;
    if (!no_stat_init) {
        std::vector<Pair> raw_pairs;
        for (size_t i_out = 0; i_out < n_out; ++i_out) {
            for (size_t i0 = 0; i0 < n_in; ++i0) {
                const auto &row0 = expr[i0].rows[i_out];
                for (size_t i1 = i0; i1 < n_in; ++i1) {
                    const auto &row1 = expr[i1].rows[i_out];
                    if (row0.empty() || row1.empty())
                        continue;
                    if (i0 == i1) {
                        for (size_t a = 1; a < row0.size(); ++a) {
                            int8_t v0 = row0[a];
                            for (size_t b = 0; b < a; ++b) {
                                raw_pairs.push_back(_make_pair(i0, i1, v0, row0[b]));
                            }
                        }
                    }
                    else {
                        for (int8_t v0 : row0) {
                            for (int8_t v1 : row1) {
                                raw_pairs.push_back(_make_pair(i0, i1, v0, v1));
                            }
                        }
                    }
                }
            }
        }
        // Sort, deduplicate, count — only keep pairs with count >= 2
        stat.initialize(raw_pairs);
    }

    std::vector<Op> ops;
    for (size_t i = 0; i < n_in; ++i) {
        ops.push_back(Op{(int64_t)i, -1, -1, 0, qintervals[i], inp_latencies[i], 0.0});
    }

    state.shift0 = shift0;
    state.shift1 = shift1;
    state.expr = std::move(expr);
    state.n_bits = n_bits;
    state.ops = std::move(ops);
    // state.freq_stat = std::move(stat);
    state.kernel = kernel_in;
    return state;
}

std::vector<std::tuple<int64_t, int64_t, int64_t>>
gather_matching_idxs(const DAState &state, const Pair &pair) {
    int64_t id0 = pair.id0, id1 = pair.id1;
    int8_t shift = pair.shift;
    bool sub = pair.sub;
    size_t n_out = state.kernel.shape(1);

    bool flip = false;
    if (shift < 0) {
        std::swap(id0, id1);
        shift = -shift;
        flip = true;
    }

    int sign = sub ? -1 : 1;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> result;

    for (size_t i_out = 0; i_out < n_out; ++i_out) {
        const auto &row0 = state.expr[id0].rows[i_out];
        const auto &row1 = state.expr[id1].rows[i_out];
        // Two-pointer merge on sorted rows
        size_t i0 = 0, i1 = 0;
        while (i0 < row0.size() && i1 < row1.size()) {
            int8_t shift0 = SparseExpr::to_shift(row0[i0]);
            int8_t j1 = SparseExpr::to_shift(row1[i1]);
            int8_t target_shift1 = shift0 + shift;
            if (j1 < target_shift1) {
                ++i1;
                continue;
            }
            if (j1 > target_shift1) {
                ++i0;
                continue;
            }
            // j1_val == target_j1
            int8_t bit0 = SparseExpr::to_sign(row0[i0]);
            int8_t bit1 = SparseExpr::to_sign(row1[i1]);
            if (sign * bit1 * bit0 == 1) {
                if (flip)
                    result.emplace_back(i_out, target_shift1, shift0);
                else
                    result.emplace_back(i_out, shift0, target_shift1);
            }
            ++i0;
            ++i1;
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
    Op op = pair_to_op(pair, state, adder_size, carry_size);
    size_t n_out = state.kernel.shape(1);

    state.ops.push_back(op);

    SparseExpr new_slice;
    new_slice.rows.resize(n_out);

    int64_t id0 = pair.id0, id1 = pair.id1;
    int8_t rel_shift = pair.shift;
    bool sub = pair.sub;

    bool flip = false;
    if (rel_shift < 0) {
        std::swap(id0, id1);
        rel_shift = -rel_shift;
        flip = true;
    }

    int8_t target_sign = sub ? -1 : 1;

    for (size_t i_out = 0; i_out < n_out; ++i_out) {
        SparseExpr &expr0 = state.expr[id0];
        SparseExpr &expr1 = state.expr[id1];
        for (int8_t bit_loc0 = 0; bit_loc0 < (int8_t)expr0.rows[i_out].size();
             ++bit_loc0) {
            if (expr0.rows[i_out][bit_loc0] == 0)
                continue; // marked for removal
            auto [shift0, sign0] = expr0.pos_sign(i_out, bit_loc0);
            int8_t shift1 = shift0 + rel_shift;
            int8_t bit_loc1 = expr1.to_idx(i_out, shift1);
            if (shift1 >= state.n_bits)
                continue;
            int8_t sign1 =
                bit_loc1 >= 0 ? SparseExpr::to_sign(expr1.rows[i_out][bit_loc1]) : 0;
            if (target_sign * sign1 * sign0 != 1)
                continue;

            // Match found.
            if (!flip) {
                new_slice.set_bit(i_out, shift0, sign0);
            }
            else {
                new_slice.set_bit(i_out, shift1, sign1);
            }
            expr0.set_removal(i_out, bit_loc0);
            expr1.set_removal(i_out, bit_loc1);
        }

        expr0.compact(i_out);
        if (id0 != id1)
            expr1.compact(i_out);
    }

    state.expr.push_back(std::move(new_slice));
}

void update_stats(DAState &state, const Pair &pair) {
    int64_t id0 = pair.id0, id1 = pair.id1;

    // id0/1 are dirty now, purge all entries involving them.
    // Uses contiguous scan over FreqMap's flat vector instead of chasing
    // unordered_map node pointers.
    state.freq_stat.erase_if([&](const auto &e) {
        const Pair &p = e.first;
        return p.id0 == id0 || p.id0 == id1 || p.id1 == id0 || p.id1 == id1;
    });

    int64_t n_constructed = static_cast<int64_t>(state.expr.size());
    std::vector modified = {n_constructed - 1, id0};
    if (id0 != id1)
        modified.push_back(id1);

    size_t n_out = state.kernel.shape(1);

    // Collect all new pairs into a flat vector, then batch-merge.
    // This avoids per-pair hash lookups in the inner loops.
    std::vector<Pair> raw_pairs;

    for (size_t i_out = 0; i_out < n_out; ++i_out) {
        for (int64_t _in1 = 0; _in1 < n_constructed; ++_in1) {
            for (auto _in0 : modified) {
                if ((_in1 == n_constructed - 1 || _in1 == id0 || _in1 == id1) &&
                    _in0 > _in1)
                    continue;

                int64_t lo = std::min(_in0, _in1);
                int64_t hi = std::max(_in0, _in1);

                const auto &row_lo = state.expr[lo].rows[i_out];
                const auto &row_hi = state.expr[hi].rows[i_out];

                if (row_lo.empty() || row_hi.empty())
                    continue;

                if (lo == hi) {
                    const size_t sz = row_lo.size();
                    for (size_t a = 1; a < sz; ++a) {
                        for (size_t b = 0; b < a; ++b) {
                            raw_pairs.push_back(_make_pair(lo, lo, row_lo[a], row_lo[b]));
                        }
                    }
                }
                else {
                    for (int8_t v_hi : row_lo) {
                        for (int8_t v_lo : row_hi) {
                            raw_pairs.push_back(_make_pair(lo, hi, v_hi, v_lo));
                        }
                    }
                }
            }
        }
    }

    // Sort, deduplicate, count, and append — only pairs with count >= 2
    // are inserted. No merge needed since purge guarantees no overlap.
    state.freq_stat.batch_add(raw_pairs);
}

void update_state(DAState &state, const Pair &pair, int adder_size, int carry_size) {
    update_expr(state, pair, adder_size, carry_size);
    update_stats(state, pair);
}

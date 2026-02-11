#pragma once

#include <cstdint>
#include <cstddef>
#include <sys/types.h>
#include <vector>
#include <utility>
#include <xtensor/containers/xarray.hpp>

#ifndef __STDCPP_FLOAT32_T__
#define __STDCPP_FLOAT32_T__
#endif
#include <stdfloat>

struct QInterval {
    double min, max, step;
};

struct Op {
    int64_t id0, id1;
    int64_t opcode;
    int64_t data;
    QInterval qint;
    double latency;
    double cost;
};

struct Pair {
    int64_t id0, id1;
    int8_t shift;
    bool sub;
    bool operator==(const Pair &) const = default;
    bool operator<(const Pair &o) const {
        if (id1 != o.id1)
            return id1 < o.id1;
        if (id0 != o.id0)
            return id0 < o.id0;
        if (sub != o.sub)
            return sub < o.sub;
        return shift < o.shift;
    }
};

class FreqMap {
  public:
    using value_type = std::pair<Pair, uint32_t>;

  private:
    std::vector<value_type> entries_;

  public:
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    auto begin() { return entries_.begin(); }
    auto end() { return entries_.end(); }
    auto begin() const { return entries_.begin(); }
    auto end() const { return entries_.end(); }

    void reserve(size_t n) { entries_.reserve(n); }

    template <typename Pred> void erase_if(Pred &&pred) {
        auto new_end = std::remove_if(entries_.begin(), entries_.end(), pred);
        entries_.erase(new_end, entries_.end());
    }
    void merge_sorted(const std::vector<value_type> &new_entries) {
        std::vector<value_type> merged(entries_.size() + new_entries.size());
        std::merge(
            entries_.begin(),
            entries_.end(),
            new_entries.begin(),
            new_entries.end(),
            merged.begin(),
            [](const value_type &a, const value_type &b) { return a.first < b.first; }
        );
        entries_ = std::move(merged);
    }
    void batch_add(std::vector<Pair> &raw) {
        std::sort(raw.begin(), raw.end());
        std::vector<value_type> new_entries;
        size_t n = raw.size();
        uint32_t count = 1;
        for (size_t i = 0; i < n; ++i) {
            if (i + 1 < n && raw[i] == raw[i + 1]) {
                count++;
            }
            else {
                if (count >= 2) {
                    new_entries.emplace_back(raw[i], count);
                }
                count = 1;
            }
        };
        std::sort(
            new_entries.begin(),
            new_entries.end(),
            [](const value_type &a, const value_type &b) { return a.first < b.first; }
        );
        merge_sorted(new_entries);
    }

    void initialize(std::vector<Pair> &raw) {
        entries_.clear();
        batch_add(raw);
    }
};

// Sparse CSD expression slice.
class SparseExpr {
  public:
    std::vector<std::vector<int8_t>> rows; // rows[i_out][j_bit_idx]

    const static int8_t to_shift(int8_t v) { return abs(v) - 1; }
    const static int8_t to_sign(int8_t v) { return v > 0 ? 1 : -1; }

    const std::pair<int8_t, int8_t> pos_sign(size_t i_out, int8_t bit_pos) {
        int8_t v = rows[i_out][bit_pos];
        return {to_shift(v), to_sign(v)};
    }

    void set_bit(size_t i_out, int8_t shift, int8_t sign) {
        int8_t v = sign * (shift + 1); // sign: +/-1. encoded as +/- (shift + 1)
        rows[i_out].push_back(v);
    }
    void set_removal(size_t i_out, int8_t j_bit) {
        int8_t v = 0; // sentinel for removal
        rows[i_out][j_bit] = 0;
    }
    void compact(size_t i_out) {
        rows[i_out].erase(
            std::remove(rows[i_out].begin(), rows[i_out].end(), (int8_t)0),
            rows[i_out].end()
        );
    }
    void compact_all() {
        for (auto &r : rows) {
            r.erase(std::remove(r.begin(), r.end(), (int8_t)0), r.end());
        }
    }
    int8_t to_idx(size_t i_out, int8_t shift) {
        for (int8_t i = 0; i < (int8_t)rows[i_out].size(); ++i) {
            if (to_shift(rows[i_out][i]) == shift)
                return i;
        }
        return -1;
    }
};

struct DAState {
    xt::xarray<int8_t> shift0; // input shifts
    xt::xarray<int8_t> shift1; // output shifts
    std::vector<SparseExpr> expr;
    size_t n_bits; // bit-width of CSD representation
    std::vector<Op> ops;
    FreqMap freq_stat;
    xt::xarray<std::float32_t> kernel;
};

struct CombLogicResult {
    std::pair<int64_t, int64_t> shape;
    std::vector<int64_t> inp_shifts;
    std::vector<int64_t> out_idxs;
    std::vector<int64_t> out_shifts;
    std::vector<int64_t> out_negs;
    std::vector<Op> ops;
    int carry_size;
    int adder_size;
};

struct PipelineResult {
    std::vector<CombLogicResult> solutions;
};

enum struct Method { MC, MC_DC, MC_PDC, WMC, WMC_DC, WMC_PDC, DUMMY };

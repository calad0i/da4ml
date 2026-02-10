#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <functional>
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
    bool sub;
    int64_t shift;
    bool operator==(const Pair &) const = default;
};

struct PairHash {
    size_t operator()(const Pair &p) const {
        size_t h = std::hash<int64_t>{}(p.id0);
        h ^= std::hash<int64_t>{}(p.id1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(p.sub) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(p.shift) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Ordered map preserving insertion order (Python dict semantics).
// Re-inserting a deleted key places it at the end.
class Map {
    struct Entry {
        Pair key;
        int value;
        bool alive;
    };
    std::vector<Entry> entries_;
    std::unordered_map<Pair, size_t, PairHash> lookup_;
    size_t alive_count_ = 0;

  public:
    size_t size() const { return alive_count_; }
    bool empty() const { return alive_count_ == 0; }

    void set(const Pair &key, int value) {
        auto it = lookup_.find(key);
        if (it != lookup_.end() && entries_[it->second].alive) {
            // Key exists and alive: update in-place
            entries_[it->second].value = value;
        }
        else {
            // New key or re-insertion after delete: append at end
            lookup_[key] = entries_.size();
            entries_.push_back({key, value, true});
            alive_count_++;
        }
    }

    int get(const Pair &key, int default_val) const {
        auto it = lookup_.find(key);
        if (it != lookup_.end() && entries_[it->second].alive) {
            return entries_[it->second].value;
        }
        return default_val;
    }

    void erase(const Pair &key) {
        auto it = lookup_.find(key);
        if (it != lookup_.end() && entries_[it->second].alive) {
            entries_[it->second].alive = false;
            alive_count_--;
        }
    }

    std::vector<Pair> keys() const {
        std::vector<Pair> result;
        result.reserve(alive_count_);
        for (auto &e : entries_) {
            if (e.alive)
                result.push_back(e.key);
        }
        return result;
    }

    std::vector<int> values() const {
        std::vector<int> result;
        result.reserve(alive_count_);
        for (auto &e : entries_) {
            if (e.alive)
                result.push_back(e.value);
        }
        return result;
    }

    Pair key_at(size_t idx) const {
        size_t count = 0;
        for (auto &e : entries_) {
            if (e.alive) {
                if (count == idx)
                    return e.key;
                count++;
            }
        }
        throw std::out_of_range("OrderedMap::key_at");
    }
};

struct DAState {
    xt::xarray<int8_t> shift0; // input shifts
    xt::xarray<int8_t> shift1; // output shifts
    std::vector<xt::xarray<int8_t>> expr;
    std::vector<Op> ops;
    std::unordered_map<Pair, int32_t, PairHash> freq_stat;
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

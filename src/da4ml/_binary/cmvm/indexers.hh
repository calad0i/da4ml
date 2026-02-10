#pragma once

#include "types.hh"
#include <utility>

int64_t idx_mc(const DAState &state);
int64_t idx_mc_dc(const DAState &state, bool absolute = false);
std::pair<double, double> overlap_and_accum(const QInterval &q0, const QInterval &q1);
int64_t idx_wmc(const DAState &state);
int64_t idx_wmc_dc(const DAState &state, bool absolute = false);

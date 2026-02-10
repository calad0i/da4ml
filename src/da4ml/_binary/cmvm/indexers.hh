#pragma once

#include "types.hh"
#include <utility>

Pair idx_mc(const DAState &state);
Pair idx_mc_dc(const DAState &state, bool absolute = false);
std::pair<double, double> overlap_and_accum(const QInterval &q0, const QInterval &q1);
Pair idx_wmc(const DAState &state);
Pair idx_wmc_dc(const DAState &state, bool absolute = false);

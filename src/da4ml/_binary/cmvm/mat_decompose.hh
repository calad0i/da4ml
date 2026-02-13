#pragma once

#include "bit_decompose.hh"
#include <xtensor/containers/xarray.hpp>

xt::xarray<int32_t> prim_mst_dc(const xt::xarray<int64_t> &cost_mat, int dc = -1);

std::pair<xt::xarray<float>, xt::xarray<float>>
kernel_decompose(xt::xarray<float> kernel, int dc = -2);

nb::tuple kernel_decompose_numpy(const nb::ndarray<float> &in, int dc = -2);

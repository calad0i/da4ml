#pragma once

#include "bit_decompose.hh"
#include <xtensor/containers/xarray.hpp>

xt::xarray<int32_t> prim_mst_dc(const xt::xarray<int64_t> &cost_mat, int dc = -1);

std::pair<xt::xarray<std::float32_t>, xt::xarray<std::float32_t>>
kernel_decompose(xt::xarray<std::float32_t> kernel, int dc = -2);

nb::tuple kernel_decompose_numpy(const nb::ndarray<std::float32_t> &in, int dc = -2);

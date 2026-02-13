#pragma once

#include <cstdint>
#include <xtensor/core/xtensor_forward.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>

namespace nb = nanobind;
using namespace nb::literals;

int8_t get_lsb_loc(float x);

xt::xarray<int8_t> _volatile_int_arr_to_csd(xt::xarray<int32_t> &x);

template <typename T> inline auto _shift_amount(T &x, int32_t axis) {
    return xt::amin(xt::vectorize(get_lsb_loc)(x), axis);
}

template <typename T> auto _center(T &arr) {
    if (arr.dimension() != 2) {
        throw std::runtime_error("csd_decompose only supports 2D arrays.");
    }
    xt::xarray<int8_t> shift1 = _shift_amount(arr, 0);
    arr = arr * xt::pow(2.0f, -shift1);
    xt::xarray<int8_t> shift0 = _shift_amount(arr, 1);
    arr = arr * xt::view(xt::pow(2.0f, -shift0), xt::all(), xt::newaxis());
    return std::make_tuple(arr, shift0, shift1);
}

std::tuple<xt::xarray<int8_t>, xt::xarray<int8_t>, xt::xarray<int8_t>>
csd_decompose(xt::xarray<float> &arr, bool center = true);

#pragma once

#include <xtensor/core/xtensor_forward.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/containers/xtensor.hpp>

#ifndef __STDCPP_FLOAT32_T__
#define __STDCPP_FLOAT32_T__
#endif
#include <stdfloat>
#include <xtensor/io/xio.hpp>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
concept fp32_container = std::same_as<typename T::value_type, std::float32_t>;

template <typename T>
concept int32_container = std::same_as<typename T::value_type, int32_t>;

template <typename T> xt::xarray<int8_t> _volatile_int_arr_to_csd(T &x);

int8_t get_lsb_loc(std::float32_t x);

template <fp32_container T> inline auto _shift_amount(T &x, int32_t axis) {
    return xt::amin(xt::vectorize(get_lsb_loc)(x), axis);
}

template <fp32_container T> auto _center(T &arr);

template <fp32_container T> auto csd_decompose(T &arr, bool center = true);

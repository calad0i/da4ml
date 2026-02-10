#pragma once

#include <xtensor/core/xtensor_forward.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>

#ifndef __STDCPP_FLOAT32_T__
#define __STDCPP_FLOAT32_T__
#endif
#include <stdfloat>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
concept fp32_container = std::same_as<typename T::value_type, std::float32_t>;

template <typename T>
concept int32_container = std::same_as<typename T::value_type, int32_t>;

int8_t get_lsb_loc(std::float32_t x);

// --- Template implementations (in header for cross-TU use) ---

template <int32_container T> xt::xarray<int8_t> _volatile_int_arr_to_csd(T &x) {
    int32_t max_val = static_cast<int32_t>(xt::amax(xt::abs(x))());
    size_t N = static_cast<size_t>(
        std::ceil(std::log2(static_cast<float>(max_val) * 1.5 + 1e-19))
    );
    N = std::max(N, size_t(1));
    auto out_shape = x.shape();
    out_shape.push_back(N);

    x.reshape({x.size()});
    xt::xarray<int8_t> buf = xt::empty<int8_t>({x.size(), N});
    for (int n = N - 1; n >= 0; --n) {
        int32_t _2pn = static_cast<int32_t>(1U << n);
        int32_t thres = _2pn * 2 / 3;
        auto slice = xt::view(buf, xt::all(), n);
        slice = xt::cast<int8_t>(x > thres) - xt::cast<int8_t>(x < -thres);
        x -= _2pn * xt::cast<int32_t>(slice);
    }
    buf.reshape(out_shape);
    return buf;
}

template <fp32_container T> inline auto _shift_amount(T &x, int32_t axis) {
    return xt::amin(xt::vectorize(get_lsb_loc)(x), axis);
}

template <fp32_container T> auto _center(T &arr) {
    if (arr.dimension() != 2) {
        throw std::runtime_error("csd_decompose only supports 2D arrays.");
    }
    xt::xarray<int8_t> shift1 = _shift_amount(arr, 0);
    arr = arr * xt::pow(2.0f, -shift1);
    xt::xarray<int8_t> shift0 = _shift_amount(arr, 1);
    arr = arr * xt::view(xt::pow(2.0f, -shift0), xt::all(), xt::newaxis());
    return std::make_tuple(arr, shift0, shift1);
}

template <fp32_container T> auto csd_decompose(T &arr, bool center = true) {
    xt::xarray<std::float32_t> arr_cpy(arr);
    if (arr_cpy.dimension() != 2) {
        throw std::runtime_error("csd_decompose only supports 2D arrays.");
    }
    xt::xarray<int8_t> shift0 = xt::empty<int8_t>({arr_cpy.shape(0)});
    xt::xarray<int8_t> shift1 = xt::empty<int8_t>({arr_cpy.shape(1)});
    if (center) {
        std::tie(arr_cpy, shift0, shift1) = _center(arr_cpy);
    }
    else {
        shift0.fill(0);
        shift1.fill(0);
    }
    xt::xarray<int32_t> arr_int = xt::cast<int32_t>(arr_cpy);
    auto csd = _volatile_int_arr_to_csd(arr_int);
    return std::make_tuple(csd, shift0, shift1);
}

// Numpy wrapper declarations
nb::ndarray<nb::numpy, int8_t>
_volatile_int_arr_to_csd_numpy(const nb::ndarray<int32_t> &in);

nb::tuple csd_decompose_numpy(const nb::ndarray<std::float32_t> &in, bool center = true);

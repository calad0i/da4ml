#include "bit_decompose.hh"
#include "xtensor/io/xio.hpp"

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

nb::ndarray<nb::numpy, int8_t>
_volatile_int_arr_to_csd_numpy(const nb::ndarray<int32_t> &in) {
    size_t ndim = in.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = in.shape(i);
    }
    auto arr =
        xt::adapt(const_cast<int32_t *>(in.data()), in.size(), xt::no_ownership(), shape);
    auto *result = new xt::xarray(_volatile_int_arr_to_csd(arr));
    auto *out_ptr = result->data();
    nb::capsule owner(result, [](void *p) noexcept {
        delete static_cast<xt::xarray<int8_t> *>(p);
    });
    std::vector<size_t> out_shape(result->shape().begin(), result->shape().end());
    return nb::ndarray<nb::numpy, int8_t>(
        out_ptr, out_shape.size(), out_shape.data(), owner
    );
}

int8_t get_lsb_loc(std::float32_t x) {
    // s1, m24, e7
    if (x == 0.0f) {
        return 127;
    }
    uint32_t bits = std::bit_cast<uint32_t>(x);
    uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    int mtz = __builtin_ctz(mant + (1 << 23));

    return static_cast<int8_t>(exp + mtz - 150);
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

template <fp32_container T> auto csd_decompose(T &arr, bool center) {
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

nb::tuple csd_decompose_numpy(const nb::ndarray<std::float32_t> &in, bool center = true) {
    size_t ndim = in.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = in.shape(i);
    }
    auto arr = xt::adapt(
        const_cast<std::float32_t *>(in.data()), in.size(), xt::no_ownership(), shape
    );
    auto [csd, shift0, shift1] = csd_decompose(arr, center);

    // Prepare outputs
    auto *csd_ptr = new xt::xarray(csd);
    auto *shift0_ptr = new xt::xarray(shift0);
    auto *shift1_ptr = new xt::xarray(shift1);

    nb::capsule csd_owner(csd_ptr, [](void *p) noexcept {
        delete static_cast<xt::xarray<std::int8_t> *>(p);
    });
    nb::capsule shift0_owner(shift0_ptr, [](void *p) noexcept {
        delete static_cast<xt::xarray<std::int8_t> *>(p);
    });
    nb::capsule shift1_owner(shift1_ptr, [](void *p) noexcept {
        delete static_cast<xt::xarray<std::int8_t> *>(p);
    });

    std::vector<size_t> csd_shape(csd.shape().begin(), csd.shape().end());
    std::vector<size_t> shift0_shape(shift0.shape().begin(), shift0.shape().end());
    std::vector<size_t> shift1_shape(shift1.shape().begin(), shift1.shape().end());

    nb::ndarray<nb::numpy, std::int8_t> csd_out(
        csd_ptr->data(), csd_shape.size(), csd_shape.data(), csd_owner
    );
    nb::ndarray<nb::numpy, std::int8_t> shift0_out(
        shift0_ptr->data(), shift0_shape.size(), shift0_shape.data(), shift0_owner
    );
    nb::ndarray<nb::numpy, std::int8_t> shift1_out(
        shift1_ptr->data(), shift1_shape.size(), shift1_shape.data(), shift1_owner
    );
    return nb::make_tuple(csd_out, shift0_out, shift1_out);
}

NB_MODULE(cmvm_bin, m) {
    m.def(
        "_volatile_int_arr_to_csd", &_volatile_int_arr_to_csd_numpy, "in"_a.noconvert()
    );
    m.def("get_lsb_loc", &get_lsb_loc, "x"_a);

    m.def("csd_decompose", &csd_decompose_numpy, "in"_a.noconvert(), "center"_a = true);
}

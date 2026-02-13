#include "bit_decompose.hh"

int8_t get_lsb_loc(float x) {
    // s1, m24, e7
    if (x == 0.0f) {
        return 127;
    }
    uint32_t bits = std::bit_cast<uint32_t>(_Float32(x));
    uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    int mtz = __builtin_ctz(mant + (1 << 23));
    return static_cast<int8_t>(exp + mtz - 150);
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

nb::tuple csd_decompose_numpy(const nb::ndarray<float> &in, bool center) {
    size_t ndim = in.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = in.shape(i);
    }
    auto arr =
        xt::adapt(const_cast<float *>(in.data()), in.size(), xt::no_ownership(), shape);
    auto [csd, shift0, shift1] = csd_decompose(arr, center);

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

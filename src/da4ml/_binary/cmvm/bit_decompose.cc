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

xt::xarray<int8_t> _volatile_int_arr_to_csd(xt::xarray<int32_t> &x) {
    int32_t max_val = static_cast<int32_t>(xt::amax(xt::abs(x))());
    size_t N = static_cast<size_t>(
        std::ceil(std::log2(std::max(static_cast<float>(max_val), 1.0f) * 1.5))
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

std::tuple<xt::xarray<int8_t>, xt::xarray<int8_t>, xt::xarray<int8_t>>
csd_decompose(xt::xarray<float> &arr, bool center) {
    xt::xarray<float> arr_cpy(arr);
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

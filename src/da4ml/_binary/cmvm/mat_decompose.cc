#include "mat_decompose.hh"
#include <cmath>
#include <limits>
#include <algorithm>

xt::xarray<int32_t> prim_mst_dc(const xt::xarray<int64_t> &cost_mat, int dc) {
    size_t N = cost_mat.shape(0);
    auto lat_mat =
        xt::cast<double>(xt::ceil(xt::log2(xt::cast<double>(xt::maximum(cost_mat, 1)))));
    std::vector<int32_t> parent(N, -2);
    parent[0] = -1;

    xt::xarray<int32_t> mapping = xt::empty<int32_t>({N - 1, size_t(2)});
    std::vector<int32_t> latency(N, 0);

    double _dc = -1;
    if (dc >= 0) {
        double max_cost0 = xt::amax(xt::view(cost_mat, 0, xt::all()))();
        _dc = (std::pow(2.0, dc) - 1) + std::ceil(std::log2(max_cost0 + 1e-32));
    }

    for (size_t n_impl = 1; n_impl < N; ++n_impl) {
        // Build index arrays for implemented and not-implemented
        std::vector<size_t> not_impl, impl;
        for (size_t i = 0; i < N; ++i) {
            if (parent[i] != -2)
                impl.push_back(i);
            else
                not_impl.push_back(i);
        }

        // Find minimum cost edge
        int64_t best_cost = std::numeric_limits<int64_t>::max();
        size_t best_i = 0, best_j = 0;
        for (size_t ii = 0; ii < not_impl.size(); ++ii) {
            for (size_t jj = 0; jj < impl.size(); ++jj) {
                size_t i = not_impl[ii], j = impl[jj];
                int64_t c = cost_mat(i, j);
                if (dc >= 0) {
                    double lat = lat_mat(i, j);
                    double max_lat = std::max(lat, (double)latency[j]) + 1;
                    if (max_lat > _dc) {
                        c = std::numeric_limits<int64_t>::max() / 2;
                    }
                }
                if (c < best_cost) {
                    best_cost = c;
                    best_i = ii;
                    best_j = jj;
                }
            }
        }
        size_t i = not_impl[best_i], j = impl[best_j];
        parent[i] = static_cast<int32_t>(j);
        mapping(n_impl - 1, 0) = static_cast<int32_t>(j);
        mapping(n_impl - 1, 1) = static_cast<int32_t>(i);
        latency[i] =
            static_cast<int32_t>(std::max(lat_mat(i, j), (double)latency[j]) + 1);
    }
    return mapping;
}

std::pair<xt::xarray<float>, xt::xarray<float>>
kernel_decompose(xt::xarray<float> kernel, int dc) {
    auto [centered, shift0, shift1] = _center(kernel);
    auto scale0 = xt::pow(2.0f, xt::cast<float>(shift0));
    auto scale1 = xt::pow(2.0f, xt::cast<float>(shift1));

    size_t m = centered.shape(0);
    size_t n = centered.shape(1) + 1;
    xt::xarray<float> mat_aug = xt::zeros<float>({m, n});
    xt::view(mat_aug, xt::all(), xt::range(1, n)) = centered;

    // diff0[i, a, b] = mat_aug[i, a] - mat_aug[i, b]
    // diff1[i, a, b] = mat_aug[i, a] + mat_aug[i, b]
    xt::xarray<float> diff0 = xt::view(mat_aug, xt::all(), xt::all(), xt::newaxis()) -
                              xt::view(mat_aug, xt::all(), xt::newaxis(), xt::all());
    xt::xarray<float> diff1 = xt::view(mat_aug, xt::all(), xt::all(), xt::newaxis()) +
                              xt::view(mat_aug, xt::all(), xt::newaxis(), xt::all());

    // CSD Hamming weight
    xt::xarray<int32_t> diff0_int = xt::cast<int32_t>(diff0);
    xt::xarray<int32_t> diff1_int = xt::cast<int32_t>(diff1);
    auto csd0 = _volatile_int_arr_to_csd(diff0_int);
    auto csd1 = _volatile_int_arr_to_csd(diff1_int);
    // Sum of non-zero bits over last axis (bit dimension) and first axis (input dimension)
    auto dist0 =
        xt::sum(xt::sum(xt::cast<int64_t>(xt::not_equal(csd0, int8_t(0))), {3}), {0});
    auto dist1 =
        xt::sum(xt::sum(xt::cast<int64_t>(xt::not_equal(csd1, int8_t(0))), {3}), {0});

    auto sign_arr =
        xt::where(dist1 - dist0 < 0, xt::xarray<int64_t>({-1}), xt::xarray<int64_t>({1}));
    xt::xarray<int64_t> dist = xt::minimum(dist0, dist1);

    auto mapping_arr = prim_mst_dc(dist, dc);

    size_t n_in = centered.shape(0), n_out = centered.shape(1);
    xt::xarray<float> m0 = xt::zeros<float>({n_in, n_out});
    xt::xarray<float> m1 = xt::zeros<float>({n_out, n_out});

    if (dc == -1) {
        m0 = centered;
        m1 = xt::eye<float>(n_out);
        m0 = m0 * xt::view(scale0, xt::all(), xt::newaxis());
        m1 = m1 * scale1;
        return {m0, m1};
    }

    size_t cnt = 0;
    for (size_t k = 0; k < mapping_arr.shape(0); ++k) {
        int32_t _from = mapping_arr(k, 0);
        int32_t _to = mapping_arr(k, 1);
        auto col0 = xt::view(mat_aug, xt::all(), _to) -
                    xt::view(mat_aug, xt::all(), _from) *
                        static_cast<float>(sign_arr(_to, _from));

        xt::xarray<float> col1;
        if (_from != 0) {
            col1 = xt::view(m1, xt::all(), _from - 1) *
                   static_cast<float>(sign_arr(_to, _from));
        }
        else {
            col1 = xt::zeros<float>({n_out});
        }

        if (xt::any(xt::not_equal(col0, 0.0f))) {
            col1(cnt) = 1.0f;
            xt::view(m0, xt::all(), cnt) = col0;
            cnt++;
        }
        xt::view(m1, xt::all(), _to - 1) = col1;
    }

    m0 = m0 * xt::view(scale0, xt::all(), xt::newaxis());
    m1 = m1 * scale1;
    return {m0, m1};
}

nb::tuple kernel_decompose_numpy(const nb::ndarray<float> &in, int dc) {
    size_t ndim = in.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i)
        shape[i] = in.shape(i);

    auto arr =
        xt::adapt(const_cast<float *>(in.data()), in.size(), xt::no_ownership(), shape);
    auto [m0, m1] = kernel_decompose(xt::xarray<float>(arr), dc);

    auto *m0_ptr = new xt::xarray<float>(m0);
    auto *m1_ptr = new xt::xarray<float>(m1);

    nb::capsule m0_owner(m0_ptr, [](void *p) noexcept {
        delete static_cast<xt::xarray<float> *>(p);
    });
    nb::capsule m1_owner(m1_ptr, [](void *p) noexcept {
        delete static_cast<xt::xarray<float> *>(p);
    });

    std::vector<size_t> m0_shape(m0.shape().begin(), m0.shape().end());
    std::vector<size_t> m1_shape(m1.shape().begin(), m1.shape().end());

    nb::ndarray<nb::numpy, float> m0_out(
        m0_ptr->data(), m0_shape.size(), m0_shape.data(), m0_owner
    );
    nb::ndarray<nb::numpy, float> m1_out(
        m1_ptr->data(), m1_shape.size(), m1_shape.data(), m1_owner
    );
    return nb::make_tuple(m0_out, m1_out);
}

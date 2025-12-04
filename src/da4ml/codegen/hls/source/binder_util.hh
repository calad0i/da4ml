#pragma once
#include <cstddef>

#ifdef _OPENMP
#include <algorithm>
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

template <typename CONFIG_T, typename T>
void _inference(T *c_inp, T *c_out, size_t n_samples) {
    typename CONFIG_T::inp_t in_fixed_buf[CONFIG_T::N_inp];
    typename CONFIG_T::out_t out_fixed_buf[CONFIG_T::N_out];

    for (size_t i = 0; i < n_samples; ++i) {
        size_t offset_in = i * CONFIG_T::N_inp;
        size_t offset_out = i * CONFIG_T::N_out;
        for (size_t j = 0; j < CONFIG_T::N_inp; ++j) {
            in_fixed_buf[j] = c_inp[offset_in + j];
        }

        CONFIG_T::f(in_fixed_buf, out_fixed_buf);

        for (size_t j = 0; j < CONFIG_T::N_out; ++j) {
            c_out[offset_out + j] = out_fixed_buf[j];
        }
    }
}

template <typename CONFIG_T, typename T>
void batch_inference(T *c_inp, T *c_out, size_t n_samples, size_t n_threads) {
    if (n_threads > 1 || n_threads == 0) {
#ifdef _OPENMP
        size_t min_samples_per_thread;
        size_t n_max_threads;
        if (n_threads == 0) {
            min_samples_per_thread = 1;
            n_max_threads = omp_get_max_threads();
        }
        else {
            min_samples_per_thread = std::max<size_t>(1, n_samples / n_threads);
            n_max_threads = n_threads;
        }
        size_t n_sample_per_thread =
            n_samples / n_max_threads + (n_samples % n_max_threads ? 1 : 0);
        n_sample_per_thread =
            std::max<size_t>(n_sample_per_thread, min_samples_per_thread);
        size_t n_thread = n_samples / n_sample_per_thread;
        n_thread += (n_samples % n_sample_per_thread) ? 1 : 0;

#pragma omp parallel for num_threads(n_thread) schedule(static)
        for (size_t i = 0; i < n_thread; ++i) {
            size_t start = i * n_sample_per_thread;
            size_t end = std::min<size_t>(start + n_sample_per_thread, n_samples);
            size_t n_samples_this_thread = end - start;
            size_t offset_in = start * CONFIG_T::N_inp;
            size_t offset_out = start * CONFIG_T::N_out;
            _inference<CONFIG_T, T>(
                &c_inp[offset_in], &c_out[offset_out], n_samples_this_thread
            );
        }
#else
        _inference<CONFIG_T, T>(c_inp, c_out, n_samples);
#endif
    }
    else {
        _inference<CONFIG_T, T>(c_inp, c_out, n_samples);
    }
}

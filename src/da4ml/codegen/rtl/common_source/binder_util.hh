#include "ioutil.hh"
#include <verilated.h>

#ifdef _OPENMP
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

template <typename CONFIG_T>
std::enable_if_t<CONFIG_T::II != 0>
_inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {
    auto dut = std::make_unique<typename CONFIG_T::dut_t>();

    size_t clk_req = n_samples * CONFIG_T::II + (CONFIG_T::latency - CONFIG_T::II) + 1;

    for (size_t t_inp = 0; t_inp < clk_req; ++t_inp) {
        size_t t_out = t_inp - CONFIG_T::latency;

        if (t_inp < n_samples * CONFIG_T::II && t_inp % CONFIG_T::II == 0) {
            write_input<CONFIG_T::N_inp, CONFIG_T::max_inp_bw>(
                dut->model_inp, &c_inp[t_inp / CONFIG_T::II * CONFIG_T::N_inp]
            );
        }

        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();

        if (t_inp >= CONFIG_T::latency && t_out % CONFIG_T::II == 0) {
            read_output<CONFIG_T::N_out, CONFIG_T::max_out_bw>(
                dut->model_out, &c_out[t_out / CONFIG_T::II * CONFIG_T::N_out]
            );
        }
    }

    dut->final();
}

template <typename CONFIG_T>
std::enable_if_t<CONFIG_T::II == 0>
_inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {
    auto dut = std::make_unique<typename CONFIG_T::dut_t>();

    for (size_t i = 0; i < n_samples; ++i) {
        write_input<CONFIG_T::N_inp, CONFIG_T::max_inp_bw>(
            dut->model_inp, &c_inp[i * CONFIG_T::N_inp]
        );
        dut->eval();
        read_output<CONFIG_T::N_out, CONFIG_T::max_out_bw>(
            dut->model_out, &c_out[i * CONFIG_T::N_out]
        );
    }

    dut->final();
}

template <typename CONFIG_T>
void batch_inference(int32_t *c_inp, int32_t *c_out, size_t n_samples, size_t n_threads) {
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
            _inference<CONFIG_T>(
                &c_inp[offset_in], &c_out[offset_out], n_samples_this_thread
            );
        }
#else
        _inference<CONFIG_T>(c_inp, c_out, n_samples);
#endif
    }
    else {
        _inference<CONFIG_T>(c_inp, c_out, n_samples);
    }
}

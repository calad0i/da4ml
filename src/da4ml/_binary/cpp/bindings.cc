#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "DAISInterpreter.hh"
#include <cstring>
#include <omp.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

void _run_interp(
    const std::vector<int32_t> &binary_data,
    const std::span<const double_t> &inputs,
    std::span<double_t> &outputs,
    size_t n_samples
) {
    int32_t n_in = binary_data[2];
    int32_t n_out = binary_data[3];
    dais::DAISInterpreter interp;
    interp.load_from_binary(binary_data);

    for (size_t i = 0; i < n_samples; ++i) {
        const std::span<const double_t> inp_span(&inputs[i * n_in], n_in);
        std::span<double_t> out_span(&outputs[i * n_out], n_out);
        interp.inference(inp_span, out_span);
    }
}

void run_interp(
    const std::vector<int32_t> &bin_logic,
    const std::vector<double_t> &input,
    std::span<double_t> &output,
    int64_t n_threads
) {
    const int32_t *bin_logic_ptr = bin_logic.data();
    const double_t *input_ptr = input.data();
    if (bin_logic.size() < 4) {
        throw std::runtime_error("Invalid binary logic data");
    }

    // =============== version check and init ===============

    int32_t spec_version = bin_logic_ptr[0];
    if (spec_version != dais::DAISInterpreter::dais_version) {
        throw std::runtime_error(
            "DAIS version mismatch: expected version " +
            std::to_string(dais::DAISInterpreter::dais_version) + ", got version " +
            std::to_string(spec_version)
        );
    }

    const std::vector<int32_t> bin_logic_vec(
        bin_logic_ptr, bin_logic_ptr + bin_logic.size()
    );
    const std::vector<double_t> data_vec(input_ptr, input_ptr + input.size());

    int32_t n_in = bin_logic[2];
    int32_t n_out = bin_logic[3];

    // =============== openmp config ===============

    size_t n_samples = input.size() / n_in;

    size_t n_max_threads = std::max<size_t>(n_threads, omp_get_max_threads());
    size_t n_samples_per_thread = std::max<size_t>(n_samples / n_max_threads, 32);
    size_t n_thread = n_samples / n_samples_per_thread;
    n_thread += (n_samples % n_samples_per_thread) ? 1 : 0;

    // =============== exec ===============

#pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < n_thread; ++i) {
        size_t start = i * n_samples_per_thread;
        size_t end = std::min<size_t>(start + n_samples_per_thread, n_samples);
        size_t n_samples_this_thread = end - start;
        size_t offset_in = start * n_in;
        size_t offset_out = start * n_out;

        const std::span<const double_t> inp_span(
            &input_ptr[offset_in], n_samples_this_thread * n_in
        );
        std::span<double_t> out_span(&output[offset_out], n_samples_this_thread * n_out);
        _run_interp(bin_logic_vec, inp_span, out_span, n_samples_this_thread);
    }
}

nb::ndarray<nb::numpy, double_t> run_interp_numpy(
    const nb::ndarray<int32_t> &bin_logic,
    const nb::ndarray<double_t> &input,
    int64_t n_threads
) {
    const int32_t *bin_logic_ptr = bin_logic.data();
    const double_t *input_ptr = input.data();
    if (bin_logic.size() < 4) {
        throw std::runtime_error("Invalid binary logic data");
    }

    int32_t n_in = bin_logic_ptr[2];
    int32_t n_out = bin_logic_ptr[3];
    size_t n_samples = input.size() / n_in;
    double_t *output_ptr = new double_t[n_samples * n_out];

    const std::vector<int32_t> bin_vec(bin_logic_ptr, bin_logic_ptr + bin_logic.size());
    const std::vector<double_t> inp_vec(input_ptr, input_ptr + input.size());
    std::span<double_t> out_span(output_ptr, n_samples * n_out);

    run_interp(bin_vec, inp_vec, out_span, n_threads);

    nb::capsule owner(output_ptr, [](void *p) noexcept { delete[] (double_t *)p; });
    return nb::ndarray<nb::numpy, double_t>(
        output_ptr, {n_samples, (size_t)n_out}, owner
    );
}

NB_MODULE(dais_bin, m) {
    m.def("run_interp", &run_interp_numpy, "bin_logic"_a, "data"_a, "n_threads"_a = 1);
}

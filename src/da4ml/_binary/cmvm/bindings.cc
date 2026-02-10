#include "bit_decompose.hh"
#include "mat_decompose.hh"
#include "api.hh"
#include "types.hh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

// Convert C++ CombLogicResult -> Python CombLogic NamedTuple
static nb::object make_py_comblogic(const CombLogicResult &sol) {
    auto types = nb::module_::import_("da4ml.cmvm.types");
    auto CombLogic_cls = types.attr("CombLogic");
    auto Op_cls = types.attr("Op");
    auto QInterval_cls = types.attr("QInterval");

    nb::list ops;
    for (auto &op : sol.ops) {
        auto qint = QInterval_cls(op.qint.min, op.qint.max, op.qint.step);
        ops.append(Op_cls(op.id0, op.id1, op.opcode, op.data, qint, op.latency, op.cost));
    }

    nb::list inp_shifts, out_idxs, out_shifts, out_negs;
    for (auto v : sol.inp_shifts)
        inp_shifts.append(v);
    for (auto v : sol.out_idxs)
        out_idxs.append(v);
    for (auto v : sol.out_shifts)
        out_shifts.append(v);
    for (auto v : sol.out_negs)
        out_negs.append(nb::bool_(v != 0));

    auto shape = nb::make_tuple(sol.shape.first, sol.shape.second);
    return CombLogic_cls(
        shape,
        inp_shifts,
        out_idxs,
        out_shifts,
        out_negs,
        ops,
        sol.carry_size,
        sol.adder_size
    );
}

// Convert C++ PipelineResult -> Python Pipeline NamedTuple
static nb::object make_py_pipeline(const PipelineResult &result) {
    auto types = nb::module_::import_("da4ml.cmvm.types");
    auto Pipeline_cls = types.attr("Pipeline");

    nb::list solutions;
    for (auto &sol : result.solutions) {
        solutions.append(make_py_comblogic(sol));
    }
    return Pipeline_cls(nb::tuple(solutions));
}

// Extract QIntervals from Python list of tuples/QInterval
static std::vector<QInterval> extract_qintervals(nb::object obj) {
    std::vector<QInterval> result;
    if (obj.is_none())
        return result;
    auto lst = nb::cast<nb::list>(obj);
    for (size_t i = 0; i < lst.size(); ++i) {
        auto item = lst[i];
        double mn = nb::cast<double>(item[nb::int_(0)]);
        double mx = nb::cast<double>(item[nb::int_(1)]);
        double st = nb::cast<double>(item[nb::int_(2)]);
        result.push_back(QInterval{mn, mx, st});
    }
    return result;
}

// Extract latencies from Python list
static std::vector<double> extract_latencies(nb::object obj) {
    std::vector<double> result;
    if (obj.is_none())
        return result;
    auto lst = nb::cast<nb::list>(obj);
    for (size_t i = 0; i < lst.size(); ++i) {
        result.push_back(nb::cast<double>(lst[i]));
    }
    return result;
}

// Python-facing solve function
static nb::object solve_numpy(
    const nb::ndarray<std::float32_t> &kernel_arr,
    const std::string &method0,
    const std::string &method1,
    int hard_dc,
    int decompose_dc,
    nb::object qintervals_obj,
    nb::object latencies_obj,
    int adder_size,
    int carry_size,
    bool search_all_decompose_dc
) {
    // Adapt numpy array to xtensor
    size_t ndim = kernel_arr.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i)
        shape[i] = kernel_arr.shape(i);
    auto kernel = xt::adapt(
        const_cast<std::float32_t *>(kernel_arr.data()),
        kernel_arr.size(),
        xt::no_ownership(),
        shape
    );

    auto qintervals = extract_qintervals(qintervals_obj);
    auto latencies = extract_latencies(latencies_obj);

    auto result = solve(
        xt::xarray<std::float32_t>(kernel),
        method0,
        method1,
        hard_dc,
        decompose_dc,
        qintervals,
        latencies,
        adder_size,
        carry_size,
        search_all_decompose_dc
    );

    return make_py_pipeline(result);
}

NB_MODULE(cmvm_bin, m) {
    m.def(
        "_volatile_int_arr_to_csd", &_volatile_int_arr_to_csd_numpy, "in"_a.noconvert()
    );
    m.def("get_lsb_loc", &get_lsb_loc, "x"_a);
    m.def("csd_decompose", &csd_decompose_numpy, "in"_a.noconvert(), "center"_a = true);
    m.def(
        "kernel_decompose", &kernel_decompose_numpy, "kernel"_a.noconvert(), "dc"_a = -2
    );
    m.def(
        "solve",
        &solve_numpy,
        "kernel"_a.noconvert(),
        "method0"_a = "wmc",
        "method1"_a = "auto",
        "hard_dc"_a = -1,
        "decompose_dc"_a = -2,
        "qintervals"_a = nb::none(),
        "latencies"_a = nb::none(),
        "adder_size"_a = -1,
        "carry_size"_a = -1,
        "search_all_decompose_dc"_a = true
    );
}

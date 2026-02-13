#include "bit_decompose.hh"
#include "mat_decompose.hh"
#include "api.hh"
#include "types.hh"
#include "state_opr.hh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

// Stub type so nanobind generates "da4ml.types.Pipeline" in the .pyi return type
struct PyPipeline {};
namespace nanobind::detail {
    template <> struct type_caster<PyPipeline> {
        NB_TYPE_CASTER(PyPipeline, const_name("da4ml.types.Pipeline"))
    };
} // namespace nanobind::detail

nb::typed<nb::tuple, float, float> cost_add_numpy(
    const nb::typed<nb::tuple, float, float, float> &q0_obj,
    const nb::typed<nb::tuple, float, float, float> &q1_obj,
    int64_t shift,
    bool sub,
    int adder_size,
    int carry_size
) {
    QInterval q0{
        nb::cast<double>(q0_obj[0]),
        nb::cast<double>(q0_obj[1]),
        nb::cast<double>(q0_obj[2])
    };
    QInterval q1{
        nb::cast<double>(q1_obj[0]),
        nb::cast<double>(q1_obj[1]),
        nb::cast<double>(q1_obj[2])
    };
    auto [lat, cost] = cost_add(q0, q1, shift, sub, adder_size, carry_size);
    return nb::make_tuple(lat, cost);
}

nb::ndarray<nb::numpy, int8_t> int_arr_to_csd_numpy(const nb::ndarray<int32_t> &in) {
    size_t ndim = in.ndim();
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = in.shape(i);
    }
    auto arr =
        xt::adapt(const_cast<int32_t *>(in.data()), in.size(), xt::no_ownership(), shape);
    xt::xarray<int32_t> arr_cpy(arr);
    auto *result = new xt::xarray(_volatile_int_arr_to_csd(arr_cpy));
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

    xt::xarray<float> arr_cpy(arr);
    auto [csd, shift0, shift1] = csd_decompose(arr_cpy, center);

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

// Convert C++ CombLogicResult -> Python CombLogic NamedTuple
static nb::object make_py_comblogic(const CombLogicResult &sol) {
    auto types = nb::module_::import_("da4ml.types");
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
    auto types = nb::module_::import_("da4ml.types");
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

using vec_of_qints = nb::typed<nb::sequence, nb::typed<nb::tuple, float, float, float>>;

// Python-facing solve function
static nb::typed<nb::object, PyPipeline> solve_numpy(
    const nb::ndarray<float> &kernel_arr,
    const std::string &method0,
    const std::string &method1,
    int hard_dc,
    int decompose_dc,
    nb::typed<nb::object, vec_of_qints> qintervals_obj,
    nb::typed<nb::object, std::vector<float>> latencies_obj,
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
        const_cast<float *>(kernel_arr.data()),
        kernel_arr.size(),
        xt::no_ownership(),
        shape
    );

    auto qintervals = extract_qintervals(qintervals_obj);
    auto latencies = extract_latencies(latencies_obj);

    auto result = solve(
        xt::xarray<float>(kernel),
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
    m.def("int_arr_to_csd", &int_arr_to_csd_numpy, "inp"_a.noconvert());
    m.def("get_lsb_loc", &get_lsb_loc, "x"_a);
    m.def("csd_decompose", &csd_decompose_numpy, "inp"_a.noconvert(), "center"_a = true);
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
    m.def(
        "cost_add",
        &cost_add_numpy,
        "q0"_a,
        "q1"_a,
        "shift"_a,
        "sub"_a,
        "adder_size"_a,
        "carry_size"_a,
        nb::sig(
            "def cost_add(q0: tuple[float, float, float], q1: tuple[float, float, "
            "float], shift: int, sub: bool, adder_size: int, carry_size: int) -> "
            "tuple[float, float]"
        )
    );
}

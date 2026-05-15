// lightgp Python bindings via pybind11.
//
// Build: see python/build_python.sh — produces python/lightgp/_core.*.so which
// the python/lightgp/__init__.py imports.

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/backend.h"
#include "core/kernel.h"
#include "core/mean.h"
#include "core/solver.h"
#include "core/tensor.h"
#include "inference/gp_exact.h"
#include "inference/gp_sparse.h"
#include "kernels/composite_kernel.h"
#include "kernels/kernel_base.h"
#include "kernels/linear_kernel.h"
#include "kernels/matern_kernel.h"
#include "kernels/periodic_kernel.h"
#include "kernels/rbf_kernel.h"

namespace py = pybind11;

// ===== Tensor ↔ numpy helpers =====

static lightgp::Tensor numpy_to_tensor(py::array arr) {
    // Always convert to float32 contiguous.
    py::array_t<float, py::array::c_style | py::array::forcecast> cast(arr);
    auto buf = cast.request();
    if (buf.ndim == 1) {
        lightgp::Tensor t(static_cast<std::size_t>(buf.shape[0]), 1);
        std::memcpy(t.data(), buf.ptr, static_cast<std::size_t>(buf.shape[0]) * sizeof(float));
        return t;
    } else if (buf.ndim == 2) {
        lightgp::Tensor t(static_cast<std::size_t>(buf.shape[0]),
                         static_cast<std::size_t>(buf.shape[1]));
        std::memcpy(t.data(), buf.ptr,
                    static_cast<std::size_t>(buf.shape[0] * buf.shape[1]) * sizeof(float));
        return t;
    }
    throw std::runtime_error("Expected 1D or 2D numpy array");
}

static py::array_t<float> tensor_to_numpy_1d(const lightgp::Tensor& t) {
    // Returns a 1D array when t has a single column (typical for mean/variance vectors).
    if (t.cols() == 1) {
        auto r = py::array_t<float>({static_cast<py::ssize_t>(t.rows())});
        std::memcpy(r.mutable_data(), t.data(), t.rows() * sizeof(float));
        return r;
    }
    auto r = py::array_t<float>({static_cast<py::ssize_t>(t.rows()),
                                 static_cast<py::ssize_t>(t.cols())});
    std::memcpy(r.mutable_data(), t.data(), t.size() * sizeof(float));
    return r;
}

// Adapter for predict() out-parameter API.
static py::dict gp_predict_dict_exact(lightgp::GPExact& gp, py::array X_test) {
    lightgp::Tensor X = numpy_to_tensor(X_test);
    lightgp::Tensor mean, var;
    if (!gp.predict(X, mean, var)) {
        throw std::runtime_error("predict failed (model not fitted?)");
    }
    py::dict d;
    d["mean"] = tensor_to_numpy_1d(mean);
    d["var"] = tensor_to_numpy_1d(var);
    return d;
}
static py::dict gp_predict_dict_sparse(lightgp::GPSparse& gp, py::array X_test) {
    lightgp::Tensor X = numpy_to_tensor(X_test);
    lightgp::Tensor mean, var;
    if (!gp.predict(X, mean, var)) {
        throw std::runtime_error("predict failed (model not fitted?)");
    }
    py::dict d;
    d["mean"] = tensor_to_numpy_1d(mean);
    d["var"] = tensor_to_numpy_1d(var);
    return d;
}

// ===== Module =====

PYBIND11_MODULE(_core, m) {
    m.doc() = "lightgp: Lightweight Gaussian Process inference (C++/Metal/CUDA)";

    // --- Enums ---
    py::enum_<lightgp::Backend>(m, "Backend")
        .value("CPU", lightgp::Backend::CPU)
        .value("Metal", lightgp::Backend::Metal)
        .value("CUDA", lightgp::Backend::CUDA)
        .value("Auto", lightgp::Backend::Auto)
        .export_values();

    py::enum_<lightgp::Solver>(m, "Solver")
        .value("Cholesky", lightgp::Solver::Cholesky)
        .value("CG", lightgp::Solver::CG)
        .value("SKI", lightgp::Solver::SKI)
        .export_values();

    // --- Kernels ---
    py::class_<lightgp::Kernel, std::shared_ptr<lightgp::Kernel>>(m, "Kernel")
        .def("name", &lightgp::Kernel::name)
        .def("num_params", &lightgp::Kernel::num_params)
        .def("get_params", &lightgp::Kernel::get_log_params)
        .def("set_params", &lightgp::Kernel::set_log_params)
        .def("__add__", [](std::shared_ptr<lightgp::Kernel> a,
                            std::shared_ptr<lightgp::Kernel> b) {
            return std::shared_ptr<lightgp::Kernel>(
                std::make_shared<lightgp::SumKernel>(a, b));
        }, py::is_operator())
        .def("__mul__", [](std::shared_ptr<lightgp::Kernel> a,
                            std::shared_ptr<lightgp::Kernel> b) {
            return std::shared_ptr<lightgp::Kernel>(
                std::make_shared<lightgp::ProductKernel>(a, b));
        }, py::is_operator())
        .def("__repr__", [](const lightgp::Kernel& k) {
            return "<lightgp.Kernel '" + k.name() +
                   "' params=" + std::to_string(k.num_params()) + ">";
        });

    py::class_<lightgp::RBFKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::RBFKernel>>(m, "RBF")
        .def(py::init<float, float>(),
             py::arg("length_scale") = 1.0f,
             py::arg("signal_var") = 1.0f);

    py::class_<lightgp::MaternKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::MaternKernel>>(m, "Matern")
        .def(py::init<float, float, float>(),
             py::arg("nu") = 2.5f,
             py::arg("length_scale") = 1.0f,
             py::arg("signal_var") = 1.0f);

    py::class_<lightgp::PeriodicKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::PeriodicKernel>>(m, "Periodic")
        .def(py::init<float, float, float>(),
             py::arg("length_scale") = 1.0f,
             py::arg("period") = 1.0f,
             py::arg("signal_var") = 1.0f);

    py::class_<lightgp::LinearKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::LinearKernel>>(m, "Linear")
        .def(py::init<float, std::size_t, float>(),
             py::arg("signal_var") = 1.0f,
             py::arg("input_dim") = 1,
             py::arg("offset") = 0.0f);

    py::class_<lightgp::ScaleKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::ScaleKernel>>(m, "Scale")
        .def(py::init<std::shared_ptr<lightgp::Kernel>, float>(),
             py::arg("base"),
             py::arg("scale") = 1.0f);

    py::class_<lightgp::SumKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::SumKernel>>(m, "SumKernel")
        .def(py::init<std::shared_ptr<lightgp::Kernel>,
                       std::shared_ptr<lightgp::Kernel>>(),
             py::arg("a"), py::arg("b"));

    py::class_<lightgp::ProductKernel, lightgp::Kernel,
               std::shared_ptr<lightgp::ProductKernel>>(m, "ProductKernel")
        .def(py::init<std::shared_ptr<lightgp::Kernel>,
                       std::shared_ptr<lightgp::Kernel>>(),
             py::arg("a"), py::arg("b"));

    // --- Mean functions ---
    py::class_<lightgp::MeanFunction,
               std::shared_ptr<lightgp::MeanFunction>>(m, "Mean")
        .def("name", &lightgp::MeanFunction::name)
        .def("num_params", &lightgp::MeanFunction::num_params);

    py::class_<lightgp::ZeroMean, lightgp::MeanFunction,
               std::shared_ptr<lightgp::ZeroMean>>(m, "ZeroMean")
        .def(py::init<>());

    py::class_<lightgp::ConstantMean, lightgp::MeanFunction,
               std::shared_ptr<lightgp::ConstantMean>>(m, "ConstantMean")
        .def(py::init<float>(), py::arg("c") = 0.0f);

    py::class_<lightgp::LinearMean, lightgp::MeanFunction,
               std::shared_ptr<lightgp::LinearMean>>(m, "LinearMean")
        .def(py::init<std::size_t>(), py::arg("input_dim"));

    // --- GPExact (new-API constructor with Kernel + Mean objects) ---
    py::class_<lightgp::GPExact>(m, "GPExact")
        .def(py::init([](std::shared_ptr<lightgp::Kernel> kernel,
                          std::shared_ptr<lightgp::MeanFunction> mean,
                          float noise_var,
                          lightgp::Backend backend,
                          lightgp::Solver solver) {
                 if (!mean) mean = std::make_shared<lightgp::ZeroMean>();
                 return new lightgp::GPExact(kernel, mean, noise_var, backend, solver);
             }),
             py::arg("kernel"),
             py::arg("mean") = std::shared_ptr<lightgp::MeanFunction>{},
             py::arg("noise_var") = 0.1f,
             py::arg("backend") = lightgp::Backend::Auto,
             py::arg("solver") = lightgp::Solver::Cholesky)
        .def("fit", [](lightgp::GPExact& gp, py::array X, py::array y) {
            return gp.fit(numpy_to_tensor(X), numpy_to_tensor(y));
        }, py::arg("X"), py::arg("y"))
        .def("predict", &gp_predict_dict_exact, py::arg("X_test"))
        .def("optimize",
             [](lightgp::GPExact& gp, int steps, float lr, bool verbose) {
                 return gp.optimize_hyperparameters(steps, lr, verbose);
             },
             py::arg("steps") = 100, py::arg("lr") = 0.05f, py::arg("verbose") = false)
        .def("log_marginal_likelihood", &lightgp::GPExact::log_marginal_likelihood)
        .def("fitted", &lightgp::GPExact::fitted);

    // --- GPSparse (legacy API — wraps GPSparseHyperparams; new Kernel-object
    //     constructor for GPSparse is on the next-moves list). ---
    py::class_<lightgp::GPSparseHyperparams>(m, "GPSparseHyperparams")
        .def(py::init<>())
        .def_readwrite("length_scale", &lightgp::GPSparseHyperparams::length_scale)
        .def_readwrite("signal_variance", &lightgp::GPSparseHyperparams::signal_variance)
        .def_readwrite("noise_variance", &lightgp::GPSparseHyperparams::noise_variance);

    py::class_<lightgp::GPSparse>(m, "GPSparse")
        .def(py::init([](float length_scale, float signal_var, float noise_var,
                          lightgp::Backend backend) {
                 lightgp::GPSparseHyperparams hp;
                 hp.length_scale = length_scale;
                 hp.signal_variance = signal_var;
                 hp.noise_variance = noise_var;
                 return new lightgp::GPSparse(hp, backend);
             }),
             py::arg("length_scale") = 1.0f,
             py::arg("signal_var") = 1.0f,
             py::arg("noise_var") = 0.1f,
             py::arg("backend") = lightgp::Backend::Auto)
        .def("fit", [](lightgp::GPSparse& gp, py::array X, py::array y,
                        std::size_t M) {
            return gp.fit(numpy_to_tensor(X), numpy_to_tensor(y), M);
        }, py::arg("X"), py::arg("y"), py::arg("num_inducing") = 100)
        .def("predict", &gp_predict_dict_sparse, py::arg("X_test"))
        .def("log_marginal_likelihood",
             &lightgp::GPSparse::log_marginal_likelihood)
        .def("fitted", &lightgp::GPSparse::fitted);

    m.attr("__version__") = "0.1.0";
}

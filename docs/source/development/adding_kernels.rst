Adding a Kernel
===============

A walkthrough of adding a new stationary kernel to LightGP. We'll use a
hypothetical "rational quadratic" kernel as the running example, but the
pattern is the same for any kernel that depends on the pairwise distance
:math:`r = \|x - x'\|`.

1. Pick the math
----------------

The rational-quadratic kernel:

.. math::

   k_{\mathrm{RQ}}(r) = \sigma^2 \left(1 + \frac{r^2}{2 \alpha \ell^2}\right)^{-\alpha}

Three log-hyperparameters: ``log_length_scale``, ``log_signal_var``, and
``log_alpha``.

2. Implement the C++ kernel
---------------------------

Create ``kernels/rational_quadratic_kernel.h`` and ``.cpp``:

.. code-block:: cpp

   // kernels/rational_quadratic_kernel.h
   #pragma once
   #include "kernel_base.h"

   namespace lightgp {

   class RationalQuadraticKernel : public Kernel {
   public:
       explicit RationalQuadraticKernel(float length_scale = 1.0f,
                                        float signal_var   = 1.0f,
                                        float alpha        = 1.0f);

       Tensor compute(const Tensor& X1, const Tensor& X2, Backend) const override;
       Tensor compute_diag(const Tensor& X) const override;
       int    num_params() const override { return 3; }
       std::vector<float> get_log_params() const override;
       void   set_log_params(const std::vector<float>& p) override;
       std::string name() const override { return "RationalQuadratic"; }

   private:
       float l_, sf2_, alpha_;
   };

   }  // namespace lightgp

The CPU implementation re-uses the BLAS distance trick from ``rbf_cpu.cpp``:

.. code-block:: cpp

   Tensor RationalQuadraticKernel::compute(const Tensor& X1, const Tensor& X2,
                                            Backend /*backend*/) const {
       const auto r2 = pairwise_squared_distance(X1, X2);  // shared helper
       Tensor K(r2.rows(), r2.cols());
       const float factor = 1.0f / (2.0f * alpha_ * l_ * l_);
       for (std::size_t i = 0; i < r2.size(); ++i) {
           K.data()[i] = sf2_ * std::pow(1.0f + factor * r2.data()[i], -alpha_);
       }
       return K;
   }

   Tensor RationalQuadraticKernel::compute_diag(const Tensor& X) const {
       // k(x, x) = sf2 for any stationary kernel.
       Tensor d(X.rows(), 1);
       std::fill(d.data(), d.data() + X.rows(), sf2_);
       return d;
   }

3. Wire it into the build
-------------------------

Add the source file to ``build.sh``:

.. code-block:: bash

   CPP_SRCS=(
       ...
       kernels/rbf_kernel.cpp
       kernels/rational_quadratic_kernel.cpp   # new
       ...
   )

CMake (if you use it) picks up new files automatically when listed in
``CMakeLists.txt``.

4. Test against the reference
-----------------------------

Add a C++ test in ``tests/test_rational_quadratic.cpp``:

.. code-block:: cpp

   void run_rational_quadratic_tests() {
       std::printf("[rational_quadratic] starting...\n");
       Tensor X = make_grid(8, -1.0f, 1.0f);
       RationalQuadraticKernel rq(/*l=*/1.0f, /*sf2=*/1.0f, /*alpha=*/0.5f);
       Tensor K = rq.compute(X, X);
       // Compare against hand-derived reference values at r=0 and r=1.
       LIGHTGP_CHECK_NEAR(K(0, 0), 1.0f, 1e-5f);  // k(x, x) = sf2
       // ... more checks ...
   }

Register it in ``tests/run_tests.cpp`` and ``./build/run_tests``.

5. Add the GPU kernel (optional)
--------------------------------

If you want a Metal / CUDA shader for the kernel matrix construction, mirror
the RBF pattern in ``kernels/metal/rbf_metal.mm`` /
``kernels/cuda/rbf_cuda.cu``. For most kernels you can re-use the
distance-then-scalar-function approach: compute :math:`r^2` via a tiled
shader, then apply the kernel's scalar function in a final pass.

Until you add the GPU path, the dispatcher falls back to CPU automatically.

6. Bind to Python
-----------------

In ``python/bindings.cpp``:

.. code-block:: cpp

   py::class_<lightgp::RationalQuadraticKernel, lightgp::Kernel,
              std::shared_ptr<lightgp::RationalQuadraticKernel>>(m, "RationalQuadratic")
       .def(py::init<float, float, float>(),
            py::arg("length_scale") = 1.0f,
            py::arg("signal_var") = 1.0f,
            py::arg("alpha") = 1.0f);

Re-export it from ``python/lightgp/__init__.py``:

.. code-block:: python

   from lightgp._core import (
       ...
       RationalQuadratic,
   )

7. Add a Python test
--------------------

In ``python/tests/test_kernels.py``:

.. code-block:: python

   def test_rational_quadratic_basic():
       k = gp.RationalQuadratic()
       assert k.num_params() == 3
       assert "RationalQuadratic" in k.name()

       # k(x, x) == sf2
       X = np.zeros((4, 1), dtype=np.float32)
       diag = gp.GPExact(k).fit  # smoke test fit
       # ... etc ...

8. Document it
--------------

Add a class entry to ``docs/source/api/kernels.rst`` matching the existing
style — math, parameter description, optional example. Re-build the docs:

.. code-block:: bash

   cd docs && sphinx-build -b html source build/html

That's the whole pipeline. The Kernel ABC plus the dispatch layer mean a new
kernel typically lands in 100–200 LOC plus tests, with no changes to the
inference layer.

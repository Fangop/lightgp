Using LightGP from C++
======================

LightGP is a dependency-free C++17 library. The Python bindings are a thin
pybind11 shim — every line of GP math lives in the C++ core. This means you
can embed LightGP in iOS apps, robotics stacks, game engines, or any other
C++ codebase, with no Python runtime requirement.

Building the C++ library
------------------------

.. code-block:: bash

   git clone https://github.com/Fangop/lightgp.git
   cd lightgp
   ./build.sh        # auto-detects Metal + Accelerate on macOS

After ``./build.sh``, every public symbol is available by including the
relevant header and linking the produced ``build/*.o`` object files (or use
the CMake target ``lightgp`` from the install tree).

Basic regression
----------------

.. code-block:: cpp

   #include "core/tensor.h"
   #include "kernels/rbf_kernel.h"
   #include "core/mean.h"
   #include "inference/gp_exact.h"

   using namespace lightgp;

   // Generate 100 points from sin(x).
   const int N = 100;
   Tensor X(N, 1);
   Tensor y(N, 1);
   for (int i = 0; i < N; ++i) {
       const float x = -3.0f + 6.0f * static_cast<float>(i) / (N - 1);
       X(i, 0) = x;
       y(i, 0) = std::sin(x);
   }

   auto kernel = std::make_shared<RBFKernel>(/*length_scale=*/1.0f);
   auto mean   = std::make_shared<ZeroMean>();
   GPExact gp(kernel, mean, /*noise_var=*/0.01f);

   gp.fit(X, y);
   gp.optimize_hyperparameters(/*steps=*/100);

   Tensor mean_pred, var_pred;
   gp.predict(X, mean_pred, var_pred);

Kernel composition
------------------

.. code-block:: cpp

   #include "kernels/composite_kernel.h"
   #include "kernels/periodic_kernel.h"

   auto kernel = gp::scale(std::make_shared<RBFKernel>())
               + gp::scale(std::make_shared<PeriodicKernel>(1.0f, 1.0f, 1.0f));

The free functions ``scale``, ``operator+``, and ``operator*`` are defined in
``kernels/composite_kernel.h`` and accept any ``std::shared_ptr<Kernel>``.

Backend selection
-----------------

.. code-block:: cpp

   #include "core/backend.h"

   // Force the Metal backend even when Auto would pick CPU:
   GPExact gp(kernel, mean, 0.01f, Backend::Metal);

   // Use CG (matrix-free on Metal) for very large N:
   GPExact gp(kernel, mean, 0.01f, Backend::Auto, Solver::CG);

   // SKI on macOS uses the Accelerate vDSP FFT path automatically:
   GPExact gp(kernel, mean, 0.01f, Backend::Auto, Solver::SKI);

iOS embedding (sketch)
----------------------

The C++ core has no runtime dependencies beyond Accelerate and Metal — both
are stock iOS frameworks. To embed:

1. Add the LightGP source tree to your Xcode project (or build it as a static
   library via the bundled ``CMakeLists.txt``).
2. Compile every ``*.cpp`` and ``*.mm`` under ``core/``, ``kernels/``,
   ``solvers/``, and ``inference/``.
3. Link against ``Accelerate.framework`` and ``Metal.framework``.
4. From Swift, expose your wrapper via an Objective-C++ bridging header.

CMake integration
-----------------

For projects that consume LightGP via ``find_package`` / ``add_subdirectory``:

.. code-block:: cmake

   add_subdirectory(${LIGHTGP_DIR})
   target_link_libraries(my_app PRIVATE lightgp)

The CMake target ``lightgp`` exposes the public include directories and links
``Accelerate`` / ``Metal`` automatically when present.

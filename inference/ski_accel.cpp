#include "ski_accel.h"

#ifdef LIGHTGP_HAS_ACCELERATE

#include <cassert>
#include <vector>

#include <Accelerate/Accelerate.h>

namespace lightgp {

// ---------------------------------------------------------------------------
// ToeplitzFFTCpu
// ---------------------------------------------------------------------------

struct ToeplitzFFTCpu::Impl {
    int M = 0;
    int N_fft = 0;   // even, ≥ 2*M, length valid for vDSP_DFT_zop_CreateSetup
    vDSP_DFT_Setup fwd_setup = nullptr;
    vDSP_DFT_Setup inv_setup = nullptr;
    std::vector<float> kernel_real;  // FFT of the embedded circulant column
    std::vector<float> kernel_imag;

    ~Impl() {
        if (fwd_setup) vDSP_DFT_DestroySetup(fwd_setup);
        if (inv_setup) vDSP_DFT_DestroySetup(inv_setup);
    }
};

namespace {

// vDSP_DFT_zop_CreateSetup accepts lengths of the form f*2^n with f ∈ {1, 3, 5, 15};
// a power of 2 is always valid and avoids worrying about that constraint.
int next_power_of_two(int x) {
    int p = 1;
    while (p < x) p *= 2;
    return p;
}

}  // namespace

ToeplitzFFTCpu::ToeplitzFFTCpu(const Tensor& toeplitz_col)
    : impl_(std::make_unique<Impl>()) {
    impl_->M = static_cast<int>(toeplitz_col.rows());
    assert(toeplitz_col.cols() == 1);
    assert(impl_->M >= 1);

    impl_->N_fft = std::max(2, next_power_of_two(2 * impl_->M));
    impl_->fwd_setup = vDSP_DFT_zop_CreateSetup(
        /*previous=*/nullptr, impl_->N_fft, vDSP_DFT_FORWARD);
    impl_->inv_setup = vDSP_DFT_zop_CreateSetup(
        /*previous=*/nullptr, impl_->N_fft, vDSP_DFT_INVERSE);
    assert(impl_->fwd_setup && "vDSP_DFT_zop_CreateSetup(FORWARD) failed");
    assert(impl_->inv_setup && "vDSP_DFT_zop_CreateSetup(INVERSE) failed");

    // Build the circulant first column for a symmetric Toeplitz embedding:
    // c[0]      = t[0]
    // c[i]      = t[i]            for i = 1..M-1
    // c[N - i]  = t[i]            for i = 1..M-1
    // c[other]  = 0
    std::vector<float> circ_real(impl_->N_fft, 0.0f);
    std::vector<float> circ_imag(impl_->N_fft, 0.0f);
    for (int i = 0; i < impl_->M; ++i) circ_real[i] = toeplitz_col(i, 0);
    for (int i = 1; i < impl_->M; ++i) circ_real[impl_->N_fft - i] = toeplitz_col(i, 0);

    impl_->kernel_real.assign(impl_->N_fft, 0.0f);
    impl_->kernel_imag.assign(impl_->N_fft, 0.0f);
    vDSP_DFT_Execute(impl_->fwd_setup,
                     circ_real.data(), circ_imag.data(),
                     impl_->kernel_real.data(), impl_->kernel_imag.data());
}

ToeplitzFFTCpu::~ToeplitzFFTCpu() = default;
ToeplitzFFTCpu::ToeplitzFFTCpu(ToeplitzFFTCpu&&) noexcept = default;
ToeplitzFFTCpu& ToeplitzFFTCpu::operator=(ToeplitzFFTCpu&&) noexcept = default;

int ToeplitzFFTCpu::M() const { return impl_->M; }

Tensor ToeplitzFFTCpu::matvec(const Tensor& v) const {
    assert(static_cast<int>(v.rows()) == impl_->M);
    assert(v.cols() == 1);

    const int M = impl_->M;
    const int N = impl_->N_fft;

    std::vector<float> in_real(N, 0.0f);
    std::vector<float> in_imag(N, 0.0f);
    for (int i = 0; i < M; ++i) in_real[i] = v(i, 0);

    std::vector<float> fv_real(N), fv_imag(N);
    vDSP_DFT_Execute(impl_->fwd_setup,
                     in_real.data(), in_imag.data(),
                     fv_real.data(), fv_imag.data());

    // Pointwise complex multiply: (kr + ki·i) · (fr + fi·i)
    //                            = (kr·fr − ki·fi) + (kr·fi + ki·fr)·i
    std::vector<float> prod_real(N), prod_imag(N);
    const float* kr = impl_->kernel_real.data();
    const float* ki = impl_->kernel_imag.data();
    for (int i = 0; i < N; ++i) {
        const float fr = fv_real[i];
        const float fi = fv_imag[i];
        prod_real[i] = kr[i] * fr - ki[i] * fi;
        prod_imag[i] = kr[i] * fi + ki[i] * fr;
    }

    std::vector<float> out_real(N), out_imag(N);
    vDSP_DFT_Execute(impl_->inv_setup,
                     prod_real.data(), prod_imag.data(),
                     out_real.data(), out_imag.data());

    // vDSP_DFT_zop inverse doesn't normalize. Scale by 1/N.
    const float scale = 1.0f / static_cast<float>(N);
    Tensor out(M, 1);
    for (int i = 0; i < M; ++i) out(i, 0) = out_real[i] * scale;
    return out;
}

// ---------------------------------------------------------------------------
// Multi-axis (Kronecker-Toeplitz) matvec
// ---------------------------------------------------------------------------

namespace {

void apply_toeplitz_along_axis_fft(const ToeplitzFFTCpu& plan,
                                   std::vector<float>& buf,
                                   const std::vector<int>& shape,
                                   int axis) {
    const int D = static_cast<int>(shape.size());
    const int M_d = shape[axis];
    assert(plan.M() == M_d);

    std::vector<int> strides(D, 1);
    for (int d = D - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];
    const int stride_axis = strides[axis];

    const int total = static_cast<int>(buf.size());
    const int num_fibers = total / M_d;

    // Reusable temp buffers for one fiber at a time.
    Tensor fiber(M_d, 1);
    for (int f = 0; f < num_fibers; ++f) {
        // Decode f into the D-1 non-axis coordinates and compute the base offset.
        int rem = f;
        int base = 0;
        for (int d = D - 1; d >= 0; --d) {
            if (d == axis) continue;
            const int idx_d = rem % shape[d];
            rem /= shape[d];
            base += idx_d * strides[d];
        }
        for (int j = 0; j < M_d; ++j) fiber(j, 0) = buf[base + j * stride_axis];
        Tensor out = plan.matvec(fiber);
        for (int j = 0; j < M_d; ++j) buf[base + j * stride_axis] = out(j, 0);
    }
}

}  // namespace

Tensor kron_toeplitz_matvec_accelerate(
    const std::vector<std::unique_ptr<ToeplitzFFTCpu>>& fft_plans,
    const std::vector<int>& grid_sizes,
    const Tensor& v) {
    const int D = static_cast<int>(grid_sizes.size());
    assert(static_cast<int>(fft_plans.size()) == D);
    int total = 1;
    for (int d = 0; d < D; ++d) total *= grid_sizes[d];
    assert(static_cast<int>(v.rows()) == total);

    std::vector<float> buf(total);
    for (int i = 0; i < total; ++i) buf[i] = v(i, 0);

    for (int d = 0; d < D; ++d) {
        apply_toeplitz_along_axis_fft(*fft_plans[d], buf, grid_sizes, d);
    }

    Tensor out(total, 1);
    for (int i = 0; i < total; ++i) out(i, 0) = buf[i];
    return out;
}

}  // namespace lightgp

#endif  // LIGHTGP_HAS_ACCELERATE

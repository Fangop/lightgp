"""Type stubs for lightgp._core (pybind11-generated module)."""
from typing import Dict, List
import numpy as np
import numpy.typing as npt


class Backend:
    CPU: "Backend"
    Metal: "Backend"
    CUDA: "Backend"
    Auto: "Backend"


class Solver:
    Cholesky: "Solver"
    CG: "Solver"


class Kernel:
    def name(self) -> str: ...
    def num_params(self) -> int: ...
    def get_params(self) -> List[float]: ...
    def set_params(self, params: List[float]) -> None: ...
    def __add__(self, other: "Kernel") -> "Kernel": ...
    def __mul__(self, other: "Kernel") -> "Kernel": ...


class RBF(Kernel):
    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0) -> None: ...


class Matern(Kernel):
    def __init__(self, nu: float = 2.5, length_scale: float = 1.0,
                 signal_var: float = 1.0) -> None: ...


class Periodic(Kernel):
    def __init__(self, length_scale: float = 1.0, period: float = 1.0,
                 signal_var: float = 1.0) -> None: ...


class Linear(Kernel):
    def __init__(self, signal_var: float = 1.0, input_dim: int = 1,
                 offset: float = 0.0) -> None: ...


class Scale(Kernel):
    def __init__(self, base: Kernel, scale: float = 1.0) -> None: ...


class SumKernel(Kernel):
    def __init__(self, a: Kernel, b: Kernel) -> None: ...


class ProductKernel(Kernel):
    def __init__(self, a: Kernel, b: Kernel) -> None: ...


class Mean:
    def name(self) -> str: ...
    def num_params(self) -> int: ...


class ZeroMean(Mean):
    def __init__(self) -> None: ...


class ConstantMean(Mean):
    def __init__(self, c: float = 0.0) -> None: ...


class LinearMean(Mean):
    def __init__(self, input_dim: int) -> None: ...


class GPExact:
    def __init__(self, kernel: Kernel, mean: Mean | None = None,
                 noise_var: float = 0.1, backend: Backend = ...,
                 solver: Solver = ...) -> None: ...
    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> bool: ...
    def predict(self, X_test: npt.NDArray[np.float32]) -> Dict[str, npt.NDArray[np.float32]]: ...
    def optimize(self, steps: int = 100, lr: float = 0.05, verbose: bool = False) -> bool: ...
    def log_marginal_likelihood(self) -> float: ...
    def fitted(self) -> bool: ...


class GPSparseHyperparams:
    length_scale: float
    signal_variance: float
    noise_variance: float
    def __init__(self) -> None: ...


class GPSparse:
    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0,
                 noise_var: float = 0.1, backend: Backend = ...) -> None: ...
    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.float32],
            num_inducing: int = 100) -> bool: ...
    def predict(self, X_test: npt.NDArray[np.float32]) -> Dict[str, npt.NDArray[np.float32]]: ...
    def log_marginal_likelihood(self) -> float: ...
    def fitted(self) -> bool: ...

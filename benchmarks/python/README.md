# Python benchmarks — GPyTorch comparison

Apples-to-apples timing of GPyTorch (PyTorch CPU + MPS) against the lightgp C++ benchmarks on the same hardware.

## Setup

```bash
pip install torch gpytorch
```

On Apple Silicon, PyTorch auto-detects MPS (`torch.backends.mps.is_available()` → True on macOS 12.3+). Some GPyTorch operations may fall back to CPU on MPS — those are documented per-bench in the script output.

## Scripts

- [`bench_gpytorch.py`](bench_gpytorch.py) — exact GP, CG GP, sparse GP across the same (N, D, M) grid the C++ benches use. Emits JSON to stdout matching the `bench_*` C++ format so plotting code can ingest both.

## Run

```bash
python3 benchmarks/python/bench_gpytorch.py > gpytorch_results.json
./build/bench_paper > lightgp_results.json     # lightgp equivalent (see bench_paper.cpp)
```

The two JSON files have matching schemas — join on `(method, N, D, M)` for comparison.

## Notes for fair comparison

- We compare against GPyTorch's stock ExactGP / SGPR — not the LazyTensor experimental path. That's what most GPyTorch users hit.
- PyTorch CPU is single-threaded (`torch.set_num_threads(1)`) by default in the script to match our single-thread CPU baseline; pass `--threads N` to bench multi-thread.
- Hyperparameter optimization: both libraries run the same number of L-BFGS or Adam steps for the fit phase.
- Predict timing excludes the warmup pass.
- We do not currently install or run these from CI — the actual numbers depend on the PyTorch / GPyTorch versions installed, so they live in the artifact directory rather than the code.

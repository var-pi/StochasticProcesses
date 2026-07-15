# StochasticProcesses

A computational companion to Pavliotis, *Stochastic Processes and Applications*, Ch. 1. Each
**unit** adds one self-contained vertical slice — a type or algorithm from the text, plus a
numerical experiment that checks it against a known closed-form result — building a gallery of
reproducible stochastic-process demos.

## The through-line

Everything in Chapter 1 is a statement about **one object**: the *covariance operator* of a
process, `(𝒞f)(t) = ∫ R(t,s) f(s) ds`, built from the covariance kernel `R(t,s) = 𝔼[XₜXₛ]`.
Every unit is one of two ways to take that operator apart:

- **Fourier / Bochner** — available only when the process is **stationary** (`R(t,s)` depends
  on the lag `t−s` alone). Then the covariance lives on the frequency axis as a *spectral
  density* `S(ω)`. *(Unit 1.)*
- **Mercer / Karhunen–Loève** — available on any **compact domain**, stationary or not. The
  operator has an eigenbasis, and the process expands in it. *(Unit 2.)*

Sampling a process is a third face of the same operator: it means applying a *square root* of
`𝒞` to white noise. Cholesky (Unit 0) is one such square root.

## Gallery

| Unit | Topic | Headline result |
|------|-------|-----------------|
| **0** — [`covariance core`](experiments/00_covariance_core/) | Gaussian process over a kernel `R(t,s)`; jittered-Cholesky sampling; empirical-covariance convergence | The empirical covariance of `N` sample paths converges to the true `Σ` at the Monte-Carlo rate — the fitted log–log slope collapses onto **−½**. |
| **1** — [`spectral / Bochner`](experiments/01_spectral_bochner/) | Stationary OU process; Welch & Bochner-FFT spectral estimators; circulant-embedding sampler | Two independent estimators of the OU spectrum both land on the analytic **Lorentzian**, and integrate back to the total variance `R(0)` under a pinned 2π convention. |
| **2** — [`KL quadrature`](experiments/02_kl_quadrature/) | Karhunen–Loève eigenbasis by symmetrized Nyström quadrature; BM closed-form check; torus contrast; KL-truncation sampler | The numerical eigenpairs match Brownian motion's closed form, `λ_k ∝ (2k−1)⁻²`, the trace identity pins assembly, and the torus coincidence `λ_k = R̂(k)` holds exactly on the circle but breaks on the interval. |

## Conventions

These are load-bearing for reproducibility — the *why* behind them; each experiment records its
own concrete values (seed, jitter, grid).

- **RNG.** Every stochastic routine seeds via `StableRNGs.StableRNG(seed)`, never the global RNG
  or bare `randn()` — the default stream is not stable across Julia versions, so a bare draw
  would silently change every committed number. The seed is always recorded.
- **Cholesky nugget.** Factorizations add a small nugget `Σ + εI` and report `ε`. Too small →
  an indefinite matrix throws; too large → a biased covariance. Default `ε = 1e-10`.
- **CI vs. local.** Deterministic analytic identities run in CI (`Pkg.test()`); Monte-Carlo
  experiments are run and verified **locally**, with their figures committed as artifacts.
- **Environment.** `Manifest.toml` is committed (not gitignored) so the pinned environment is
  reproducible — both the root package env and the separate `experiments/` env (which `dev`s the
  package; experiments run under `--project=experiments`).

## Commands

```sh
julia --project -e 'using Pkg; Pkg.test()'            # run the test suite (what CI runs)
julia --project -e 'using Pkg; Pkg.instantiate()'     # instantiate the pinned environment
julia --project=experiments experiments/00_covariance_core/run.jl # run an experiment (writes figures)
```

# StochasticProcesses

A computational companion to Pavliotis, *Stochastic Processes and Applications*, Ch. 1. Each unit
adds one self-contained vertical slice — a type/algorithm from the text plus a numerical experiment
that checks it against a known analytic result — building toward a gallery of reproducible
stochastic-process demos.

## Gallery

| Unit | Topic | Experiment |
|------|-------|------------|
| 0 | `GaussianProcess` over a covariance kernel `R(t,s)`; jittered-Cholesky sampling; empirical-covariance convergence | [`experiments/00_covariance_core/`](experiments/00_covariance_core/) — error-vs-N log–log plot, slope collapses onto −½ |

## Conventions

Every stochastic routine seeds via `StableRNGs.StableRNG(seed)` (never the global RNG, and the seed
is recorded); Cholesky factorizations always add a nugget `Σ + εI` and report `ε`; `Manifest.toml`
is committed for reproducibility; and Monte-Carlo experiments are run and verified locally, kept out
of CI (CI runs `Pkg.test()` only).

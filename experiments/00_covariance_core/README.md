# 00 · Covariance core: Monte-Carlo convergence of the empirical covariance

Run: `julia --project experiments/00_covariance_core/run.jl`

## Concept

For a Gaussian process with covariance kernel `R(s,t)`, the empirical covariance
`Σ̂_N` estimated from `N` i.i.d. sample paths converges to the true covariance
`Σ` at the standard Monte-Carlo rate: `‖Σ̂_N − Σ‖_F = O(N^{-1/2})`.

## Why this check

`assemble_cov`, `sample_cholesky`, and `empirical_cov` are the three primitives
the rest of this library is built on. This experiment exercises all three
together end-to-end and checks that the sampler's output is statistically
consistent with the kernel it was sampled from — not just that the code runs,
but that the numbers converge at the rate theory predicts.

## What it proves

1. Draw sample paths of Brownian motion via `sample_cholesky` from the
   assembled covariance `Σ = assemble_cov(gp, t_grid)`.
2. For `N` in a log-spaced ladder over `[10², 10⁴]`, form `Σ̂_N =
   empirical_cov(paths)` and measure `‖Σ̂_N − Σ‖_F`.
3. Fit the log–log slope of error vs. `N` with a self-contained OLS and gate
   it against the theoretical `−1/2` exponent: `|slope + 1/2| < 2.5·SE(slope)`.
4. Negative control: at `jitter = 0`, the same Brownian-motion `Σ` is exactly
   singular (its `t=0` row is identically zero, since `R(0,s)=0`), so
   `sample_cholesky` must throw `PosDefException`. This confirms the nugget
   (`jitter`) is load-bearing, not decorative, for the very `Σ` used above.

## Recorded configuration

- **Seed:** `StableRNG(20240501)`, never the global RNG.
- **Jitter:** `1e-10` for the main check; `0.0` for the negative control.
- **Grid:** `N_GRID = 64` points on `[0, 1]`.
- **N-ladder:** `{100, 316, 1000, 3162, 10000}`.
- **Kernel:** `brownian_motion_kernel` (`Σ = min(t,s)`), for both the main
  check and the negative control — *not* `exponential_kernel`, which is
  well-conditioned (`cond ≈ 5.9e4` at 200 points) and would not throw at
  `jitter = 0`.
- **Error metric:** Frobenius norm `‖Σ̂_N − Σ‖_F`.
- **Gate:** `|slope + 1/2| < 2.5 · SE(slope)`, from a self-contained OLS on
  `(log10 N, log10 error)` — no regression package dependency.

## Expected outcome

At this configuration, a dry run on Julia 1.10.11 prints:

```
fitted slope = -0.6211 +/- 0.0533 (SE);  target = -0.5
GATE: PASS  (|slope + 1/2| < 2.5*SE)
```

`|slope + 1/2| = 0.121 < 2.5·SE = 0.133` (margin ≈ 9%). The seed is fixed, so
this is deterministic: every rerun reproduces the exact `(N, error)` table and
fitted slope. A `FAIL`, or any drift from `slope = -0.6211`, indicates the
library code changed — not an unlucky draw. (`−1/2` is the asymptotic
exponent; the 5-point finite-`N` fit lands at `−0.62`, which is expected and
still inside the gate.)

## Tolerance regime

Monte-Carlo: the gate is a small multiple (2.5×) of the fitted slope's own
standard error, not a fixed absolute tolerance.

## Not run in CI

This experiment is Monte-Carlo and is not part of `test/runtests.jl` or CI.
Run it locally; the two figures below are committed artifacts.

## Figures

- `figures/sample_paths.png` — 5 Brownian motion sample paths from the
  Cholesky sampler.
- `figures/error_vs_N.png` — log–log Frobenius error vs. `N`, with the
  empirical fit and the `−1/2` reference line.

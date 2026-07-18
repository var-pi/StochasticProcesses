# Commit 1 — the `Ergodic` module (Unit 4 "ergodicity", feature `04-ergodicity`)

## TL;DR

Adds `src/ergodic.jl` (module `Ergodic`), the estimator spine of Unit 4: the pieces that
check the **L² law of large numbers** (Pavliotis Prop. 1.16) — that the time-average of one
long stationary path converges to the mean. Five public functions (three path-side, two
covariance-side) plus one shared private helper, all Base-only. Wired flat into
`src/StochasticProcesses.jl` (one `include`/`using`/`export` block matching the existing
`gof.jl` shape, one docstring bullet), with a new deterministic `@testset` and 5 names added
to the public-surface regression guard. The suite goes from 172 to 177 passing tests.

This is a **pure library commit** — no experiment yet (the `04_ergodicity` gallery is a later
commit in this unit). The change is strictly additive: `src/ergodic.jl` is new; the two
edited files only gain lines. **No deviations from the plan.**

---

## Background: the ergodic loop

If you already know the ergodic theorem for stationary processes you can skip to "What
changed". This section defines the objects the module estimates so a reader who knows basic
probability but not this specific machinery never has to tab away.

- **Stationary process, zero mean.** Every sampler in this repo draws a **zero-mean**
  Gaussian law, so the true mean is `μ = 0`, known exactly. The correlation depends only on
  the lag: `C(u) = E[X_s X_{s+u}]`, and `C(0)` is the variance at zero lag.

- **Time-average along one path.** `A_T = (1/T)∫₀ᵀ X_s ds`. The **L² law of large numbers**
  (Pavliotis Prop. 1.16): for a stationary process with integrable correlation, `A_T → μ`
  in mean square as `T → ∞`. Since `μ = 0` here, one long path's running average should decay
  toward 0 — the thing the experiment (later commit) will show.

- **The two constants that are easy to confuse.** These coincide only at `α = 1`, and the
  module docstring calls the difference out explicitly (Pavliotis Example 1.18):

  ```
  R(0) = C(0) = D/α       the variance at zero lag
  D*   = ∫₀^∞ C(u) du = D/α²   the transport / Green–Kubo coefficient
  ```

  for the OU correlation `C(u) = (D/α)·exp(-α|u|)`. The finite-T variance of the time-average
  decays like `2D*/T`, so the **rate constant carries α², not α** — swapping one for the other
  is a silent factor-of-α error that a test at `α = 1` could never catch.

- **The exact finite-T variance (Pavliotis Lemma 1.17).** Before the `T → ∞` limit, the
  variance of the time-average has a closed form:

  ```
  Var( (1/T)∫₀ᵀ X_s ds ) = (2/T²) ∫₀ᵀ (T − u) C(u) du.
  ```

  The `(T − u)` weight is the load-bearing detail: it vanishes at the upper endpoint `u = T`,
  so the top-lag correlation contributes nothing.

---

## What changed

Three files, all additive.

### 1. `src/ergodic.jl` — the new module (whole file, 143 lines)

A prominent preamble comment (lines 1–24) states the ergodic-loop purpose, the `n_grid × N`
one-path-per-**column** orientation, the zero-mean simplification, and the `R(0)` vs `D*`
distinction. Then `module Ergodic` with five exported functions and one private helper.

**Private helper — the shared discretization.**

```julia
function _cumulative_integral(path::AbstractVector, dt)
    n = length(path)
    I = zeros(float(eltype(path)), n)
    @inbounds for k in 2:n
        I[k] = I[k-1] + dt * (path[k-1] + path[k]) / 2
    end
    return I
end
```

`I_k = ∫₀^{t_k} X_s ds` by the trapezoid rule, `I[1] = 0`. Both `running_time_average` and
`mean_square_displacement` build on this one routine, so the three path estimators cannot
drift into subtly different discretizations of the same integral.

**Path-side (consume a sampled path matrix, `n_grid × N`, one path per column):**

- `running_time_average(path, dt)` — `A_k = I_k / t_k` for `k ≥ 2`, with `A[1] = path[1]`
  as the `T → 0` limit (the average over a vanishing window is the initial value). One path
  in, one running-average curve out.

- `time_average_variance(paths, dt)` — the ensemble variance of that running average at each
  `T`, across the columns. Computed as the raw mean-square `mean(A[k,:].^2)`, **not** a
  sample variance with an `N−1` correction: because the law is zero-mean, the mean-square
  about 0 already *is* the unbiased estimator of the variance about the true mean `μ = 0`
  that Lemma 1.17 predicts — there is no `N` vs `N−1` ambiguity to get wrong.

- `mean_square_displacement(paths, dt)` — the integrated MSD `E[(∫₀ᵗ X_s ds)²]`, which grows
  like `2D*·t`. Algebraically exactly `t_k² · time_average_variance` on the same matrix
  (since the running average is the integral divided by `t_k`), and pinned as an exact
  identity in the tests.

**Covariance-side (operate on a stationary covariance sequence `r = [C(0), C(dt), C(2dt), …]`):**

- `green_kubo(r, dt)` — the transport coefficient `D* = ∫₀^∞ C(u) du` by the trapezoid rule:

  ```julia
  return dt * (sum(r) - (r[1] + r[end]) / 2)
  ```

  For the OU correlation this converges to `D/α²`, deliberately not `R(0) = D/α`. A length-1
  `r` is a zero-width domain and returns `0`.

- `time_average_variance_exact(r, dt)` — the exact Lemma 1.17 curve at each `T = t_k`. The
  double integral is reduced from `O(n²)` to `O(n)` with two running cumulative sums
  `S1_k = Σ_{j≤k} r_j` and `S2_k = Σ_{j≤k} j·r_j`:

  ```
  V_k = (2/(k−1)²) · (k·S1_k − S2_k − (k−1)·r_1/2),   k ≥ 2,
  ```

  with `V[1] = r[1] = C(0)` the `T → 0` limit. On a uniform grid `dt` cancels analytically;
  it is kept in the signature for API symmetry with the other estimators. This `O(n)` form is
  what keeps the planned `n_grid ~ 2^14` calls of commit 2 tractable.

### 2. `src/StochasticProcesses.jl` — the wiring

Two edits, symmetric with every other submodule:

```julia
include("ergodic.jl")
using .Ergodic
export running_time_average, time_average_variance, mean_square_displacement,
       green_kubo, time_average_variance_exact
```

plus one line in the module docstring's submodule bullet list naming `Ergodic` as "the
ergodic loop: time-average estimators and the Green–Kubo coefficient", mirroring how the
other submodules are listed. The names are exported from *both* `Ergodic` and the umbrella
module, or they will not resolve for a `using StochasticProcesses` caller.

### 3. `test/runtests.jl` — the new testset + surface guard

A new `@testset "Ergodic — time-average estimators"` (inserted after the KS testset, before
the public-surface guard) with six nested testsets, plus the five new names appended to the
public-surface regression tuple. Detailed below.

---

## Why these design choices

- **Base-only, no `Statistics` dependency.** The obvious way to write the ensemble
  mean-square is `mean(...)` from `Statistics` — but `Statistics` is *not* in the package
  `[deps]` (only `FFTW, LinearAlgebra, Plots, StableRNGs` are; it appears only transitively
  via the Manifests), so `using Statistics` inside `ergodic.jl` would fail to load under
  `Pkg.test()` in the root env. Instead the mean-square is a Base incremental accumulator:

  ```julia
  acc = zeros(float(eltype(paths)), n)
  for j in 1:N
      acc .+= abs2.(running_time_average(view(paths, :, j), dt))
  end
  return acc ./ N
  ```

  This is the same idiom `empirical_cov` (`src/gaussianprocess.jl`) already uses elsewhere in
  the repo. It also sidesteps building an `n × N` matrix and an `O(N²)` `reduce(hcat, …)` over
  a generator (a generator misses Julia's fast concrete-array `hcat` reduction) — the whole
  pass stays `O(n·N)`, which matters for the large `n_grid ~ 2^14` calls planned for commit 2.

- **Mean-square about 0, not a sample variance.** Justified above: the zero-mean law makes
  the mean-square the correct unbiased estimator, removing the `N` vs `N−1` decision entirely.
  This is a deliberate exploitation of a repo-wide invariant, not a shortcut.

- **One shared `_cumulative_integral`.** Three path estimators over the same integral is three
  chances to discretize it differently. Factoring the trapezoid into one private helper makes
  the `MSD = t² · time_average_variance` identity hold *by construction*, and the test then
  confirms it.

- **The length-1 branch in `green_kubo` is kept though redundant.** For a length-1 `r` the
  general trapezoid formula already evaluates to exactly 0 (the halving is exact in binary
  floating point), so `length(r) == 1 && return zero(...)` is mathematically unnecessary. It
  is kept verbatim as specified, for documentation clarity about the zero-width-domain case —
  see "Deviations".

---

## Every test, and why it bites

All new assertions live in `@testset "Ergodic — time-average estimators"` (`dt = 0.5`). Per
the repo convention, tests prefer hand-computed targets over re-running the routine's own
formula, and non-square fixtures so orientation bugs cannot hide.

1. **Cumulative integral & running average on deterministic paths.** A constant path
   `X ≡ 2.0`: the running average is `2.0` at every `T`, including the `T → 0` limit `A[1]`. A
   linear ramp `X_k = t_k`: the trapezoid is **exact on a line**, so the hand targets
   `I_k = t_k²/2` and `A_k = t_k/2` are computed independently of the routine, not a re-run of
   its own formula.

2. **`green_kubo`: trapezoid hand value and the `D/α²` asymptote.** A 3-point hand value
   `green_kubo([3,5,1], 1) = 5 + (3+1)/2 = 7`; the length-1 zero-width case returns `0`; empty
   input throws `ArgumentError`. Then the OU asymptote (`D=1, α=2` on a ~40-correlation-time
   grid): `green_kubo` lands within `1e-3` relative of `D/α² = 0.25` **and** is explicitly
   checked *far* from `D/α = 0.5` via `!isapprox(...; rtol = 0.2)`. That negative assertion is
   the one that catches someone silently swapping in the wrong one of the two easily-confused
   constants — the failure mode the docstring warns about.

3. **`time_average_variance_exact`: Lemma 1.17 hand value & `T → 0` limit.** `r = [4,3,9]` at
   `dt = 1` gives `V[1] = C(0) = 4` and `V[3] ≈ 3.5`. The discriminating part: the `(T − u)`
   weight zeroes the endpoint, so `C(2) = 9` must **drop out entirely** for `V[3] = 3.5` to
   hold — a wrong cumulative-sum index (running through `k` instead of `k−1`, or vice versa)
   would leak the `9` back in and miss the target. A 2-point prefix collapses to `C(0)`, and
   empty input throws.

4. **`MSD = t² · time_average_variance` (exact algebraic identity).** A deliberately
   **non-square 5×4** fixture so a transpose bug cannot hide. Checks `length == size(P, 1) = 5`
   for both estimators, the exact relation `msd ≈ t².*tav` to `1e-12`, `msd[1] == 0` (since
   `I_1 = 0`), and `tav[1] ≈ sum(P[1,:].^2)/4` — the `E[X_0²]` value across the 4 columns.

5. **Empty-input guards.** All three path-side functions throw `ArgumentError` on degenerate
   input (empty vector / `0×0` matrix), rather than returning a misleading empty result.

6. **MC consistency: empirical variance tracks the exact curve.** A seeded
   (`StableRNG(4242)`) circulant-embedding OU ensemble, 3000 paths on a 2048-point grid.
   `time_average_variance` is checked against `time_average_variance_exact` at four interior
   `T` values (`k ∈ {200, 500, 1000, 1500}`, above the noise floor) with a loose `10%` rtol —
   robust to Monte-Carlo noise at a fixed seed (realized `≤ 0.021`), yet tight enough that a
   wiring bug like a transpose or wrong divisor lands far off. This is the one test that ties
   the empirical estimator to the analytic identity **inside CI**, rather than leaving that
   check to the manually-run experiment. It preallocates the path matrix (`Matrix{Float64}(undef, …)`)
   to avoid the `O(N²)` `reduce(hcat)` pattern.

7. **Public-surface regression guard.** The five new names are appended to the tuple whose
   loop asserts `isdefined(StochasticProcesses, f)`. This catches a name exported from
   `Ergodic` but not re-exported from the umbrella (or vice versa) — the two-half export pair
   falling out of sync — which would otherwise compile fine and only surface as a confusing
   `UndefVarError` downstream.

Full run: `julia --project -e 'using Pkg; Pkg.test()'` → `177 / 177` green in ~5.8s, Units
0–3's testsets unaffected (purely additive).

## Mutation-gate evidence

To confirm the new testset bites rather than merely passes, `green_kubo`'s endpoint-halving
term was dropped — `dt * sum(r)` in place of `dt * (sum(r) - (r[1] + r[end]) / 2)` — and the
full suite re-run. `Pkg.test()` **failed as expected**. The exact plan code was then restored
and the suite re-confirmed at `177 / 177` (bit-identical to the restored file via `diff`).

## Empirical / end-to-end verification

Beyond the suite, the module was driven through the actual package boundary (`using
StochasticProcesses`) with a fresh OU ensemble independent of the test fixtures (`D=1, α=2,
dt=0.02, n_grid=3000 ≈ 60 correlation times, N=5000 paths, StableRNG(20260718)`):

- `green_kubo(r, dt) = 0.250033…` vs. theory `D/α² = 0.25` — relative error `1.3e-4`.
- `time_average_variance[end] = 0.008453` (empirical, `T ≈ 120` correlation times) vs.
  `time_average_variance_exact[end] = 0.008268` (analytic Lemma 1.17) — a `2.2%` gap,
  consistent with Monte-Carlo noise at `N = 5000`.
- The `MSD = t² · time_average_variance` identity held to `1e-9` relative at three interior
  points.
- `running_time_average` on a single path started at `A[1] = path[1] = −1.2607` and decayed
  toward 0 (`A[end] = 0.1017`) as `T` grew — the L² law of large numbers visibly taking effect
  on a real path, not just a fixture.

An independent fresh-context subagent also re-derived the `O(n)` reduction of
`time_average_variance_exact` from the Lemma 1.17 double integral by hand (checking
specifically whether the cumulative sums run through index `k` or `k−1`) and re-verified the
`green_kubo` length-1 special case. Both confirmed algebraically exact; no bugs found.

## Trade-offs and known limitations

- **No first caller yet (built ahead of need, deliberately).** The estimators ship with no
  in-repo consumer — the `04_ergodicity` experiment that will drive them is a later commit in
  this unit. This is intentional: the module is deterministic and hand-testable now, so
  placing it in `src/` with a biting testset means the experiment inherits a verified
  primitive rather than re-deriving it. The MC-consistency test already exercises the empirical
  path through the real sampler inside CI.

- **Length-1 `green_kubo` branch is redundant but retained.** Kept verbatim as plan code for
  documentation clarity about the zero-width-domain case (see "Why these design choices"); the
  cost is one unreachable-in-effect line, judged worth the explicitness.

- **`dt` carried but unused in `time_average_variance_exact`.** On a uniform grid it cancels
  analytically; it stays in the signature only for API symmetry with the other estimators. A
  caller who passes a wrong `dt` there gets the right answer anyway — a mild surprise noted for
  the record.

## Code review

Ran `/code-review` at high effort (8 finder angles across correctness, cross-file, reuse,
simplification, efficiency, altitude, and CLAUDE.md conventions). **Zero findings survived.**
The module is purely additive with no callers yet to break, matches the repo's existing
patterns (`empirical_cov`'s Base-only accumulator idiom, the `gof.jl` re-export block shape),
and is RNG-free — the RNG appears only in the test's MC-consistency block, per the repo's
`StableRNG`-only convention.

## Deviations from plan

**None.** The plan's code for `src/ergodic.jl`, the `StochasticProcesses.jl` re-export block,
and the `test/runtests.jl` testset + public-surface additions were implemented verbatim (the
plan stated bodies and tolerances were final, grounded by a real run — confirmed independently
by the empirical pass and the subagent re-derivation above).

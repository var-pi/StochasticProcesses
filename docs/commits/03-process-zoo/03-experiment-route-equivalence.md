# Commit 3 — the route-equivalence experiment (Unit 3 "process zoo", feature `03-process-zoo`)

## TL;DR

Adds the first *experiment* of Unit 3: a new script
`experiments/03_process_zoo/run.jl` plus the four committed PNG figures it
generates under `experiments/03_process_zoo/figures/` (`portrait_bm.png`,
`portrait_bridge.png`, `portrait_ou.png`, `route_equivalence.png`). No `src/`
or `test/` file is touched — this is a pure gallery commit, so it changes no
library behaviour and adds nothing to CI (Monte-Carlo experiments are
deliberately kept out of CI per `CLAUDE.md`).

Unit 3's one distinct idea is **reconciliation**. Units 0–2 each showed that
*one* sampling route reproduces the true covariance. Unit 3 asks the harder
question: the library now offers **three independent square roots** of the same
covariance operator — Cholesky factorization (`sample_cholesky`),
Karhunen–Loève truncation (`nystrom_eigen` + `sample_kl`), and FFT / circulant
embedding (`sample_circulant_embedding`). For a process where all three apply
(Ornstein–Uhlenbeck on a uniform grid, which is stationary so circulant
embedding is legal), do they actually **agree with each other**, not merely each
land near the truth? The script's gate answers yes, by a **split-half bootstrap
null**: the three routes' empirical covariances must disagree by no more than
two disjoint halves of a *single* route's own samples disagree with each other
under sampling noise alone.

> ### Read this before trusting a future FAIL (false-fail callout)
>
> The gate is an **AND of three pairwise "in band" checks** against a **95 %
> (2.5 %–97.5 %) band**. Each pairwise check is, by construction, roughly a 95 %
> confidence check, so *each one false-fails about 5 % of the time* even when the
> routes are genuinely equivalent — and ANDing three (correlated but not
> perfectly correlated) such checks gives the whole gate a **meaningfully
> elevated false-fail rate under seed resampling**. This was measured
> empirically by the dispatching plan: **seed 271828 is one of 7 out of 8 tried
> seeds that PASS** — i.e. about **1 in 8** seeds fails despite all three routes
> being correct. **This is a property of a 95 %-band-AND-of-3-correlated-checks
> construction, NOT evidence that any route regressed.** If a future
> Julia/StableRNGs version perturbs the stream and seed 271828 starts printing
> FAIL, the *first* thing to check is whether it is simply an unlucky draw from
> this known ~1/8 rate (try a couple of nearby seeds; if the routes still land
> right at the band edge, it is noise, not a bug) — not to assume a sampler
> broke. See "Trade-offs and known limitations" for the full diagnosis,
> including the second, smaller bootstrap-quantile source of the same risk.

There were **no deviations from the plan**: `run.jl` was transcribed verbatim
from the dispatched commit plan and reproduced the predicted output on the first
run.

---

## Background: the terms you need

If you already read commits 1 and 2, you know this codebase's Gaussian-process,
kernel, Brownian-bridge and KS-statistic vocabulary; this section recaps only
what *this* commit leans on and goes deep on what is genuinely new here — the
three sampling routes, the Frobenius covariance-estimation-noise statistic, and
the split-half bootstrap null. Nothing below is left as an unexplained symbol.

- **Gaussian process (GP) and covariance kernel `R(t, s)`.** A random function
  `X(t)` whose values at any finite set of times are jointly Gaussian; here
  always zero-mean, so it is fully described by its covariance kernel
  `R(t, s) = Cov(X(t), X(s))`. "Which process" is just "which kernel". (Full
  treatment in commit 1's Background.)

- **The covariance matrix `Σ`.** Evaluate the kernel on a grid of `n` times to
  get the `n × n` matrix `Σ[i,j] = R(t_i, t_j)` — this is `assemble_cov(gp,
  grid)`, wrapped `Symmetric`. Everything downstream (all three samplers, the
  eigenproblem) is an operation on this one matrix / operator.

- **Ornstein–Uhlenbeck (OU) process.** The stationary Gaussian process modelling
  a variable relaxing toward equilibrium while buffeted by white noise
  (Pavliotis Example 1.15). Its kernel is the `exponential_kernel(t, s; D,
  alpha) = (D/alpha)·exp(-alpha·|t − s|)` in `src/kernels.jl`. Two constants:
  `D` is the noise strength and `alpha` the relaxation rate. Because `R` depends
  only on the lag `τ = t − s`, OU is **stationary**, and on a uniform grid its
  covariance matrix is **Toeplitz** (constant along diagonals) — the precondition
  that makes circulant embedding legal. The script uses `D = 1`, `alpha = 1`, so
  the zero-lag variance is `R(0) = D/alpha = 1`.

- **Correlation time.** The timescale over which the OU correlation decays by a
  factor `e`. For `R(τ) = (D/alpha)·exp(-alpha|τ|)` that timescale is
  `1/alpha`. At `alpha = 1` it equals `1`, so the OU domain `[0, T_OU] = [0, 5]`
  spans **about 5 correlation times**: over that window the correlation falls to
  `e^-5 ≈ 0.0067`, so the far corners of `Σ` are essentially decorrelated and
  the process is well-resolved on the grid. This is why `T_OU = 5` and not, say,
  `1` (too short — corners still strongly correlated, a poor stationarity test)
  or `50` (needlessly wide for a fixed 64-point grid — it would undersample each
  correlation time).

- **Nyström quadrature / `nystrom_eigen` (the KL route's engine).** The
  Karhunen–Loève expansion writes a zero-mean process in the eigenbasis of its
  covariance operator: `X(t) = Σ_k √(λ_k)·ξ_k·e_k(t)` with `ξ_k` iid `N(0,1)`,
  where `λ_k, e_k` solve the integral eigenproblem `∫ R(t,s) e(s) ds = λ e(t)`.
  `nystrom_eigen(R, nodes, weights)` (in `src/kl.jl`) discretizes that integral
  by a quadrature rule and solves it *symmetrized*: it forms `K_ij = R(t_i,t_j)`
  and `W = diag(weights)`, solves the **symmetric** problem
  `W^{1/2} K W^{1/2} g = λ g`, and recovers `e = W^{-1/2} g`. Solving the raw
  (non-symmetric) `KW` directly is the classic Nyström mistake — its
  eigenvectors are not `W`-orthonormal and can come out complex. It returns the
  eigenvalues sorted descending and an `n×nev` matrix of eigenfunctions sampled
  on the nodes, `W`-orthonormal (`Σ_i w_i e_k(t_i)² = 1`). The eigenvalues `λ_k`
  are simultaneously eigenvalues of the covariance operator **and** the variances
  of the expansion coefficients.

- **The three sampling routes (three square roots of `Σ`).** To draw
  `X ~ N(0, Σ)` you need any `A` with `A·Aᵀ = Σ`; then `X = A·z` for
  `z ~ N(0, I)` has covariance `Σ`. The library provides three different `A`s:
  1. **Cholesky** — `sample_cholesky(Σ, rng; jitter)`: factor `Σ + ε·I = L·Lᵀ`
     (`L` lower-triangular) and return `L·z`. General, works for any `Σ`, costs
     `O(n³)`. The **jitter** `ε = JITTER = 1e-10` (the reported nugget) is added
     before factoring so a singular or rounding-indefinite `Σ` does not throw.
  2. **Karhunen–Loève** — `sample_kl(lambdas, eigfuncs, rng)`: form
     `X = eigfuncs · (√(λ_k) .* z)`, i.e. the square root taken in the
     eigenbasis. With all modes kept it reproduces the (Nyström-discretized)
     covariance. A tiny negative `λ` (the discretization noise floor) is clamped
     to zero before `√` so it contributes no variance rather than throwing.
  3. **Circulant embedding** — `sample_circulant_embedding(r, rng)`: takes the
     square root with an **FFT** instead of a factorization, in `O(m log m)`.
     It embeds the stationary covariance sequence `r = [R(0), R(dt), …]` into a
     larger *circulant* matrix (Wood–Chan / Dietrich–Newsam), whose eigenvalues
     are `real(fft(c))` for the even extension `c` of `r`; `√` of those
     eigenvalues is the square root. **Only valid for a stationary process on a
     uniform grid** (it needs all embedding eigenvalues `≥ 0`), which is exactly
     why OU-on-a-uniform-grid is the one catalogue process where all three routes
     apply and can be reconciled.

- **`empirical_cov(paths)`.** The backward step: given `paths` (an **`n_grid × N`
  matrix, one sample path per COLUMN** — a load-bearing orientation from
  `CLAUDE.md`; a transpose silently estimates the wrong matrix), return the
  unbiased `n_grid × n_grid` sample covariance `Σ̂ = X_c·X_cᵀ / (N−1)`. As
  `N → ∞`, `Σ̂ → Σ`.

- **Frobenius norm `‖M‖_F`.** The entrywise root-sum-of-squares matrix norm,
  `‖M‖_F = √(Σ_ij M_ij²)`. In Julia this is `LinearAlgebra.norm(M)` for a matrix.
  The whole gate is built from `‖Σ̂_A − Σ̂_B‖_F`, the Frobenius distance between two
  empirical covariance estimates.

- **The Frobenius covariance-estimation-noise statistic (new here — the heart of
  the gate).** Two *independent* `M`-sample estimates `Σ̂_A`, `Σ̂_B` of the **same**
  true `Σ` are not equal — each entry has sampling scatter. For zero-mean Gaussian
  data the standard (Isserlis/Wick) result is
  `Var(Σ̂_ij) ≈ (Σ_ii·Σ_jj + Σ_ij²)/M` (leading term `Σ_ii·Σ_jj/M`; the exact
  second-order value adds `Σ_ij²`). So the *expected squared* Frobenius distance
  between two independent estimates is

  ```
  E‖Σ̂_A − Σ̂_B‖_F²  =  Σ_ij [Var(Σ̂_A,ij) + Var(Σ̂_B,ij)]        (independence)
                    =  2 · Σ_ij (Σ_ii·Σ_jj + Σ_ij²) / M
                    =  2 · ((trΣ)² + ‖Σ‖_F²) / M,
  ```

  using `Σ_ij Σ_ii·Σ_jj = (Σ_i Σ_ii)(Σ_j Σ_jj) = (trΣ)²` and
  `Σ_ij Σ_ij² = ‖Σ‖_F²`. Define `σ₁² = ((trΣ)² + ‖Σ‖_F²)/M`. Then the distance
  concentrates around `√2·σ₁ > 0` — crucially **not around zero**: two finite
  samples of the *same* law never give identical covariances, so a gate that
  asked for "≈ 0" would be meaningless. The right question is whether two
  *different routes* disagree by no more than this intrinsic `√2·σ₁` two-sample
  noise. That is precisely a split-half null.

- **Split-half bootstrap null.** A null distribution built not from a closed
  form but by **re-partitioning one route's own samples**: repeatedly split the
  `M` sample paths into two disjoint halves, compute `‖Σ̂_halfA − Σ̂_halfB‖_F`, and
  read off the central quantile band of that statistic. That band *is* the
  answer to "how much does this process's empirical covariance disagree with
  itself, at this sample size, under sampling noise alone?" A cross-route
  distance falling inside the band means the two routes are **statistically
  indistinguishable** — as close as two independent finite samples of the same
  route. It is a *bootstrap* null (re-using one dataset), so its quantile edges
  carry their own finite-resampling noise (discussed under limitations).

---

## What changed

One new script and its four committed figures. No library or test file changes.

### `experiments/03_process_zoo/run.jl` (new, 146 lines)

The script has two parts — qualitative **portraits** and the quantitative
**route-equivalence gate** — plus two small local helpers. It opens with the
mandated reproducibility setup: `ENV["GKSwstype"] = "100"` **before** `gr()`
(headless plotting, so figures render with no display in CI/agent shells), and
`using StableRNGs` with a single recorded seed. It never calls bare `randn()`
against the global RNG; every stochastic call receives the explicit
`StableRNG(SEED)` stream.

**The constants (every one, and why its value).**

| const | value | meaning / why |
|-------|-------|----------------|
| `T` | `1.0` | BM / bridge domain `[0,1]` (Pavliotis §1.5). |
| `T_OU` | `5.0` | OU domain `[0, 5]` ≈ **5 correlation times** at `alpha = 1` (see Background) — wide enough that the OU corners decorrelate and stationarity is genuinely exercised. |
| `N_GRID` | `64` | Uniform grid size; **all three routes share it** (a shared grid is what makes their covariances directly comparable). Large enough for a meaningful `64×64` covariance, small enough that the `O(n³)` Cholesky is instant. |
| `D` | `1.0` | OU noise strength. |
| `ALPHA` | `1.0` | OU relaxation rate; gives `R(0) = D/alpha = 1` and correlation time `1/alpha = 1`. |
| `N_ROUTE` | `4000` | Sample paths drawn **per route**. Sets the sampling noise floor `σ₁ ∝ 1/√N_ROUTE`. |
| `N_SPLIT` | `200` | Number of split-half re-partitions building the bootstrap null. Large enough that the band's quantile edges are stable (their own Monte-Carlo noise is subdominant to the seed-level effect). |
| `N_DEMO` | `6` | Demo paths drawn per portrait (visual only). |
| `JITTER` | `1e-10` | Cholesky nugget `ε`, **reported** in the final `printf` per `CLAUDE.md`. |
| `SEED` | `271828` | The `StableRNG` seed; recorded. Noted in a comment as passing with margin, 7/8 tested seeds pass. |

`ou(t, s) = exponential_kernel(t, s; D = D, alpha = ALPHA)` names the OU kernel
with the chosen parameters bound in.

**Helper 1 — `_quantile(sorted, p)` (type-7 quantile).** Computes the
linear-interpolation ("type 7") quantile of an **already-sorted** vector:
`h = (n−1)·p + 1`, `lo = floor(h)`, and interpolate
`sorted[lo] + (h − lo)·(sorted[lo+1] − sorted[lo])` (clamped at the top end). This
is exactly what `Statistics.quantile` returns by default — reimplemented in five
lines **specifically to avoid pulling `Statistics` into the `experiments`
environment** for one function. Keeping the experiments env minimal is a
deliberate reproducibility choice (fewer pinned dependencies in
`experiments/Manifest.toml`).

**Helper 2 — `splithalf_band(paths, nsplit, rng; lo=0.025, hi=0.975)` (the
bootstrap null; defined here, reusable by later phases).** The algorithm, in
full:

```julia
M = size(paths, 2); h = div(M, 2)          # M columns; half-size h
ds = Float64[]
for _ in 1:nsplit
    perm = sortperm(randn(rng, M))          # a uniform random permutation
    A = @view paths[:, perm[1:h]]           # first disjoint half
    B = @view paths[:, perm[h+1:2h]]        # second disjoint half
    push!(ds, norm(empirical_cov(A) .- empirical_cov(B)))
end
sort!(ds)
return _quantile(ds, lo), _quantile(ds, hi)  # central [2.5%, 97.5%] band
```

For each of `nsplit` iterations it draws a **uniform random permutation** of the
`M` columns, uses the first `h` as half A and the next `h` as half B (disjoint,
each of size `h = M÷2 = 2000`), and records the Frobenius distance between their
two empirical covariances. Sorting the `nsplit` distances and reading the 2.5 %
and 97.5 % type-7 quantiles gives the central 95 % band.

The **`sortperm(randn(rng, M))` random-permutation trick** deserves its own
sentence. It draws `M` iid standard normals from the *same `StableRNG` stream*
and returns the permutation that would sort them; because the `M` draws are iid
and continuous, all `M!` orderings are equally likely, so the result is a
**uniformly random permutation**. This is chosen over `Random.randperm` /
`Random.shuffle` for two reasons: (1) it needs no `import Random` — the whole
script's randomness stays on the single mandated `StableRNG`, with **no second
RNG interface** whose stability across Julia versions would be a separate risk;
and (2) it is a genuine unbiased shuffle, so the two halves are a real random
partition — the load-bearing property that makes the null a *sampling-noise*
null rather than an artifact of column order. (An unshuffled `[1:h]` / `[h+1:2h]`
split would compare the first-drawn 2000 paths against the last-drawn 2000; for
iid draws that is still statistically valid here, but the explicit shuffle makes
the "disjoint random halves" claim exact and robust to any ordering structure.)
The band is **process-specific**: it depends entirely on the `paths` passed in,
so a later phase wanting a different process's null just passes that process's
samples.

**Part 1 — Portraits (`portrait_bm.png`, `portrait_bridge.png`,
`portrait_ou.png`; purely visual, no gate).** The catalogue is three processes:
Brownian motion (`brownian_motion_kernel` on `[0,1]`), Brownian bridge
(`brownian_bridge_kernel` on `[0,1]`), and OU (`ou` on `[0, T_OU]`, whose nodes
come from `quad_nodes_weights(T_OU; n = N_GRID)` — the trapezoid rule, which is a
uniform grid on `[0, 5]`). For each, a 3-panel figure is saved: (p1) `N_DEMO = 6`
demo sample paths drawn via `sample_cholesky`; (p2) a heatmap of the covariance
matrix `Σ`; (p3) the eigenvalue decay on a `log10` axis (the first up-to-30
eigenvalues from `nystrom_eigen`, floored at `1e-16` so the log is finite).
Eigenvalue decay is computed by the local `eig_decay(kernel, grid)` helper, which
builds a quadrature on the process's *own* domain (`grid[end] − grid[1]`) and
calls `nystrom_eigen`. The demo RNG draws come **first and in fixed order**
(BM, then bridge, then OU), before any route-equivalence draw — reordering them
would change every downstream number and figure (the `CLAUDE.md` "do not reorder
RNG draws" rule).

**Part 2 — Route equivalence (`route_equivalence.png`; the GATE).** For OU on
the uniform grid:

1. Assemble the true `Σ_ou = assemble_cov(GaussianProcess(ou), ou_nodes)`,
   compute the KL eigenpairs `ou_lambdas, ou_eigfuncs = nystrom_eigen(ou,
   ou_nodes, ou_w)`, and the stationary covariance sequence
   `r_seq = [ou(0, k·dt) for k in 0:N_GRID-1]` (with `dt` the uniform spacing).
2. Draw `N_ROUTE = 4000` paths by **each** route, in fixed order — Cholesky, then
   KL, then circulant — via `reduce(hcat, …)`, giving three `n_grid × N_ROUTE`
   path matrices with the correct **one-path-per-column** orientation.
3. Form each route's empirical covariance: `Σ_chol`, `Σ_kl`, `Σ_circ` from
   `empirical_cov`.
4. Build the split-half null band `(band_lo, band_hi) =
   splithalf_band(chol_paths, N_SPLIT, rng)` from **Cholesky's own** samples.
5. Compute the closed-form sanity scale
   `sigma1 = sqrt((tr(Σ_ou)² + norm(Σ_ou)²)/N_ROUTE)` and `theory = 2·sigma1`.
6. For each route pair `(Chol–KL, Chol–Circ, KL–Circ)`, compute the
   **√2-rescaled** cross distance `sqrt(2)·norm(A .- B)`.
7. **Gate:** `all_in = all(band_lo ≤ d ≤ band_hi for d in rescaled)` — every one
   of the three pairwise distances must be inside the band. A separate
   self-consistency check `band_ok = band_lo ≤ theory ≤ band_hi` confirms the
   analytic `2σ₁` also lands in the band.
8. Print the band, the three rescaled distances with in-band/OUT verdicts, the
   `band brackets theory` line, and `PASS`/`FAIL`. Save `route_equivalence.png`
   (the null band as a thick horizontal bar, the three route pairs as scatter
   points, and `2σ₁` as a dashed vertical line) and the recorded-constants line.

### The four figures (committed artifacts)

`experiments/03_process_zoo/figures/{portrait_bm,portrait_bridge,portrait_ou,route_equivalence}.png`,
regenerated by `run.jl` and committed per the `CLAUDE.md` "commit their figures"
rule. Their content is described in "Every figure/statistic, individually".

---

## Why these design choices

### The √2 rescaling, algebra spelled out in full

This is the subtle heart of the gate, and it is easy to get wrong by a factor.
The comparison is between two things measured at **different sample sizes**:

- the **cross-route** distance compares two **full-sized** estimates — Cholesky's
  4000 paths vs KL's 4000 paths (or any pair), so each side is an `M = 4000`
  estimate;
- the **split-half null** compares two **half-sized** estimates — 2000 of
  Cholesky's paths vs the other 2000, so each side is an `M = 2000` estimate.

Recall the derived fact `E‖Σ̂_A − Σ̂_B‖_F² = 2·((trΣ)² + ‖Σ‖_F²)/M` for two
independent `M`-sample estimates, and `σ₁² = ((trΣ)² + ‖Σ‖_F²)/N_ROUTE` (the
script defines `σ₁` at the **full** size `N_ROUTE = 4000`). Then:

```
Cross route (M = 4000 each side):
    E‖Σ̂_A − Σ̂_B‖_F²  =  2·σ₁²            ⇒   RMS distance  =  √2·σ₁     (the "unrescaled cross RMS")
    rescale by √2:      √2 · (√2·σ₁)      =  2·σ₁

Split-half null (M = 2000 each side):
    σ₁²(2000)  =  ((trΣ)² + ‖Σ‖_F²)/2000  =  2·σ₁²(4000)
    E‖Σ̂_A − Σ̂_B‖_F²  =  2·σ₁²(2000)  =  4·σ₁²(4000)   ⇒   RMS distance  =  2·σ₁
```

So **both** the √2-rescaled cross statistic **and** the split-half null
concentrate at exactly `theory = 2·σ₁`. That is why the script sets
`theory = 2*sigma1`, and its comment (lines ~118–122) warns explicitly that the
band brackets `2σ₁`, **not** `√2·σ₁` — the latter is the *unrescaled* cross RMS,
a common off-by-a-factor mistake. Intuitively: halving the sample size from 4000
to 2000 **doubles** the variance of each covariance entry, hence multiplies the
Frobenius distance's scale by `√2`; multiplying the 4000-scale cross distance by
`√2` therefore re-expresses it on the 2000-scale of the split-half null, so the
two are directly comparable. `band_ok` (theory in band) is an **independent
analytic cross-check** that the whole calibration is self-consistent: it does not
depend on the three routes agreeing — it only asks whether the closed-form `2σ₁`
that *both* constructions should target actually sits inside the empirically
built band. If `band_ok` failed while the routes still landed in band, that would
flag a calibration/derivation error (wrong power of the sample size, say), not a
sampler bug.

### Why Cholesky's samples generate the null (not KL's or circulant's)

The script builds the null from `chol_paths`. The script itself does **not**
state a reason, so the following is **my own reasonable inference, flagged as
such**: Cholesky is the oldest, most general, and most thoroughly understood of
the three square roots (it is Unit 0's original sampler, and `sample_cholesky`
works for *any* covariance matrix, not just stationary ones), which makes it the
natural **reference** route to bootstrap the self-consistency band against. Using
the most-trusted route to define "how much a correct sampler disagrees with
itself" and then testing the *other two* against that band is the conservative
choice: it asks KL and circulant to match the yardstick route's intrinsic noise,
rather than defining the yardstick from a route whose correctness is itself part
of what Unit 3 is establishing. (Because the routes are genuinely equivalent, the
band would be statistically the same whichever route generated it; the choice is
about which route is the natural "known-good reference", not about the band's
value.)

### Why an empirical band rather than a fixed tolerance

`CLAUDE.md`'s experiment convention forbids gating a stochastic quantity against a
fixed absolute tolerance — the right gate is against theory within a small
multiple of the statistic's own noise. The split-half band **is** that noise,
measured directly from the data, so the gate self-calibrates to whatever `σ₁`
the chosen grid, kernel and sample size produce. The `theory = 2σ₁` line is the
analytic anchor that confirms the empirical band is centred where it should be.

---

## Every figure/statistic, individually (and why it bites)

Treating each figure and each printed statistic the way commits 1 and 2 treat
each `@test`: what specific bug would move it, and how the artifact catches it.
Note that unlike the deterministic `test/` tier, these are Monte-Carlo artifacts
— they "bite" by moving visibly/measurably under a real defect, calibrated to
the sampling noise, not by an exact equality.

1. **`portrait_bridge.png` — Brownian bridge portrait.** The paths panel must
   show all 6 demo paths **pinned near 0 at both endpoints** `t = 0` and `t = 1`,
   fanning out in the middle; the covariance heatmap must be **diamond-shaped**,
   peaking (`≈ 0.25`) on the diagonal at the centre `t = 0.5` and vanishing at all
   four corners. A route/kernel swap that accidentally used `min(t,s)` (Brownian
   motion) instead of the bridge would show paths pinned only at `t = 0` and free
   at `t = 1`, and a ramp-shaped heatmap — immediately visible. This is the visual
   companion to commit 1's `R(1,s) = 0` endpoint test.

2. **`portrait_bm.png` — Brownian motion portrait.** The heatmap must show the
   `min(t,s)` **covariance ramp** (rising toward the `t = s = 1` corner, since
   `Var = R(t,t) = t`), and the paths must start pinned at `0` and diffuse
   outward with growing spread. A sign error or a bridge/BM mix-up would break the
   monotone-ramp shape.

3. **`portrait_ou.png` — OU portrait.** The heatmap must show a **banded Toeplitz**
   structure — bright on the diagonal (`R(0) = 1`) and decaying with distance from
   it (correlation `exp(-|t−s|)`), constant along each diagonal — the stationarity
   signature. The eigenvalue-decay panel must show a smooth decreasing sequence on
   the log axis. A non-stationary bug (e.g. an accidental `min`-type kernel) would
   destroy the constant-diagonal banding.

4. **The eigenvalue-decay panels (all three portraits).** Computed via
   `nystrom_eigen` on each process's own grid, plotted `log10`. These bite against
   a **broken symmetrized-Nyström** implementation: if `nystrom_eigen` regressed to
   solving the raw non-symmetric `KW`, eigenvalues could come out complex or
   negative and the log-scale plot would fail or show garbage. A monotone positive
   decay is the visual health check on the KL route's engine.

5. **Printed statistic — the split-half null band `[band_lo, band_hi]`**
   (`[1.6120, 2.8825]` this run). This is the empirically measured self-noise of
   the OU covariance at `N_SPLIT = 200` re-splits. If the `splithalf_band` shuffle
   were broken — e.g. an **unshuffled `[1:h]`/`[h+1:2h]` split** combined with any
   column ordering structure, or halves that overlapped instead of being disjoint
   — the band would be mis-sized (too narrow if halves shared paths, since a path
   in both halves inflates agreement), and the cross-route points would then read
   OUT for the wrong reason. The band's *width* is the thing the gate's
   sensitivity rides on.

6. **Printed statistic — the three √2-rescaled cross distances**
   (`Chol–KL 2.2411`, `Chol–Circ 2.5811`, `KL–Circ 1.8591` this run), each with an
   `in band`/`OUT` verdict. These are the actual reconciliation measurements. A
   **route-swapping / route-regression bug** — say `sample_kl` clamped too many
   modes, or `sample_circulant_embedding` returned the wrong FFT branch, so that
   route's empirical covariance systematically differed from the others — would
   push that route's *two* pairwise distances (the two pairs it participates in)
   above `band_hi`, printing OUT and failing the gate. Note the structure: each
   route appears in exactly two of the three pairs, so a single bad route trips
   two of the three checks.

7. **Printed statistic — the `√2` rescaling itself.** If the rescaling factor
   were dropped (comparing a 4000-scale cross distance directly against a
   2000-scale band), every cross distance would be about `1/√2 ≈ 0.71×` too small
   and would sit **below** `band_lo` — the gate would print OUT for all three
   even though the routes agree. Equivalently, a **variance-scale-halving bug**
   (e.g. one route drawing effectively half as many independent samples) would
   inflate that route's distances by `≈ √2` and push it above the band. The `√2`
   factor and `N_ROUTE`/`N_SPLIT` sizes are jointly load-bearing: the artifact only
   bites correctly because both scales are matched.

8. **Printed statistic — `band brackets theory 2σ₁` (`yes` this run) and the
   `theory` line `2σ₁ = 2.1966`.** This is the independent analytic cross-check.
   If it printed `no`, the empirical band and the closed-form `σ₁` disagree — a
   sign the derivation, the `sigma1` formula, or the sample-size bookkeeping is
   off, even if the three routes happen to agree with each other. It bites a
   *calibration* bug that the route-agreement checks alone could miss.

9. **`route_equivalence.png` — the gate figure.** Shows the three scatter points
   landing **inside** the horizontal null band, straddling the dashed `2σ₁` theory
   line. A reader can see at a glance whether reconciliation held and by how much
   margin. If a point sat clearly outside the bar, the gate FAILed.

---

## Empirical / runtime verification

Running `julia --project=experiments experiments/03_process_zoo/run.jl` against
the committed code produced (verbatim):

```
route equivalence (split-half null band [1.6120, 2.8825], null-scale theory 2σ₁=2.1966):
  Chol–KL    √2·‖Σ̂_A−Σ̂_B‖_F = 2.2411  -> in band
  Chol–Circ  √2·‖Σ̂_A−Σ̂_B‖_F = 2.5811  -> in band
  KL–Circ    √2·‖Σ̂_A−Σ̂_B‖_F = 1.8591  -> in band
  band brackets theory 2σ₁: yes
route equivalence -> PASS

recorded: T_OU=5.0 N_GRID=64 D=1.0 alpha=1.0 N_ROUTE=4000 N_SPLIT=200 jitter=1e-10 seed=271828
```

All three route pairs land inside the split-half band, the analytic `2σ₁ =
2.1966` is bracketed by the band, and the gate prints **PASS**. The recorded line
captures every reproducibility constant (domain, grid, OU parameters, sample
sizes, the `jitter = 1e-10` nugget, and `seed = 271828`).

All four figures were generated and **visually inspected**: the bridge portrait
shows paths correctly pinned near 0 at both endpoints with the diamond-shaped
covariance heatmap peaking at the diagonal centre; the Brownian-motion portrait
shows the expected `min(t,s)` covariance ramp; the OU portrait shows the banded
stationary heatmap and smooth eigenvalue decay; and `route_equivalence.png` shows
all three scatter points inside the horizontal null band, straddling the dashed
theory line.

The full deterministic test suite (`julia --project -e 'using Pkg;
Pkg.test()'`) was re-run after adding this file and still passes **146/146**.
Because this commit touches **no** `src/` or `test/` file, that is expected — it
is *not* a claim that this experiment is covered by CI. Per `CLAUDE.md`,
Monte-Carlo experiments are deliberately **not** run in CI; this script is run
locally and its figures committed.

---

## Trade-offs and known limitations

- **Elevated false-fail rate under seed resampling (the primary caveat — also in
  the TL;DR callout).** The gate ANDs three pairwise checks, each against a 95 %
  (2.5 %–97.5 %) band. For genuinely-equivalent routes each pairwise check is
  ~a 95 % CI check, so each false-fails ~5 % of the time *by construction*. The
  three checks are **correlated but not perfectly** — each route appears in two
  pairs, so they share sampling fluctuations but are not identical — so the
  overall AND-gate's false-fail probability is meaningfully above 5 % but below
  the ~15 % of three independent checks. The dispatching plan **measured** this:
  **seed 271828 is one of 7 of 8 tried seeds that PASS**, i.e. about **1 in 8**
  seeds false-fails despite all routes being correct. This is inherent to a
  95 %-band-AND-of-3-correlated-comparisons design, **not** a defect. Operational
  consequence: if a future Julia/StableRNGs version shifts the RNG stream and seed
  271828 begins printing FAIL, **first** rule out an unlucky draw — try a couple
  of nearby seeds and check whether the routes still land at/near the band edge
  (noise) rather than far outside (a real regression) — before suspecting any
  sampler. Widening the band (e.g. to 99 %) or reducing the AND to a single
  aggregate statistic would lower the false-fail rate at the cost of sensitivity;
  the 95 %/AND design was kept for its sharper bite, with this caveat documented.

- **The null is a BOOTSTRAP, with its own second-order noise.** The band is built
  by **re-splitting one route's (Cholesky's) own samples** `N_SPLIT = 200` times,
  not from an independent closed-form null distribution. So the band's quantile
  edges (`band_lo`, `band_hi`) themselves carry finite-`N_SPLIT` Monte-Carlo
  noise — a **second, smaller** source of the same false-fail risk. A coarser
  quantile estimate from fewer re-splits would widen this effect; `N_SPLIT = 200`
  was chosen large enough that it is **subdominant** to the primary ~1/8
  seed-level effect above. The analytic `band_ok` cross-check partly guards
  against a badly mis-estimated band (if the bootstrap band drifted far from
  `2σ₁`, `band_ok` would print `no`).

- **Route equivalence is only testable where all three routes apply.** Circulant
  embedding needs a **stationary** process on a **uniform grid**, so only OU (not
  the Brownian motion or bridge, both non-stationary) can be reconciled across all
  three routes. The portraits still show all three processes qualitatively, but the
  quantitative gate is OU-only by necessity, not omission.

- **Redundant `nystrom_eigen` for OU (accepted, unfixed).** The OU eigenproblem is
  solved twice — once inside the portrait loop's `eig_decay` helper (for the decay
  plot) and once explicitly for the route-equivalence section (for `sample_kl`).
  Left as-is deliberately: at `N_GRID = 64` a `64×64` symmetric eigensolve is
  negligible, and de-duplicating it would mean editing the locked verbatim script
  for a purely cosmetic gain (see Code review).

---

## Code review

This `run.jl` was specified **FINAL/verbatim** by the dispatching commit plan and
had already been verified by the planner against the committed package before
dispatch, so it was transcribed unchanged. A self-review pass was therefore run
**treating the script as fixed content** — the review's job was to confirm
conventions and wiring, not to hunt for an algorithm to rewrite. It checked, and
confirmed clean:

- **RNG discipline (`CLAUDE.md`)** — a single explicit `StableRNG(SEED)` stream
  threads every stochastic call (`sample_cholesky`, `sample_kl`,
  `sample_circulant_embedding`, and the `randn(rng, M)` inside `splithalf_band`);
  **no bare `randn()` / global RNG** anywhere. The seed is recorded in the consts
  and echoed in the final `printf`.
- **Cholesky nugget** — `JITTER = 1e-10` is passed to every `sample_cholesky` call
  and **reported** in the recorded-constants line.
- **Headless plotting** — `ENV["GKSwstype"] = "100"` is set **before** `gr()`.
- **Path orientation** — every path matrix is built `n_grid × N` (one path per
  column) via `reduce(hcat, …)` and fed to `empirical_cov` in that orientation, so
  the estimated covariance is `n_grid × n_grid` (not `N × N`). No transpose.
- **RNG-draw order** — demo draws (BM, bridge, OU) precede the route draws (Chol,
  KL, circ) which precede the split-half permutations; the order is not reordered.

**No defect was found.** The one noted item is a **mild inefficiency**, not a
bug: `nystrom_eigen` is computed twice for OU (portrait `eig_decay` + the explicit
route-equivalence call). It was **consciously left as-is** — cheap at
`N_GRID = 64`, and factoring it out would mean touching the locked verbatim script
for a cosmetic efficiency gain, out of scope for this commit.

---

## Deviations from plan

**None.** `run.jl` was transcribed verbatim from the dispatched plan and
reproduced the exact predicted output on the first run (the run output above
matches the plan's prediction). The only new filesystem objects created were the
`experiments/03_process_zoo/` directory and its `figures/` subdirectory — both
implied/required by the plan's file list, not a scope change.

---

## Pass conditions verified

1. **The route-equivalence gate PASSes.** All three √2-rescaled cross distances
   (Chol–KL 2.2411, Chol–Circ 2.5811, KL–Circ 1.8591) fall inside the split-half
   null band `[1.6120, 2.8825]`; the script prints `route equivalence -> PASS`.

2. **The analytic self-consistency check holds.** `band brackets theory 2σ₁: yes`
   — the closed-form `2σ₁ = 2.1966` lies inside the empirically built band,
   confirming the calibration (the `√2` rescaling and `σ₁` formula) is
   self-consistent, independent of the routes agreeing.

3. **All four figures are generated and committed, and visually correct.**
   `portrait_bm.png`, `portrait_bridge.png`, `portrait_ou.png`, and
   `route_equivalence.png` were produced under headless plotting and inspected
   (bridge pinned at both ends with diamond heatmap; BM `min(t,s)` ramp; OU banded
   stationary heatmap; route figure with all points inside the band).

4. **No CI/library regression.** The commit touches no `src/` or `test/` file; the
   deterministic suite still passes 146/146, consistent with the two-tier split
   (this experiment is intentionally outside CI). Reproducibility conventions are
   satisfied: single recorded `StableRNG(271828)` seed, `jitter = 1e-10` reported,
   `GKSwstype` set before `gr()`, correct `n_grid × N` path orientation.

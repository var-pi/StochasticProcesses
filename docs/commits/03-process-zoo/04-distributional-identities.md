# Commit 4 — distributional identities (Unit 3 "process zoo", feature `03-process-zoo`)

## TL;DR

Appends a second half to the Unit 3 experiment script,
`experiments/03_process_zoo/run.jl` — a **"Phase 4 — distributional identities"**
block (~115 new lines) plus one new import line and one dependency
(`SpecialFunctions` v2.8.0, staged into `experiments/Project.toml` and
`experiments/Manifest.toml`). It generates three new committed PNG figures under
`experiments/03_process_zoo/figures/` (`cramer_wold.png`, `kl_coefficients.png`,
`negative_control.png`). No `src/` or `test/` file is touched — this is a pure
gallery commit, so it changes no library behaviour and adds nothing to CI
(Monte-Carlo experiments are deliberately kept out of CI per `CLAUDE.md`).

Commit 3 established Unit 3's core idea, **reconciliation**: the library's three
independent square roots of a covariance operator (Cholesky, Karhunen–Loève,
circulant embedding) all agree with each other on the OU process, judged against
a split-half bootstrap null. This commit exercises a *different* class of
guarantee — **distributional identities**: theorems that assert two differently
*constructed* processes have the same probability law. The headline one is the
**self-similarity of Brownian motion**, `c^{-1/2}·W(ct) =d W(t)` (the rescaled,
time-stretched motion has the same law as the original). The commit checks five
such identities and — crucially — the intellectual content of the commit is
*how* a finite Monte-Carlo run is even allowed to check an equality of
infinite-dimensional distributions. That licensing chain (Gaussian-by-
construction → covariance match → Cramér–Wold projections) is the heart of this
document and is spelled out in full below.

The five checks: (1) **self-similarity** of BM against a BM-specific split-half
null band, with a deliberately-**wrong-exponent negative control** proving the
band has teeth; (2) **Cramér–Wold** — six fixed 1-D linear projections of the
rescaled draws, each Kolmogorov–Smirnov-tested against its exact Gaussian target;
(3) **KL coefficients** — projecting OU paths onto the first eight Nyström
eigenfunctions and checking the coefficients are independent `N(0, λ_k)`;
(4) **bridge endpoints** — the Brownian bridge is pinned to zero at both ends of
`[0, 1]`. All five run on their **own** `StableRNG(SEED_DIST = 20250101)` stream
(`rng4`), completely decoupled from Commit 3's `rng` stream — and the Commit-3
route-equivalence numbers came out **byte-identical** before and after this
append, proving the two phases do not interfere.

There were **no deviations from the plan** beyond two mechanical fixes applied
during the mandatory self-review pass (a `printf`-string reproducibility
restoration and a redundant-lambda cleanup), neither of which changes any RNG
draw, numeric result, or gate outcome. Both are reported under "Code review".

---

## Background: the terms you need

If you already read Commits 1–3 you know this codebase's Gaussian-process,
kernel, Brownian-bridge, KS-statistic, `nystrom_eigen`, `sample_cholesky`,
`empirical_cov`, `splithalf_band`, and √2-rescaling vocabulary. This section
recaps the ones this commit *leans on* in one or two sentences each with a
pointer, and then goes deep on what is genuinely new: self-similarity, the
Cramér–Wold theorem, the KS statistic's finite-sample Stephens correction, the
Bonferroni correction, and the Karhunen–Loève variance identity. Nothing below is
left as an unexplained symbol.

### Recapped from earlier commits (brief, with pointers)

- **Gaussian process (GP) and covariance kernel `R(t, s)`.** A random function
  `X(t)` whose values on any finite set of times are jointly Gaussian; here
  always zero-mean, so it is fully described by its covariance kernel
  `R(t, s) = Cov(X(t), X(s))`. "Which process" is just "which kernel". Full
  treatment in Commit 1's Background; recapped in Commit 3's.

- **`assemble_cov(gp, grid)` and the covariance matrix `Σ`.** Evaluate the kernel
  on a grid of `n` times to get the `n × n` matrix `Σ[i,j] = R(t_i, t_j)`,
  wrapped `Symmetric`. Every sampler and the eigenproblem is an operation on this
  one matrix.

- **`sample_cholesky(Σ, rng; jitter)`.** Factor `Σ + ε·I = L·Lᵀ` and return
  `L·z` for `z ~ N(0, I)`; the resulting vector has covariance `Σ + εI ≈ Σ`. The
  **jitter** `ε = JITTER = 1e-10` (the reported nugget) is added before factoring
  so a singular or rounding-indefinite `Σ` does not throw. This commit uses *only*
  the Cholesky route (all four sub-checks draw via `sample_cholesky`) — see the
  design-choices section for why. Detailed in Commit 3's Background.

- **`empirical_cov(paths)`.** Given `paths` as an **`n_grid × N` matrix, one
  sample path per COLUMN** (a load-bearing orientation from `CLAUDE.md`; a
  transpose silently estimates the wrong matrix), return the unbiased
  `n_grid × n_grid` sample covariance `Σ̂ = X_c·X_cᵀ / (N−1)`. As `N → ∞`,
  `Σ̂ → Σ`. Detailed in Commit 3's Background.

- **`nystrom_eigen(R, nodes, weights)` (the KL engine).** Discretizes the
  covariance-operator eigenproblem `∫ R(t,s) e(s) ds = λ e(t)` by a quadrature
  rule and solves it *symmetrized* (`W^{1/2} K W^{1/2} g = λ g`, `e = W^{-1/2} g`,
  where `W = diag(weights)`), returning eigenvalues sorted descending and an
  `n×nev` matrix of `W`-orthonormal eigenfunctions sampled on the nodes. The
  eigenvalues `λ_k` are simultaneously eigenvalues of the covariance operator
  **and** the variances of the KL expansion coefficients — the identity this
  commit's KL-coefficient check exploits (derived in full below). Detailed in
  Commit 3's Background.

- **`ks_statistic(samples, cdf)` (Commit 2).** The Kolmogorov–Smirnov
  sup-distance `D_n = sup_x |F_n(x) − F(x)|` between the sample's empirical CDF
  `F_n` (the step function that jumps by `1/n` at each sorted sample) and a
  target CDF `F`. Small `D_n` means the sample hugs `F` everywhere; large `D_n` is
  evidence the sample is *not* from `F`. This commit is `ks_statistic`'s first
  in-repo caller — Commit 2 built it ahead of need for exactly this. Full
  treatment (including why both sides of each ECDF jump are checked) in Commit 2.

- **`splithalf_band(paths, nsplit, rng)` (Commit 3).** A **bootstrap null**: it
  repeatedly re-partitions one process's own `N` sample paths into two disjoint
  halves and records the Frobenius distance `‖Σ̂_halfA − Σ̂_halfB‖_F` between the
  two halves' empirical covariances, returning the central `[2.5%, 97.5%]`
  quantile band of that statistic. That band *is* the answer to "how much does
  this process's empirical covariance disagree with itself, at this sample size,
  under sampling noise alone?" This commit **reuses the exact same helper** from
  Commit 3, but feeds it **Brownian-motion** paths to get a BM-scale band (not
  the OU band of Commit 3). Full derivation in Commit 3's Background.

- **The Frobenius covariance-estimation-noise statistic and the √2 rescaling
  (Commit 3).** Two *independent* `M`-sample estimates of the same true `Σ` are
  never equal; for zero-mean Gaussian data the expected squared Frobenius
  distance between them is `2·((trΣ)² + ‖Σ‖_F²)/M`, so the distance concentrates
  around `√2·σ₁ > 0`, **not** zero. A cross statistic computed at the full sample
  size `N` is rescaled by `√2` to put it on the *half*-size scale of the
  split-half band (halving the sample size doubles each entry's variance, an
  extra `√2` on the distance), so both concentrate at the same `2σ₁`. The full
  factor-by-factor algebra is in Commit 3's "Why these design choices". This
  commit reuses that machinery verbatim for the self-similarity check, so
  `ss_stat = sqrt(2)·norm(empirical_cov(Y) − Σ̂_bm)` is a full-vs-full cross
  distance rescaled to the BM band's half-size scale.

### New in this commit

- **Self-similarity (scaling invariance) of Brownian motion.** Standard Brownian
  motion `W(t)` has the exact distributional identity, for any fixed scale
  `c > 0`,

  ```
  c^{-1/2} · W(c·t)  =d  W(t),
  ```

  where `=d` means "has the same probability law as" (equality in distribution,
  i.e. of the whole random function, not of one value). Read it as: stretch time
  by `c` (look at the motion running `c` times faster), then shrink amplitude by
  `√c`, and you recover a process **statistically indistinguishable** from the
  original. The reason is immediate from the covariance: `W` is zero-mean
  Gaussian with `Cov(W(t), W(s)) = min(t, s)`, so the rescaled process
  `Y(t) = c^{-1/2}·W(ct)` is zero-mean Gaussian with

  ```
  Cov(Y(t), Y(s)) = c^{-1} · Cov(W(ct), W(cs)) = c^{-1} · min(ct, cs)
                  = c^{-1} · c · min(t, s) = min(t, s),
  ```

  i.e. exactly `W`'s own covariance. The exponent `1/2` is the *only* one that
  makes the two `c`'s cancel; any other exponent leaves a residual power of `c`
  and breaks the identity. That is precisely why the wrong exponent `1/3` is a
  legitimate negative control (below): it multiplies the covariance by
  `c^{1 − 2/3} = c^{1/3} = 4^{1/3} ≈ 1.587`, a gross, easily-detected
  distortion. (Pavliotis §1.5 / Def. 1.14 gives BM's defining properties;
  self-similarity is its Brownian-scaling property.)

- **Cramér–Wold theorem (the "projections determine the law" theorem).** A
  theorem of probability: the joint law of a random vector `X ∈ ℝ^n` is
  **completely determined** by the laws of *all* its one-dimensional linear
  projections `aᵀX = Σ_i a_i X_i`, ranging over every direction `a ∈ ℝ^n`. In
  words: if you know the distribution of every "shadow" of `X` cast onto every
  line through the origin, you know `X` itself. (The proof is via characteristic
  functions: `E[exp(i·t·aᵀX)]` as `a` and `t` vary is the full multivariate
  characteristic function of `X`, which determines its law.) The practical
  consequence used here: to check "`X` has the target law", it suffices — in
  principle — to check that `aᵀX` has the correct 1-D law for *every* `a`. A
  finite experiment cannot test every `a`, so it picks a hand-chosen family of
  directions; each projected sample is a 1-D dataset that a KS test can compare
  against its exact scalar Gaussian target. This is a *strictly stronger* line of
  evidence than a covariance match (see "The licensing chain" below), which is
  the whole reason it is run *alongside* the covariance-band check.

- **KS statistic recap + why a finite-`n` correction is needed.** `ks_statistic`
  (Commit 2) returns the raw sup-distance `D_n`. Under the null (the sample truly
  comes from the target CDF), `D_n → 0` as `n → ∞`, and the *rescaled* quantity
  `√n · D_n` converges to a fixed, sample-size-free limiting distribution — the
  **Kolmogorov distribution**. That limit is what lets a single critical value
  serve all large `n`. But at finite `n` the plain `√n · D_n` still carries a
  small `n`-dependent bias away from the limiting law.

- **Stephens (1974) finite-sample correction.** Stephens' widely-tabulated
  refinement multiplies the raw statistic by an `n`-dependent factor so the
  *modified* statistic follows the asymptotic Kolmogorov distribution far more
  closely at finite `n`:

  ```
  D* = (√n + 0.12 + 0.11/√n) · D_n.
  ```

  For large `n` the bracket is `≈ √n`, so `D*` ≈ the usual `√n·D_n`; the `+0.12`
  and `+0.11/√n` terms are the small finite-sample nudges. In the code this is
  `stephens(Dks, n) = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * Dks`. Comparing `D*`
  against a fixed Kolmogorov critical value gives a hypothesis test whose actual
  rejection rate matches its nominal level even at the `n = 4000` used here.

- **The Kolmogorov critical value.** For a target false-rejection probability
  (significance level) `β`, the Kolmogorov distribution's upper-`β` quantile is,
  to the standard leading approximation, `kscrit(β) = √(−0.5·ln(β/2))`. If the
  modified statistic `D*` exceeds `kscrit(β)`, the test rejects the null at level
  `β`. Small `β` → large critical value → a *harder* test to fail (rejects only
  on egregious mismatches). In the code: `kscrit(β) = sqrt(-0.5*log(β/2))`.

- **Bonferroni correction (controlling a *family* of tests).** When you run `M`
  hypothesis tests simultaneously and want the *family-wise* probability of *any*
  false rejection to stay at most `α`, testing each one at the naive per-test
  level `α` is wrong: with `M` independent tests each at level `α`, the chance
  that at least one fires by luck is `1 − (1−α)^M ≈ M·α`, far above `α`. The
  **Bonferroni correction** is the simplest fix: test each individual hypothesis
  at the stricter level `β = α/M`. Then the union bound guarantees the family-wise
  error is at most `M·(α/M) = α` regardless of dependence between the tests. Here
  the code runs `M = 6` Cramér–Wold tests and `M = 8` KL-coefficient tests; each
  uses `kscrit(0.01/M)`, i.e. a family-wise `α = 0.01`. Section "Why these design
  choices" explains why `α = 0.01` and not the naive `0.05`.

- **The Karhunen–Loève variance identity `Var(ξ_k) = λ_k` (exact, not
  approximate).** The KL expansion writes a zero-mean process as
  `X(t) = Σ_k ξ_k·e_k(t)`, where `e_k` are the eigenfunctions of the covariance
  operator (`∫ R(t,s) e_k(s) ds = λ_k e_k(t)`, `W`-orthonormal:
  `∫ e_j e_k = δ_jk`) and the **coefficients** are the projections
  `ξ_k = ∫ X(s) e_k(s) ds`. The variances of these coefficients are *exactly* the
  eigenvalues, by direct substitution:

  ```
  Var(ξ_k) = E[ξ_k²] = E[ (∫ X(t) e_k(t) dt)(∫ X(s) e_k(s) ds) ]
           = ∫∫ E[X(t)X(s)] e_k(t) e_k(s) dt ds
           = ∫∫ R(t,s) e_k(t) e_k(s) dt ds        (R = Cov, zero mean)
           = ∫ e_k(t) [ ∫ R(t,s) e_k(s) ds ] dt
           = ∫ e_k(t) · λ_k e_k(t) dt             (eigenproblem)
           = λ_k · ∫ e_k(t)² dt = λ_k · 1 = λ_k.  (orthonormality)
  ```

  And a *cross*-covariance `Cov(ξ_j, ξ_k)` runs through the identical algebra to
  `λ_k·∫ e_j e_k = λ_k·δ_jk = 0` for `j ≠ k`, so distinct coefficients are
  **uncorrelated** — and, being jointly Gaussian, therefore **independent**. This
  is an *algebraic* identity of the KL/Nyström construction, not an approximation
  that improves with sample size or truncation depth. It is exactly why the
  KL-coefficient KS check can target `N(0, λ_k)` with the **analytic** Nyström
  eigenvalue `ou_lambdas[k]` and call it a *true* null rather than a plug-in
  estimate. (In the discretized code the integrals become quadrature sums,
  `ξ_k = Σ_i w_i X(t_i) e_k(t_i)`, and the identity holds exactly for the
  Nyström-discretized covariance — see "Why these design choices".)

---

## What changed

Two edits to the one existing file, plus its three new figures and the dependency
manifests. No library or test file changes.

### `experiments/03_process_zoo/run.jl` (+116 lines; the append)

**Edit 1 — one new import.** After the existing
`using StochasticProcesses` / `using StableRNGs, LinearAlgebra, Printf, Plots`,
a third line:

```julia
using SpecialFunctions: erf
```

`SpecialFunctions` (v2.8.0) supplies the Gauss error function `erf`, used to build
the standard-normal CDF the KS tests compare against (`normcdf`, below). The
dependency was added to `experiments/Project.toml` and pinned in
`experiments/Manifest.toml` (both staged as part of this commit). It is an
**experiments-only** dependency — it is deliberately *not* promoted to the root
package env (see the declined "altitude" code-review finding).

**Edit 2 — replace the old final `recorded:` line with the Phase-4 block.** The
file's previous last two lines (Commit 3's `recorded:` printf) are replaced by a
new `# Phase 4 — distributional identities` section plus an expanded `recorded:`
line. Everything from the `Phase 4` comment banner to the end of the file is new;
the first ~145 lines (Commit 3) are unchanged except for the one added import.

**The new constants (every one, and why its value).**

| const | value | meaning / why |
|-------|-------|----------------|
| `SEED_DIST` | `20250101` | The Phase-4 `StableRNG` seed for its own stream `rng4`, **recorded** in the final `printf`. Deliberately distinct from Commit 3's `SEED = 271828` so the two phases' RNG streams are independent (see design choices). |
| `C_SCALE` | `4.0` | The self-similarity scale `c`. At `c = 4`, the correct rescaling divides by `√4 = 2`; the wrong `c^{1/3}` control leaves a factor `4^{1/3} ≈ 1.587` in the covariance — large enough to be caught, small enough to be a plausible "near-miss" impostor. |
| `K_KL` | `8` | Number of leading KL eigen-modes whose coefficients are checked (the top-8 of the OU spectrum, which carry the bulk of the variance). Also the Bonferroni family size for the KL KS tests. |
| `N_BRIDGE` | `200` | Bridge paths drawn for the endpoint check. Endpoint pinning is a per-path structural fact (each path's endpoint is `√JITTER·z`), so a small sample already exercises the worst case over `2·N_BRIDGE = 400` endpoint values. |

It reuses Commit 3's constants unchanged: `N_ROUTE = 4000` (paths per identity
check), `N_SPLIT = 200` (split-half re-partitions for the BM band), `N_GRID = 64`,
`JITTER = 1e-10`, and the `bm_grid`, `bridge_grid`, `ou_nodes`, `ou_w`, `Σ_ou`,
`ou_lambdas`, `ou_eigfuncs` objects built earlier in the file.

**The three new local helpers.**

- `normcdf(z) = 0.5 * (1 + erf(z / sqrt(2)))` — the standard-normal CDF
  `Φ(z) = P(N(0,1) ≤ z)`, built from the error function via the textbook identity
  `Φ(z) = ½(1 + erf(z/√2))`. Every KS target below is `z ↦ normcdf(z/σ)` for the
  appropriate scalar standard deviation `σ` (i.e. the CDF of `N(0, σ²)`).
- `kscrit(β) = sqrt(-0.5 * log(β / 2))` — the Kolmogorov critical value at level
  `β` (Background).
- `stephens(Dks, n) = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * Dks` — the Stephens
  (1974) finite-sample modification of a raw KS statistic (Background).

**The five checks (in file order).** All draw via `sample_cholesky(..., rng4)`,
so every stochastic call is on the single Phase-4 `StableRNG(SEED_DIST)` stream.

1. **BM reference + its own null band.** Assemble `Σ_bm` on `bm_grid`, draw
   `N_ROUTE` BM paths (`bm_paths`), form `Σ̂_bm = empirical_cov(bm_paths)`, and
   build the BM-scale split-half band `bm_lo, bm_hi = splithalf_band(bm_paths,
   N_SPLIT, rng4)`. This is the yardstick the self-similarity statistic is judged
   against — the BM analogue of Commit 3's Cholesky-generated OU band.

2. **Self-similarity (and its negative control).** Assemble a *second* BM
   covariance on the **stretched grid** `C_SCALE .* bm_grid` (i.e. sample
   `W` at times `{c·t_i}`), and draw `wct` once from it. Then scale that **single
   draw** two ways: `Y = C_SCALE^(-0.5) .* wct` (the correct self-similar
   rescaling, whose law should equal `W`'s) and `Y_wrong = C_SCALE^(-1/3) .* wct`
   (the deliberately wrong exponent). Compute two √2-rescaled Frobenius distances
   against the BM reference covariance: `ss_stat = sqrt(2)*norm(empirical_cov(Y)
   .- Σ̂_bm)` and `ctrl_stat = sqrt(2)*norm(empirical_cov(Y_wrong) .- Σ̂_bm)`. The
   gate: `ss_ok = bm_lo <= ss_stat <= bm_hi` (correct exponent lands *in* band)
   and `ctrl_out = !(bm_lo <= ctrl_stat <= bm_hi)` (wrong exponent lands
   *outside*).

3. **Cramér–Wold.** Six fixed projection vectors `a` (`cw_funcs`, labelled
   `["mean", "ramp", "quadratic", "sine", "endpoint diff", "wiggle"]`) are dotted
   against the columns of `Y` (the *correctly* rescaled self-similar draws), each
   yielding a 1-D sample of `N_ROUTE` values `{aᵀY_j}`. Each is KS-tested against
   its exact target `N(0, aᵀΣ_bm·a)` — since `Y` is Gaussian and BM-covariant by
   construction, `aᵀY` is exactly `N(0, aᵀΣ_bm a)`, a true null. Each raw KS stat
   is Stephens-corrected, and the family passes iff every one is below
   `cw_crit = kscrit(0.01/6)` (Bonferroni family-wise `α = 0.01`).
   `cw_ok = all(cw_stats .< cw_crit)`.

4. **KL coefficients.** Draw a *fresh* set of `N_ROUTE` OU paths from `rng4`
   (`ou_paths` — distinct from Commit 3's `chol_paths`/`kl_paths`/`circ_paths`),
   and project each onto the first `K_KL = 8` Nyström eigenfunctions weighted by
   the quadrature weights: `ξ = ou_eigfuncs[:, 1:K_KL]' * (ou_w .* ou_paths)`, a
   `K_KL × N_ROUTE` matrix of KL coefficients. Two checks: (a) normalize each
   coefficient row to unit norm (`ξn`), form the empirical correlation matrix
   `Ccorr = ξn * ξn'`, and require `maxoff = maximum(abs.(Ccorr - I)) < 0.10`
   (coefficients pairwise near-uncorrelated); (b) KS-test each row against
   `N(0, ou_lambdas[k])` (the **analytic** eigenvalue), Stephens-corrected,
   Bonferroni `α = 0.01` over `K_KL = 8`. `kl_ok = maxoff < 0.10 && all(kl_stats
   .< kl_crit)`.

5. **Bridge endpoints.** Assemble the Brownian-bridge covariance `Σ_br` on
   `bridge_grid`, draw `N_BRIDGE = 200` bridge paths, and check
   `maxend = maximum(abs.(vcat(br_paths[1,:], br_paths[end,:]))) <= br_tol` with
   `br_tol = 10*sqrt(JITTER)`. Because `R(0,0) = R(1,1) = 0` exactly for the
   bridge, after the `JITTER` nugget the endpoint diagonal is `JITTER` and the
   endpoint value is `√JITTER·z` for standard normal `z` — pinned to the jitter
   floor `√JITTER ≈ 1e-5`, not to exactly 0. `br_tol = 10·√JITTER ≈ 1e-4` is a
   ~10-sigma margin around that floor. `br_ok = maxend <= br_tol`.

The block then prints a five-line report and the aggregate
`distributional identities -> PASS/FAIL` (all five sub-gates ANDed:
`ss_ok && ctrl_out && cw_ok && kl_ok && br_ok`), saves the three figures, and
prints the expanded `recorded:` constants line.

### The three figures (committed artifacts)

`experiments/03_process_zoo/figures/{cramer_wold,kl_coefficients,negative_control}.png`,
generated headless (`ENV["GKSwstype"] = "100"` set earlier in the file by
Commit 3) and committed per the `CLAUDE.md` "commit their figures" rule. Their
content is described in "Every figure/statistic, individually".

---

## Why these design choices

### The licensing chain: how a finite experiment checks an infinite-dimensional law

This is the intellectual core of the commit and deserves to be stated as its own
argument, because it is not obvious that the checks are even *legitimate*.

A distributional identity like `c^{-1/2}·W(ct) =d W(t)` is an equality of two
**probability distributions on an infinite-dimensional path space** — it asserts
`P(Y ∈ B) = P(W ∈ B)` for *every* measurable set `B` of paths. A finite
Monte-Carlo experiment cannot check that directly: you cannot enumerate all Borel
sets, and you only ever observe finitely many paths on a finite grid. So how is
the check valid at all? Through a three-step chain.

**Step 1 — `Y` is Gaussian *by construction*, not by measurement.** On the grid,
`Y = c^{-1/2}·W(ct)` is a deterministic **linear map** (here just scalar
multiplication) of the Gaussian vector `W(ct)` sampled at the stretched times
`{c·t_i}`. Any affine map of a jointly-Gaussian vector is again jointly Gaussian
— the Gaussian family is closed under linear maps: if `X ~ N(μ, Σ)` and
`Y = A·X + b`, then `Y ~ N(A·μ + b, A·Σ·Aᵀ)` (Pavliotis App. B.5, "Gaussian
random vectors"). So `Y` is **known** to be Gaussian without any test — it is an
algebraic consequence of *how it was built*, not an empirical finding. This is
what makes the whole check tractable: we are not asking "is `Y` Gaussian?" (a
hard, infinite-dimensional question) — we already know it is.

**Step 2 — for a Gaussian, "same law" reduces to "same mean and covariance".** A
Gaussian distribution is *completely* pinned down by its mean vector and
covariance matrix. Both `Y` and `W` here are zero-mean, so checking "does `Y`'s
law equal `W`'s law?" reduces to the **finite-dimensional, checkable** claim
"does `Y`'s covariance equal `W`'s covariance?". That is exactly what the
self-similarity band statistic tests: `ss_stat = √2·‖Σ̂_Y − Σ̂_bm‖_F`, judged
against the BM split-half null. The infinite-dimensional identity has been
converted into a covariance-matrix comparison at finite sample size.

**Step 3 — Cramér–Wold adds an orthogonal, *stronger* line of evidence.** The
covariance-band check has a blind spot. It is computed at a fixed sample size
with sampling noise, and — more importantly — it could *in principle* be fooled
by badly-non-Gaussian data that happens to reproduce the right second moments in
aggregate. Imagine a hypothetical sampler bug that got the covariance exactly
right but drew from a **non-Gaussian distribution with matching second moments**
(a "non-Gaussian impostor with the right covariance"). The Frobenius-norm band
check would *pass* — the covariances match — and yet the law would be wrong. The
Cramér–Wold projections catch precisely this: Cramér–Wold's theorem says a
vector's **full law** (not merely its covariance) is determined by the laws of
*all* its 1-D linear projections, so KS-testing several projections against their
exact Gaussian targets checks something **logically stronger** than a covariance
match. A non-Gaussian impostor with the right covariance would still show a KS
mismatch on some projection (its 1-D shadow would be non-Gaussian even though its
variance is right). This is the concrete reason **both** checks exist rather than
just the cheaper covariance-band check alone: the band check is fast and pins the
*second moments*; the projections are the guard against a right-covariance,
wrong-shape distribution. (In this specific experiment `Y` genuinely *is*
Gaussian by construction, so both must pass — the value of running both is that
the pair would jointly catch a broader class of sampler regressions than either
alone.)

### The wrong-exponent negative control is what gives the band teeth

Without `Y_wrong`, a reader cannot tell whether `ss_stat` landing *in band* is a
real confirmation or whether the band is simply so wide it would accept anything.
The control resolves this: `ctrl_stat` is computed identically to `ss_stat` but
from the wrong-exponent rescaling, whose covariance is inflated by
`4^{1/3} ≈ 1.587`. It lands at `23.865` versus the passing band `[0.422, 2.398]`
— roughly an **order of magnitude outside**. That gulf demonstrates the check has
real discriminating power: the band accepts the correct rescaling and decisively
rejects a plausible-looking wrong one, so an in-band `ss_stat` is a genuine
confirmation, not a vacuous one.

### Draw `W(ct)` once, scale twice (paired comparison)

`Y` and `Y_wrong` are both formed from the *same* underlying draw `wct`, rather
than from two independent BM-at-scale-`c` samples. This is a deliberate
**paired-comparison** design: because both branches share the identical sampling
noise, the *only* difference between `ss_stat` and `ctrl_stat` is the rescaling
exponent itself (0.5 vs 1/3). It isolates the effect of the exponent cleanly,
rather than conflating it with independent-sampling scatter that a two-draw design
would inject. It is also one fewer `N_ROUTE`-sized Cholesky sample to draw.

### Its own RNG stream `rng4` (not Commit 3's `rng`)

Phase 4 runs on `StableRNG(SEED_DIST = 20250101)` — a **fresh, independent**
stream — deliberately *not* threading Commit 3's `rng` into it. This decouples the
two phases: either can be reseeded or debugged without perturbing the other's
committed numbers. It was verified operationally: after appending Phase 4, the
Commit-3 route-equivalence statistics printed **byte-identical** values
(`Chol–KL = 2.2411`, `Chol–Circ = 2.5811`, `KL–Circ = 1.8591`) to Commit 3's
committed output — proof that Phase 4 pulls from a disjoint stream and does not
consume or reorder a single draw of the route-equivalence `rng`. (The
`CLAUDE.md` "do not reorder RNG draws" rule is about a *single* stream; using a
*second, independent* stream for a logically separate phase is the clean way to
add draws without disturbing the first.)

### Bonferroni family-wise `α = 0.01` for both KS families

Both KS families use `kscrit(0.01/M)` (`M = 6` for Cramér–Wold, `M = 8` for KL),
i.e. a family-wise `α = 0.01`, **not** the naive per-test `0.05`. The reason is
multiple-testing over-rejection, and it was *measured* by the dispatching plan:
at a naive per-test `α = 0.05`, the KL max-of-8 KS gate **false-failed on 2 of 8
tested seeds** — purely because running 8 simultaneous 5%-level tests gives a
family-wise false-positive rate well above 5% (roughly `1 − 0.95^8 ≈ 34%`). At
family-wise `α = 0.01` (per-test `β = 0.01/8`), the family passes **all 8** tested
seeds *while the gate still has real power*: the deliberately-wrong-covariance
control statistic sits at `ctrl_stat = 23.865` versus the passing band's scale of
~2.4 — an order-of-magnitude separation — so tightening `α` traded away spurious
false-fails without blunting the check's ability to catch a genuine impostor.

### Why the analytic eigenvalue `λ_k` is a *true* null, not a plug-in

The KL-coefficient KS test targets `N(0, ou_lambdas[k])` using the **analytic**
Nyström eigenvalue, not an empirically re-estimated coefficient variance. This is
legitimate because `Var(ξ_k) = λ_k` **exactly** — the algebraic identity derived
in full in the Background (substitute the eigenproblem into the coefficient's
second moment; the eigenvalue falls straight out via orthonormality). It is not
an approximation that improves with `N_ROUTE` or with truncation depth; it holds
for the Nyström-discretized covariance operator by construction. So targeting
`N(0, λ_k)` with the analytic `λ_k` tests against the *exact* null law, giving the
KS test its full, uncompromised power. Had the target instead used a
sample-estimated variance, the test would be checking the coefficients against a
noisy plug-in and would lose bite.

### Bridge endpoints checked structurally, via the covariance route

The bridge endpoint identity `R(0,0) = R(1,1) = 0` is checked by sampling through
the **covariance/Cholesky route** and inspecting the drawn endpoint values —
*not* by a second, independent sampling route. The natural "second route" would be
the time-change construction `B_t = (1−t)·W(t/(1−t))`, but that formula is
**undefined at `t = 1`** (division by zero in the argument `t/(1−t)`), the very
endpoint the identity is about. So a second-sampling-route cross-check is not
available at the boundary; the structural covariance check is the correct
substitute rather than a weaker stand-in — it verifies the exact fact the identity
asserts (`R = 0` at the ends, hence paths pinned there), just via the covariance
matrix instead of via a redundant sampler. See "Trade-offs".

### Cholesky for every sub-check

All four sub-checks sample via `sample_cholesky`, not a mix of routes. Route
equivalence — the fact that Cholesky, KL, and circulant agree — was *established*
in Commit 3; this commit takes it as given and uses the single most general route
(Cholesky works for any covariance matrix, stationary or not, and all four
processes here — BM, stretched BM, OU, bridge — are handled uniformly). The
`√JITTER` nugget is added on every factorization and is what makes the bridge
endpoint variance `JITTER` rather than exactly 0.

---

## Every figure/statistic, individually (and why it bites)

Treating each figure and each printed statistic the way Commits 1–2 treat each
`@test`: what specific bug would move it, and how the artifact catches it. Unlike
the deterministic `test/` tier, these are Monte-Carlo artifacts — they "bite" by
moving visibly/measurably under a real defect, calibrated to the sampling noise,
not by an exact equality.

1. **Printed — self-similarity `ss_stat` in the BM band** (`1.439` in
   `[0.422, 2.398]` this run → PASS). The direct test of `c^{-1/2}W(ct) =d W(t)`,
   reduced (via the licensing chain) to a covariance match on the BM null scale.
   A wrong self-similarity exponent, or a `sample_cholesky` bug that mis-scaled
   the stretched-grid covariance, would push this out of the band. Its bite is
   calibrated *by the BM band itself* — the same split-half self-noise machinery
   Commit 3 validated.

2. **Printed — negative control `ctrl_stat` OUTSIDE the band** (`23.865`, band
   `[0.422, 2.398]` → "outside band (fails on purpose)"). This is the gate's
   proof-of-power: it *must* land outside, and the aggregate PASS requires
   `ctrl_out == true`. If a bug made the band absurdly wide (accepting even the
   `c^{1/3}` distortion), `ctrl_out` would flip to false and the whole
   distributional-identities gate would FAIL — catching a band-too-wide
   pathology that `ss_stat` alone (which only asks the correct rescaling to land
   *in* band) could never surface. The ~10–20× separation between `23.865` and
   the band is the quantitative evidence the check discriminates.

3. **Printed — Cramér–Wold `max Stephens KS` vs `crit`** (`1.099 < 1.883`, family
   `α = 0.01`, 6 proj → PASS). The strongest single line of evidence that `Y`'s
   *full law* (not just covariance) matches `W`'s. A non-Gaussian impostor with
   the right covariance, or a projection whose target variance `aᵀΣ_bm a` was
   miscomputed, would push one of the six Stephens-corrected statistics above the
   Bonferroni critical value. The six directions are chosen to probe different
   structural axes (a flat average, a linear ramp, a quadratic, a smooth sine, a
   sparse endpoint-difference, and an irregular two-frequency "wiggle") so the
   family is not accidentally blind to one class of departure.

4. **Printed — KL coefficients `max|corr offdiag|` and `max KS`** (`0.0375 <
   0.10` and `1.358 < 1.921` → PASS). Two independent facts about the KL
   coefficients: (a) the off-diagonal correlations near zero confirm the
   coefficients are pairwise **uncorrelated** (the KL independence property); a
   `nystrom_eigen` regression that returned non-orthogonal eigenfunctions, or a
   wrong weighting in `ξ = ou_eigfuncs[:,1:K_KL]' * (ou_w .* ou_paths)`, would
   inflate the off-diagonals. (b) each coefficient's marginal is `N(0, λ_k)` with
   the *analytic* λ_k; a wrong eigenvalue or a mis-projection would fail the KS
   test. Together they check both the joint (independence) and the marginal
   (variance-`λ_k`) structure of the KL expansion.

5. **Printed — bridge endpoints `maxend` ≤ `br_tol`** (`2.76e-05 ≤ 1.00e-04` →
   PASS). Confirms `R(0,0) = R(1,1) = 0`: every one of the `2·N_BRIDGE = 400`
   endpoint values sits at the `√JITTER ≈ 1e-5` floor, not larger. A kernel bug
   that left the endpoints with real variance (e.g. accidentally using BM's
   `min(t,s)`, whose `R(1,1) = 1`) would blow `maxend` up to order 1, far above
   `br_tol`. The tolerance is anchored to the *jitter floor*, so it bites at the
   correct scale — it is not a loose absolute constant.

6. **Printed — aggregate `distributional identities -> PASS`.** The AND of all
   five sub-gates (`ss_ok && ctrl_out && cw_ok && kl_ok && br_ok`). One line that
   flips to FAIL if any single identity, or the negative control, regresses.

7. **Printed — the byte-identical route-equivalence numbers.** Not a Phase-4
   statistic per se, but a load-bearing one: the Commit-3 block still prints
   `2.2411 / 2.5811 / 1.8591`. Any change to those values would mean Phase 4 had
   perturbed the Commit-3 `rng` stream (a reordered or shared draw) — so their
   invariance is the concrete evidence for the independent-streams design.

8. **`cramer_wold.png` — the 2×3 projection grid.** One panel per Cramér–Wold
   direction, each showing the empirical CDF of `{aᵀY_j}` (solid) against its
   exact Gaussian target `N(0, aᵀΣ_bm a)` (dashed), titled with that projection's
   Stephens KS statistic. A reader sees at a glance whether every empirical curve
   hugs its dashed target. A single panel whose solid curve visibly departed from
   its dashed target would be the visual signature of a projection-level
   distributional mismatch.

9. **`kl_coefficients.png` — two panels.** Left: a `|correlation|` heatmap of the
   8 KL coefficients (`clims (0,1)`), which must be **bright on the diagonal and
   dark off it** (≈ identity) — the visual companion to `maxoff < 0.10`. Right: an
   ECDF overlay for `ξ_1` and `ξ_8` against their respective `N(0, λ_k)` targets;
   `ξ_1` (largest variance `λ_1`) and `ξ_8` (small `λ_8`) have visibly different
   spreads, and each empirical curve should track its own dashed Gaussian — a
   check that the *per-mode* variance scaling is right, not just one mode.

10. **`negative_control.png` — the horizontal null-band bar.** The BM split-half
    band drawn as a thick horizontal bar, with `ss_stat` marked as a passing
    scatter point *inside* it and `ctrl_stat` marked with an `xcross` far
    *outside*. This is the single most legible artifact of the commit: the correct
    exponent sits in the band, the wrong one is visibly off the chart to the
    right, making the "the check discriminates" claim self-evident at a glance.

---

## Empirical / runtime verification

Running `julia --project=experiments experiments/03_process_zoo/run.jl` against
the committed code produced (verbatim; the run was executed **twice and printed
identically both times**, confirming determinism):

```
route equivalence (split-half null band [1.6120, 2.8825], null-scale theory 2σ₁=2.1966):
  Chol–KL    √2·‖Σ̂_A−Σ̂_B‖_F = 2.2411  -> in band
  Chol–Circ  √2·‖Σ̂_A−Σ̂_B‖_F = 2.5811  -> in band
  KL–Circ    √2·‖Σ̂_A−Σ̂_B‖_F = 1.8591  -> in band
  band brackets theory 2σ₁: yes
route equivalence -> PASS

distributional identities (SEED_DIST=20250101):
  self-similarity c^-1/2 W(ct): √2‖Σ̂_Y−Σ̂_BM‖=1.439 in bm_band [0.422,2.398] -> PASS
  negative control c^-1/3:      √2‖Σ̂_wrong−Σ̂_BM‖=23.865 -> outside band (fails on purpose)
  Cramér–Wold: max Stephens KS=1.099 < crit=1.883 (family α=0.01, 6 proj) -> PASS
  KL coeffs: max|corr offdiag|=0.0375 (<0.10), max KS=1.358 < crit=1.921 -> PASS
  bridge endpoints: max|X(0)|,|X(1)|=2.76e-05 ≤ 1.00e-04 -> PASS
distributional identities -> PASS

recorded: T_OU=5.0 N_GRID=64 D=1.0 alpha=1.0 N_ROUTE=4000 N_SPLIT=200 N_BRIDGE=200 c=4.0 K_KL=8 jitter=1e-10 seed=271828 seed_dist=20250101
```

Every sub-gate reads PASS, the negative control correctly reads "outside band
(fails on purpose)", and the aggregate prints
`distributional identities -> PASS`. Two facts worth calling out:

- **The route-equivalence block is byte-identical to Commit 3's committed
  output** (`2.2411 / 2.5811 / 1.8591`, band `[1.6120, 2.8825]`, theory
  `2.1966`), confirming the Phase-4 append on its own `rng4` stream did not
  perturb the Commit-3 `rng` stream in any way — no shared or reordered draw.

- **The expanded `recorded:` line captures every constant that governs the run**,
  including the two OU kernel parameters (`D`, `alpha`), the new Phase-4 constants
  (`N_BRIDGE`, `c = C_SCALE`, `K_KL`, `seed_dist`), and the `jitter = 1e-10`
  nugget — matching this repo's convention that the recorded line lists every
  reproducibility constant.

All three figures (`cramer_wold.png`, `kl_coefficients.png`,
`negative_control.png`) were generated headless and **visually inspected**: the
Cramér–Wold grid shows all six empirical CDFs hugging their dashed Gaussian
targets; the KL heatmap is bright-diagonal / dark-off-diagonal (≈ identity) with
`ξ_1` and `ξ_8` ECDFs tracking their `N(0, λ_k)` targets at visibly different
spreads; and the negative-control bar shows the `c^{-1/2}` point inside the BM
band with the `c^{-1/3}` xcross far outside.

The full deterministic test suite (`julia --project -e 'using Pkg; Pkg.test()'`)
was re-run after this change and still passes **146/146**. Because this commit
touches **no** `src/` or `test/` file, that is expected — it is *not* a claim that
this experiment is covered by CI. Per `CLAUDE.md`, Monte-Carlo experiments are
deliberately **not** run in CI; this script is run locally and its figures
committed.

---

## Trade-offs and known limitations

- **Self-similarity and Cramér–Wold are BM-only.** Both checks are run only for
  Brownian motion, not repeated for OU or the bridge. This is a scope choice: BM's
  scaling identity is the canonical, cleanest distributional identity to
  demonstrate the licensing chain on, and the KL-coefficient check already
  exercises the OU process's distributional structure by a different route. A
  future increment could add an OU-specific stationarity/self-similarity identity,
  but doing so here would multiply the figure count without adding a new *kind* of
  evidence.

- **The bridge-endpoint check is structural, not a second sampling route.** It
  verifies `R(0,0) = R(1,1) = 0` via the covariance/Cholesky route and inspecting
  drawn endpoint values, rather than cross-checking against an independent
  sampler. The natural alternative — the time-change identity
  `B_t = (1−t)W(t/(1−t))` — is **undefined at `t = 1`** (division by zero at the
  very endpoint being tested), so a second-route cross-check is genuinely
  unavailable at the boundary. The structural check verifies exactly the fact the
  identity asserts, so it is a legitimate substitute, not a weaker stand-in — but
  it *is* a single-route check, which is worth naming.

- **Six Cramér–Wold projections is a finite, hand-picked family.** Cramér–Wold's
  theorem is a statement about **all** projection directions; testing six of them
  is necessarily an incomplete sample of an infinite claim. A pathological
  departure that hid in a direction orthogonal to all six hand-chosen `a`'s would
  slip through. This is the same *category* of limitation as the KL-coefficient
  check covering only `K_KL = 8` of infinitely many modes — inherent to any finite
  Monte-Carlo check of an infinite-dimensional claim, mitigated (not eliminated) by
  choosing directions/modes that span structurally different axes.

- **Multiple-testing tuning is seed-informed, not adversarially proven.** The
  Bonferroni `α = 0.01` was chosen because it passes all 8 tested seeds while
  `α = 0.05` false-failed 2 of 8; that is empirical calibration over a finite seed
  set, not a proof that no seed will ever false-fail. The order-of-magnitude
  separation between the passing band and the negative control gives confidence the
  gate retains power, but a future seed at a very unlucky draw could in principle
  still trip a family — the same residual bootstrap/quantile-noise caveat Commit 3
  documents for its route-equivalence gate applies in spirit here.

---

## Code review

A `/code-review` pass (medium effort, 8 parallel finder agents spanning
correctness / removed-behavior / cross-file / reuse / simplification / efficiency
/ altitude / `CLAUDE.md`-conventions angles) was run against this diff before
commit. **Two findings were accepted and fixed; several were declined as out of
scope.** Reported factually below.

### Fixed

1. **`recorded:` line dropped `D`/`ALPHA` and never printed `N_BRIDGE`.** As
   transcribed from the plan, the new final `recorded:` printf omitted the OU
   kernel's `D` (noise strength) and `alpha` (relaxation rate) — which Commit 3's
   line *had* included — and never printed the new Phase-4 constant `N_BRIDGE`.
   Two independent finder angles flagged this as a reproducibility regression
   against this repo's own convention (every unit's `recorded:` line lists every
   constant that governs the run). **Fix:** restored `D=%.1f alpha=%.1f` and added
   `N_BRIDGE=%d` to the final `recorded:` line. This is a pure printf-string
   change with **zero** effect on any RNG draw or numeric result — confirmed by
   rerunning: all PASS/FAIL outcomes and every printed statistic were identical
   before and after.

2. **Redundant immediately-invoked lambda reimplementing `normalize`.** The sixth
   ("wiggle") Cramér–Wold projection vector was originally built as
   `(v -> v ./ norm(v))(sin.(7 .* bm_grid) .+ cos.(3 .* bm_grid))` — an
   immediately-invoked lambda that reimplements `LinearAlgebra.normalize`, which
   is already imported at the top of the file (`using ..., LinearAlgebra, ...`).
   **Fix:** replaced with `normalize(sin.(7 .* bm_grid) .+ cos.(3 .* bm_grid))`.
   Elementwise division of a vector by its own Euclidean norm is *exactly* what
   `normalize` computes, so this is numerically identical with zero effect on any
   RNG draw or result.

### Declined (with reason)

- **Promote `normcdf`/`kscrit`/`stephens` into `src/gof.jl`** (an "altitude"
  finding from 3 independent angles, reasoning that `src/gof.jl` was built in
  Commit 2 as a shared cross-unit statistics module and these helpers look like
  natural reuse candidates for, e.g., Unit 5). Declined as out of scope for this
  increment: the commit plan specified this exact code verbatim (validated
  end-to-end by the planner against the committed package with pinned expected
  output), and promoting the helpers to `src/` is a **structural library change**
  — it would need its own test coverage, would make `SpecialFunctions` a
  root-package dependency rather than experiments-only, and would need export
  wiring. That belongs to a future commit's planning, not a same-commit scope
  expansion.

- **Redundant recomputation / efficiency** (the Cramér–Wold projection vector and
  the quadratic form `a'*Σ_bm*a` are each computed twice — once for the KS
  statistic, again for the plot; plus the per-draw Cholesky factorization pattern
  already present since Commit 3). Declined per `CLAUDE.md`'s explicit
  correctness-over-efficiency preference (flag efficiency as low priority; defer
  unless it will bite the moment input grows): this is a Monte-Carlo script run
  once locally, not a hot path, and at `N_GRID = 64` / `N_ROUTE = 4000` these are
  sub-second costs.

- **Speculative future coupling** (two findings: `cw_funcs` sizing via the
  constant `N_GRID` rather than `length(bm_grid)`; and `ou_eigfuncs[:, 1:K_KL]`
  assuming `nystrom_eigen` returned at least `K_KL = 8` columns without an explicit
  bounds assertion). Declined: no concrete failure path exists in the code as
  written today — `bm_grid` is literally `range(0, T; length = N_GRID)` a few
  lines above the reused constant, and `nystrom_eigen`'s default `nev` already
  exceeds `K_KL`. These are hypothetical robustness concerns about *future* edits,
  not defects in this commit.

The self-review also confirmed the load-bearing `CLAUDE.md` conventions hold for
the new block: a single explicit `StableRNG(SEED_DIST)` stream threads every
stochastic call (no bare `randn()` / global RNG); `JITTER = 1e-10` is passed to
every `sample_cholesky` and reported in the recorded line; every path matrix is
built `n_grid × N` (one path per column) and fed to `empirical_cov` in that
orientation; and `ENV["GKSwstype"] = "100"` (set earlier by Commit 3) governs the
new figures too.

---

## Deviations from plan

The algorithm and code were transcribed **verbatim** from the dispatched plan
(per its explicit "Transcribe verbatim" instruction), with exactly the **two
mechanical fixes** described under "Code review" applied afterward during the
mandatory `/code-review` self-review pass: (1) restoring `D`/`alpha` and adding
`N_BRIDGE` to the final `recorded:` printf string, and (2) replacing a redundant
immediately-invoked lambda with `normalize`. Neither changes any numeric result,
RNG draw, or gate outcome (both verified by rerun). **No other deviation.**

---

## Pass conditions verified

1. **Script runs clean end to end.**
   `julia --project=experiments experiments/03_process_zoo/run.jl` completes
   without error and produces the transcript above.

2. **Route-equivalence numbers stayed byte-identical.** The Commit-3 block still
   prints `Chol–KL = 2.2411`, `Chol–Circ = 2.5811`, `KL–Circ = 1.8591` (band
   `[1.6120, 2.8825]`, theory `2.1966`) — proving the two RNG streams are
   independent and Phase 4 perturbed nothing upstream.

3. **`distributional identities -> PASS`** with all five sub-gates PASS
   (self-similarity in-band, Cramér–Wold, KL coefficients, bridge endpoints) and
   the negative control correctly **outside** the band (`ctrl_stat = 23.865`).

4. **Three new figures written** (`cramer_wold.png`, `kl_coefficients.png`,
   `negative_control.png`) under `experiments/03_process_zoo/figures/`, generated
   headless and visually inspected.

5. **`SpecialFunctions` resolves.** `using SpecialFunctions: erf` succeeds; the
   dependency (v2.8.0) is present and pinned in both `experiments/Project.toml`
   and `experiments/Manifest.toml`.

6. **No CI/library regression.** The commit touches no `src/` or `test/` file; the
   deterministic suite still passes **146/146**, consistent with the two-tier
   split (this experiment is intentionally outside CI).

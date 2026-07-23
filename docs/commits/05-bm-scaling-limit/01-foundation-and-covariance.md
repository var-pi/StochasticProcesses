# Unit 5 · Commit 01 — Foundation and the covariance gate

## TL;DR

Unit 5 steps off the spine that carried Units 0–4. Those units treated a stochastic process as a
covariance operator — assembled, diagonalized, square-rooted, sampled. Unit 5 demonstrates the
*other* face of Brownian motion (BM): it is the **weak (Donsker) limit of a rescaled random walk**,
a statement about laws on path space rather than an operation on a covariance operator. Per the
master plan the rescaled-walk machinery lives entirely in the experiment file, never in `src/`.

This is commit 01 of five in the unit, and it is **experiment-only** — a single new file,
`experiments/05_bm_scaling_limit/run.jl` (~245 lines), plus two committed figures. It touches no
`src/` and no `test/` code. Its job is twofold:

1. **Lay the full shared contract surface** that commits 02–04 append to and reuse: three copied
   helper functions, the three increment-law samplers, the pinned iteration order `LAW_ORDER`, and
   the `rescaled_walk` lattice builder.
2. **Add GATE 01a** — for every increment law, the rescaled walk's own two-time covariance is
   *exactly* `min(s,t)` on the lattice, and the Monte-Carlo estimate of that covariance converges to
   it at the standard `N^(-1/2)` rate. This is Unit 0's headline machinery, reused once per law.

The interesting content here is not the arithmetic — it is the **gate design**, which took three
iterations to get honest. The first two SE estimators both under-reported the true noise and either
false-failed or false-passed; the final design is classical batch means. That war story, and the
precise (and deliberately narrow) scope of what GATE 01a can catch, are the heart of this doc.

---

## Background — what Donsker's theorem says, and why the lattice suffices

Take any iid sequence of increments `X_1, X_2, …` with **mean 0 and variance 1** (nothing more — no
Gaussianity, no higher moments). Form the partial sums `S_k = X_1 + … + X_k` with `S_0 = 0`, and
rescale:

```
W_n(t) = S_{floor(n·t)} / sqrt(n),    t ∈ [0, 1].
```

Donsker's invariance principle says `W_n ⇒ W`, a standard Brownian motion, weakly on path space — and
the limit is **universal**: the same BM regardless of the increment law's shape. The `1/sqrt(n)`
scaling is exactly the CLT scaling, promoted from a single time to the whole path.

Two facts make this cheap to check computationally, and both are exploited here:

- **Only the lattice values matter.** Donsker's continuous path is built by linear interpolation
  between the lattice points `(k/n, S_k/sqrt(n))`. But every functional this unit checks — the
  two-time covariance here, the KS-vs-n marginal rate in commit 02, the running-max functional in
  commit 03 — is a function of the **lattice values alone**. So `rescaled_walk` never computes an
  interpolation; the `(n+1)`-vector of lattice values is already a complete representation.
- **The covariance is exact and n-free.** For iid variance-1 increments,
  `Cov(S_i, S_j) = min(i, j)` by linearity (the cross terms vanish by independence, the diagonal terms
  each contribute 1). Dividing by `n`,

  ```
  Cov(S_i/sqrt(n), S_j/sqrt(n)) = min(i, j)/n = min(t_i, t_j),    t_k = k/n,
  ```

  **exactly, at every lattice time, for every n.** No CLT, no limit — just linearity of covariance
  plus the variance-1 normalization. This is the identity GATE 01a pins down.

---

## What changed

One new file and two committed figures. No library or test changes.

- **`experiments/05_bm_scaling_limit/run.jl`** (new, ~245 lines) — the experiment. Structure below.
- **`experiments/05_bm_scaling_limit/figures/rescaled_paths.png`** — the qualitative Donsker picture.
- **`experiments/05_bm_scaling_limit/figures/covariance_vs_min.png`** — the GATE 01a evidence.

### The shared contract surface

Because every later phase (commits 02–04) *appends to this same file*, commit 01 lands the whole
reusable surface up front, so no later commit re-copies a helper. Three helpers are copied
**verbatim** from earlier units, with attribution comments kept honest:

- `ols_slope_se(x, y) -> (slope, se)` — self-contained OLS fit plus residual SE, copied from
  `experiments/04_ergodicity/run.jl:49-57`. Used here to fit one replicate's N-ladder at a time.
- `normcdf(z) = 0.5*(1 + erf(z/sqrt(2)))` — copied from `experiments/03_process_zoo/run.jl:161`.
- `_quantile(sorted, p)` — type-7 quantile on a pre-sorted vector, copied from
  `experiments/03_process_zoo/run.jl:23-29`.

**`normcdf` and `_quantile` are defined but unused in this commit.** They land here only because
the shared-context spec requires the full contract surface in commit 01, ready for the QQ / KS work
in commits 02–04 to consume. The code comments say so explicitly; this is deliberate, not dead code
left by accident.

### The three increment-law samplers

Donsker needs only two moments, so the unit re-tests the same machinery against three qualitatively
different laws — a bounded 2-point law, a bounded continuous law, and a skewed unbounded law — so a
shape-specific normalization bug cannot hide behind the other two passing. Each is an iid, mean-0,
variance-1 closure of signature `f(rng, dims...) -> Array{Float64}`:

```julia
const increment_samplers = Dict{Symbol,Function}(
    :rademacher  => (rng, dims...) -> rand(rng, (-1.0, 1.0), dims...),                # P(±1)=1/2: mean 0, var 1
    :uniform     => (rng, dims...) -> sqrt(3.0) .* (2.0 .* rand(rng, dims...) .- 1.0), # U(-√3,√3): mean 0, var 1
    :exponential => (rng, dims...) -> (-log.(1.0 .- rand(rng, dims...))) .- 1.0,       # Exp(1)-1: mean 0, var 1, skew 2
)
```

The exponential is the one skewed law (skewness 2), built by inverse-CDF sampling and shifted by
`-1` to zero its mean. The uniform's `sqrt(3)` scale is exactly what makes a `U(-√3, √3)` variable
have variance 1 (`(2√3)^2/12 = 1`) — and, as GATE 01a's mutation test below shows, it is precisely
this factor the gate is designed to defend.

### `LAW_ORDER` — a reproducibility guardrail, not a cosmetic choice

```julia
const LAW_ORDER = (:rademacher, :uniform, :exponential)
```

The per-law loop threads **one** `StableRNG` stream across all three laws. Julia's `Dict` iteration
order is not stable across versions, so iterating `increment_samplers` directly would silently
reorder the draws pulled from that shared stream — changing every committed number and figure. The
pinned tuple fixes the order. (Commit 04 adds its `:pareto` law in its own block, not by extending
this tuple, so this stream stays byte-stable.)

### `rescaled_walk` — the lattice builder

```julia
function rescaled_walk(sampler::Function, n::Int, N::Int, rng)
    increments = sampler(rng, n, N)                    # n × N iid mean-0 var-1 increments
    S = vcat(zeros(1, N), cumsum(increments; dims = 1)) # (n+1) × N; row 1 is S_0 = 0
    return S ./ sqrt(n)
end
```

Returns an `(n+1) × N` matrix, **one path per column** (the repo-wide `n_grid × N` convention), so
the result feeds `empirical_cov` directly with no transpose. Column `j` is
`[S_0, S_1, …, S_n]/sqrt(n)` with `S_0 = 0`. Orientation matters: a transpose would silently estimate
the wrong matrix (`empirical_cov` returns `n_grid × n_grid`, not `N × N`).

---

## The Donsker picture

![Donsker picture: one rescaled Rademacher walk at each of n = 4, 16, 64, 256, 1024, overlaid on
[0,1], the lattice paths visually tightening toward a continuous limit as n grows](../../../experiments/05_bm_scaling_limit/figures/rescaled_paths.png)

One Rademacher walk at each lattice fineness `n ∈ {4, 16, 64, 256, 1024}`, all confined to `[0,1]`,
drawn from an **own small StableRNG stream** (seed 20260722) fully independent of the gate's stream
so it can never perturb the gate numbers. The `n = 4` path is a coarse 4-segment zig-zag; by
`n = 1024` the path is a visibly continuous, BM-like trace. This is the qualitative content of
Donsker's theorem, shown ahead of any quantitative gate. The lattice values are plotted directly and
connected by straight lines — the same linear interpolation Donsker uses, but never materialized as
its own array.

---

## GATE 01a — covariance → min(s,t), per law

The gate subsamples `M_SUB = 8` interior lattice rows out of `N_LATTICE = 64` (excluding the trivial
`t = 0` row, where the covariance is identically zero), builds the analytic target
`M[i,j] = min(t_i, t_j)` on those eight lattice times, and reuses **`empirical_cov`
(`src/gaussianprocess.jl`)** rather than hand-rolling a covariance. For each law it checks that the
Frobenius error `‖Ĉ_N − M‖_F` decays like `N^(-1/2)` as the ensemble size `N` grows.

![GATE 01a evidence: left, a log-log plot of mean Frobenius error vs N for all three laws with a
dashed -1/2 reference line, all three curves tracking it; right, a heatmap of |Ĉ_N − M| for the
Rademacher law at N=10000, values everywhere below 0.01](../../../experiments/05_bm_scaling_limit/figures/covariance_vs_min.png)

The left panel shows all three laws' mean error (over 20 replicates) tracking the dashed `-1/2`
reference across two decades of `N`. The right panel is a fresh representative Rademacher draw at the
top rung `N = 10000`: the residual `|Ĉ_N − M|` is everywhere below ~0.01, i.e. visually near-zero —
a direct look at the matrix the log-log slope is summarizing. The heatmap is purely illustrative and
is not part of any gate.

### Why there is no n-limit to gate — and what the gate therefore actually tests

Because `Cov(S_i/sqrt(n), S_j/sqrt(n)) = min(t_i, t_j)` holds **exactly for every n**, there is no
convergence-in-n to check. The only thing that can be wrong is whether the builder produces
correctly-normalized, independent increments. And covariance is **mean-invariant** —
`Cov(X, Y) = Cov(X - EX, Y - EY)` always — while `empirical_cov` demeans its input before computing.
So:

> **GATE 01a is purely a variance-normalization check.** It verifies each law's increments have
> variance 1 (up to the Monte-Carlo `N^(-1/2)` estimation error). It is, by construction, blind to
> the increment *mean*.

Running it **once per law** is what gives it bite: a wrong variance scale in any single law — the
canonical example being the uniform law's `sqrt(3)` factor — is caught here, at the foundation,
rather than corrupting a downstream figure silently.

---

## The gate-design war story — three iterations to an honest SE

The mathematical target is trivial; the **statistics of gating it** were not. The repo convention is
to gate a stochastic slope against a small multiple (2.5–3×) of *its own* standard error, never a
fixed absolute tolerance. Getting that SE to honestly reflect the slope's true noise took three
designs. Two failed empirically before the third landed — this is a real methodological finding from
building this commit, worth recording so it is not re-derived.

**Attempt 1 — nested prefixes of one path pool (failed: understates noise).** This literally mirrored
Unit 4's row-nested T-ladder pattern: build one big pool of paths per law, read the ladder off nested
prefixes, and split the pool into blocks for a batch SE. It under-estimated the true noise, because
the point estimate and one block's estimate were **identical by construction** (block 1 *was* the
prefix range). Every law's reported slope was thus as noisy as a single realization, while the batch
SE from the block split did not reflect that. Rademacher failed outright: slope `-0.25` against a
target `-0.5`, when the batch SE claimed it should land within `0.20`.

**Attempt 2 — independent per-rung draws, but a scale-mismatched SE (failed: too tight).** Here each
rung drew independently, but the "point estimate" was computed as its *own single* independent draw
(the size of one replicate), while being gated against the SE **of the mean** over `NGROUP`
replicates. That is a scale mismatch: a single-draw point estimate compared against the SE of an
*average*. Direct empirical check confirmed the mismatch: a 100-repeat average showed the true
asymptotic slope is solidly `-0.5` (`mean_err·sqrt(N) ≈ 4.7–5.1` stably across `N = 100..32000`), but
any *single* realization has relative SD ~55% at `N = 100` — there are only 36 independent
upper-triangular entries in the `8×8` target to average over — so single-draw noise routinely blows
past a tight batch-mean SE.

**Attempt 3 — classical batch means (validated).** Draw `NGROUP = 20` fully independent replicate
N-ladders. Each replicate draws **fresh, independent columns at every rung** of
`N_LADDER = [100, 320, 1000, 3200, 10000]` — the exact `00_covariance_core` method, just repeated 20
times on one continuing `StableRNG` stream — and is fit to one slope. Then report:

```
slope = mean of the NGROUP replicate slopes          (the low-noise point estimate)
SE    = std(replicate slopes) / sqrt(NGROUP)          (the honest SE of that mean)
gate  = |slope + 0.5| < 2.5 · SE
```

This is **internally consistent**: the quantity gated (a mean over `NGROUP` replicates) and the SE
(the SE of that same mean) are the same statistical object. It is the direct analogue of Unit 4's
group-slope SE, adapted to an axis — sample count `N` — where "more data" means more independent
*replicates* rather than a longer nested *time* axis. That distinction is exactly why the nested
prefix pattern works in Unit 4 (time and ensemble-column-count are orthogonal) but fails here (the
prefix pool and the point estimate collapse onto each other). `ols_slope_se`'s residual SE is still
printed per replicate-mean curve, but only **for contrast** — a single fit's residual SE cannot see
the replicate-to-replicate scatter that batch means measures directly.

Validated across four seeds: the z-scores `|slope + 0.5| / SE` cluster in `0–2.7`, consistent with
the `2.5–3×` multiple the repo convention specifies.

---

## Mutation-gate evidence

The gate's discriminating power was checked directly, on throwaway copies of the script (ad hoc, not
committed):

- **Wrong variance scale — caught decisively.** Mutating the uniform law's scale from `sqrt(3)` to
  `sqrt(2)` made the slope jump from `-0.56` to `+0.01` — a huge, unmissable FAIL
  (`|slope + 0.5| = 0.51` against a threshold of `0.03`). This is the class of bug GATE 01a exists to
  catch, and it catches it hard.
- **Missing mean-shift — NOT caught, and provably so.** Removing the exponential's `-1` shift (using
  `Exp(1)` instead of `Exp(1) - 1`) slipped through — **necessarily**, because `empirical_cov`
  demeans and covariance is mean-invariant. This is not a hole in the gate; it is the precise, correct
  scope of a covariance-based check. Such a bug is nowhere near silent, though: it makes the walk
  **drift like sqrt(n)** instead of settling down (immediately visible in a paths figure) and fails
  commit 02's marginal-distribution KS check outright (a mean-shift breaks convergence to `N(0, t)`).

This second finding is what drove the code-comment corrections noted under Deviations.

### Final gate numbers

Seeds `seed_paths = 20260722`, `seed_cov = 20260723`; `n_lattice = 64`, `m_sub = 8`,
`N_ladder = [100, 320, 1000, 3200, 10000]`, `ngroup = 20`, `se_mult = 2.5`. Runtime ~4 s, fully
deterministic given the seeds.

```
GATE 01a [rademacher ] slope: -0.5261;  |slope+0.5| = 0.0261  vs  2.5*SE = 0.0654  -> PASS
GATE 01a [uniform    ] slope: -0.5585;  |slope+0.5| = 0.0585  vs  2.5*SE = 0.0830  -> PASS
GATE 01a [exponential] slope: -0.5163;  |slope+0.5| = 0.0163  vs  2.5*SE = 0.0730  -> PASS
GATE 01a (all laws) -> PASS
ALL GATES: PASS
```

---

## Empirical / end-to-end verification

- Ran `run.jl` repeatedly: deterministic — identical numbers every run under the fixed seeds.
- Inspected both saved PNGs directly. An initial render clipped the Donsker-picture title at the left
  edge; fixed by adding explicit `size` / `margin` keywords to the `Plots` calls (visible in the
  file: `size = (800, 480)` and `*_margin` keywords throughout). Titles, legends, and axis labels are
  all fully visible in the committed figures.
- `julia --project -e 'using Pkg; Pkg.test()'` still green — **177/177 passing, unchanged**. This
  commit touches no `src/` or `test/` file, so the suite is a regression sanity check only; the gate
  above is this commit's real verification.

---

## Code review

`/code-review` is not programmatically invokable in this environment (it errors with
`disable-model-invocation`), so an equally rigorous manual self-review was substituted: the full diff
was re-read line by line; the three copied helpers were verified byte-for-byte against their sources;
`rescaled_walk`'s orientation and docstring were checked against the shared-context spec; constant
naming/alignment was checked; and — most substantively — the two mutation tests above were run to
confirm the gate's true discriminating power, which is what surfaced the comment corrections. No other
findings; nothing was declined.

---

## Deviations from plan

1. **Ladder and replicate counts were derived empirically, not given.** The plan said "derive
   theory-first" without pinning `N_LADDER` or `NGROUP`. The final values
   (`[100, 320, 1000, 3200, 10000]`, `NGROUP = 20`) came out of the three-design iteration above,
   mirroring Unit 0's proven 2-decade geometric ladder rather than the plan's suggested nested-prefix
   pattern — which was tried first and found to understate the true noise on this sample-count axis.
2. **A plan-text inaccuracy about GATE 01a's reach was corrected in the code.** The plan's rationale
   claimed GATE 01a "catches a forgotten exponential `-1` shift." It cannot — a covariance-based gate
   is mean-invariant by construction, as the mutation test confirmed. The `run.jl` banner and the
   increment-samplers comment, which had initially echoed the plan's language, now state the accurate
   scope: GATE 01a tests **variance** normalization; the **mean** is caught downstream by commit 02's
   marginal-distribution check. This is a plan-level inaccuracy, not an implementation defect.
</content>
</invoke>

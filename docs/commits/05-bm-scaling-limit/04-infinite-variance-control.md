# Unit 5 · Commit 04 — The infinite-variance falsifier (negative control)

## TL;DR

Commits 01–03 verified Donsker's invariance principle for three increment laws that all share the
**same two hypotheses**: mean 0 and *finite* variance (normalized to 1). Every gate confirmed the
`n^(-1/2)` picture — the marginal converges to `N(0,1)`, its variance sits at 1, the running-max
functional lands on the half-normal. This commit closes the unit with the mirror-image question:
**what breaks if we drop the finite-variance hypothesis?** It feeds the *same* `rescaled_walk`
machinery a symmetric, mean-0, but **genuinely infinite-variance** increment and shows the picture
fails — not vaguely "worse," but in the *opposite* direction of every earlier gate.

The increment is a **symmetric Pareto**: `sign · U^(-1/γ)` with `U ~ Uniform(0,1)` drawn
**untruncated** and tail index `γ = 1.5 ∈ (1, 2)`. Because `γ > 1` the law has an honest mean of
exactly 0 (via odd symmetry), but because `γ ≤ 2` its second moment `E[X²]` genuinely **diverges** —
this is a provably infinite-variance law, not merely a large-variance one.

Two contrasting gates pin the failure against commit 02's finite-variance baseline, both on the
ladder `n = [50, 100, 200, 400, 800]`, `N = 2000`, `NGROUP = 20` batch-means groups:

- **Gate 04a (variance grows):** the `Var(S_n/√n)`-vs-`n` batch-means slope must be **positive**
  (`> +0.10`). Under a finite-variance law that quantity is *identically 1* for every `n` (slope 0);
  under the Pareto it *grows* — the scale of `S_n/√n` diverges as `n^(1/γ − 1/2) = n^(1/3)`.
- **Gate 04b (no Gaussian limit):** the `KS(S_n/√n, Φ)`-vs-`n` slope must **not** decay like the
  finite-variance laws (`> −0.35`, i.e. flat-to-positive) — the mirror image of commit 02's clean
  `≈ −0.50` decay.

The **load-bearing correctness point**: the Pareto tail must be left **untruncated**. A bug that
clamps `U` away from 0 or caps the magnitude would silently restore *finite* variance and defeat the
falsifier — and Gate 04a is exactly what catches that (a secretly-finite variance would settle,
slope → 0, and 04a would fail). This is why the falsifier is gated on "variance *grows*," not merely
on "KS is large."

The change is **experiment-only**: ~175 lines appended to the *same* file,
`experiments/05_bm_scaling_limit/run.jl` (the Phase 4 block, from the falsifier banner at line 660 to
EOF), plus one committed figure `figures/infinite_variance_control.png`. No `src/` or `test/` code —
the suite stays green at **177/177** (a pure regression check, since nothing under test changed).

For the shared conventions this commit inherits (RNG rules, the batch-means SE pattern, the
`n_grid × N` orientation, `rescaled_walk`, `ols_slope_se`, `ks_ladder_replicate`, `normcdf`), see the
sibling docs `01-foundation-and-covariance.md` and `02-ks-rate.md` — not restated here.

---

## Background — why infinite variance breaks Donsker, and in which direction

**Donsker needs finite variance.** The invariance principle `W_n → B` requires iid increments with
mean 0 *and finite variance*; the variance-1 normalization merely fixes the scale, but the
*finiteness* is load-bearing. Drop it and the classical `n^(-1/2)` scaling is the wrong normalization
altogether — a sum of heavy-tailed increments converges (after the *right* rescaling) to a
**stable** law, not Brownian motion.

**The symmetric-Pareto increment.** `U^(-1/γ)` with `U ~ Uniform(0,1)` is a standard Pareto variable
on `[1, ∞)` with the textbook power-law tail

```
P(U^(-1/γ) > x) = P(U < x^(-γ)) = x^(-γ),   x ≥ 1.
```

An independent Rademacher sign `±1` symmetrizes it around 0. The tail index `γ = 1.5` sits in the
critical window `(1, 2)`:

- **Mean is exactly 0** — `γ > 1` makes `E|X|` finite, and odd symmetry makes the mean *exactly* 0
  (by construction, not by empirical de-meaning). So this is *not* a mean-shift artifact; it fails
  Donsker only through the second moment.
- **Variance is genuinely infinite** — `E[X²] = γ ∫₁^∞ x^(1−γ) dx`. At `γ = 1.5` the exponent
  `1 − γ = −0.5`, so the integrand is `x^(-0.5)` and `∫ x^(-0.5) dx` diverges at the *upper* limit.
  This is a provably infinite second moment, not a numerically-large-but-finite one.

**Which direction does it break?** Both gates test a *direction*, not a rate-within-SE:

1. **The scale diverges (Gate 04a).** Write `S_n/√n = n^(1/γ − 1/2) · Y_n`, where `Y_n` converges in
   law to a `γ`-stable variable of `O(1)` typical scale (the stable central limit theorem for
   regularly-varying tails). Since `γ = 1.5 < 2`, the exponent `1/γ − 1/2 = 1/3 > 0` is *strictly
   positive*, so the scale of `S_n/√n` **grows** with `n` — the exact opposite of commits 01–03,
   where the analogous quantity is *identically 1* for every `n`. The finite-`N` *sample* variance of
   `S_n/√n` inherits that growth in expectation (heavy-tailed sums are dominated by the "one big
   jump," so the sample variance tracks the largest draw), giving a positive `Var`-vs-`n` slope.

2. **No Gaussian limit (Gate 04b).** Because the scale of `S_n/√n` diverges, at any fixed `x` the
   marginal CDF `F_n(x) = P(S_n/√n ≤ x) → 1/2` as `n → ∞` (a diverging-scale symmetric law puts
   vanishing mass in any fixed finite window). So `KS(S_n/√n, Φ)` trends *toward* the same ~0.5
   "totally uninformative" ceiling seen in commit 03's wrong-target control — it does **not** decay
   like commit 02's finite-variance `n^(-1/2)`.

---

## What changed

One file, appended; one new figure. No library or test changes.

- **`experiments/05_bm_scaling_limit/run.jl`** — ~175 lines appended (Phase 4 block, from the
  falsifier banner at line 660 to EOF). Structure: the `:pareto` sampler added to
  `increment_samplers` (its own block) → the control constants (`SEED_CTRL`, ladder, `N_MC_CTRL`,
  `NGROUP_CTRL`) → Gate 04a (`var_ladder_replicate` + batch-means) → Gate 04b (reusing
  `ks_ladder_replicate`) → the figure → the recorded-parameters line → the updated `ALL GATES` line
  (now `gate_01a && gate_02 && gate_03 && gate_04`).
- **`experiments/05_bm_scaling_limit/figures/infinite_variance_control.png`** — a two-panel figure:
  the variance-growth curve (Gate 04a) and a heavy-tailed QQ departure.

### The Pareto sampler — added in its OWN block, not by extending `LAW_ORDER`

```julia
increment_samplers[:pareto] = (rng, dims...) -> begin
    U   = rand(rng, dims...)                # Uniform(0,1), the FULL open support -- no clamping
    mag = U .^ (-1.0 / GAMMA_PARETO)         # Pareto(scale=1, shape=γ) on [1,∞): P(mag>x)=x^(-γ)
    sgn = rand(rng, (-1.0, 1.0), dims...)    # independent Rademacher sign -> symmetric about 0
    sgn .* mag
end
```

`:pareto` is added to the `increment_samplers` Dict so it is reachable by the *same* name-based
lookup as the other three laws — but it is **deliberately not** appended to `LAW_ORDER`. `LAW_ORDER`
pins the draw order of a *shared* `StableRNG` stream across commits 01–03's finite-variance loops;
appending to it would silently reorder those draws and change every already-committed number in the
file (a CLAUDE.md non-negotiable). Mutating the Dict is fine — `const` pins the Dict *object*, not
its contents — and the new key is only ever used from this phase's own code, with its own
`SEED_CTRL` stream, so nothing earlier is perturbed.

### The critical correctness guard — untruncated tail

The single most important line is the *absence* of any clamp:

```julia
U = rand(rng, dims...)   # the FULL open support -- no clamping of U or of mag
```

A "numerical safety" bug — clamping `U` to `(eps, 1)`, or capping `mag` at some large finite value —
would cap the Pareto tail away from infinity and **silently restore a finite variance**, defeating
the whole falsifier. Nothing here clips `U` or `mag` in any way, and Gate 04a is the structural
check that this stays true (see below).

### Gate 04a's per-replicate statistic

```julia
function var_ladder_replicate(sampler::Function, n_ladder, N, rng)
    vars = [empirical_cov(reshape(rescaled_walk(sampler, n, N, rng)[end, :], 1, :))[1, 1]
            for n in n_ladder]
    slope, _ = ols_slope_se(log10.(n_ladder), log10.(vars))
    return slope, vars
end
```

Per rung `n`, it takes the endpoint row `W[end, :]` (the `N` values of `S_n/√n`), reshapes it into a
`1 × N` "single-time-point path matrix," and reuses the library's `empirical_cov` as a variance
estimator rather than hand-rolling one. It then fits a per-replicate log-log slope of `Var` vs `n`.
This is exactly the `cov_ladder_replicate` pattern from Gate 01a, on a different per-rung statistic.

**Why batch-means, not a single ladder fit.** The *raw* sample variance at a single replicate is
wildly noisy — dominated by whether that replicate happened to catch a huge outlier (observed to
range over several *orders of magnitude* between adjacent rungs). Averaging the raw variances does
not help (the arithmetic mean of an infinite-variance quantity is itself unstable). Averaging the
per-replicate *slope* over `NGROUP_CTRL = 20` groups is well-behaved, because a log-log slope is far
less sensitive to a single large outlier than the raw magnitude is.

### Gate 04b reuses commit 02's estimator verbatim

Gate 04b calls `ks_ladder_replicate` (defined in Phase 2) unchanged — same estimator, same
batch-means machinery, same `n`-ladder — so the contrast with commit 02 is apples-to-apples: nothing
changed but the increment law.

---

## The gate design — two contrasting falsifier gates

GATE 04 passes only if **both** halves pass — a falsifier must fail in *both* the ways theory
predicts, so a single accidental pass cannot rescue it.

- **Gate 04a — variance grows.** Batch-means slope of `Var(S_n/√n)` vs `n` must exceed
  `VAR_SLOPE_MARGIN = +0.10`. This is a **fixed-direction** control margin (à la Unit 4's
  `slope_c > −0.75` and commit 03's absolute-bound gates), *not* a rate gated against theory-within-
  SE. It only needs "clearly, robustly positive." `0.10` sits with real headroom below both the
  analytic exponent `2/γ − 1 = 0.333` and the observed batch-mean slope (`~0.42`). A finite-variance
  law under the same machinery would give slope ≈ 0 and fail here — which is precisely how this gate
  catches a truncation bug.

- **Gate 04b — no Gaussian limit.** Slope of `KS(S_n/√n, Φ)` vs `n` must exceed
  `KS_CTRL_SLOPE_FLOOR = −0.35`. Commit 02's two `n^(-1/2)` laws both sat *at* `−0.5` within a small
  SE-multiple; anything above `−0.35` is decisively **not** that regime. The observed slope here is
  `~+0.06` with a tiny batch SE (`~0.002`) — flat-to-positive, the mirror image of commit 02's clean
  `−1/2` decay. (The KS-vs-`n` curve is far more stable run-to-run than the raw-variance curve, since
  KS is a bounded, non-heavy-tailed statistic.)

### Why 04a is gated on direction, not the exact exponent

The measured 04a slope (`+0.42`) is **shallower than the `+2/3` theory** (`2/γ − 1 = 0.667` for the
*expected* sample variance). That gap is expected and benign: the sample variance of an
infinite-variance quantity is a **downward-biased, high-variance estimator** — dominated by rare
extremes, so a finite `N` systematically under-captures the true (infinite) second moment and lands
below the theoretical growth rate. Gating against the exact `2/3` exponent would be gating a noisy,
biased estimator against a rate it cannot reliably hit. The point of the falsifier is not the
*value* of the slope but its *sign*: it is decisively positive, which is impossible for a
finite-variance law. Batch-means over 20 groups stabilizes the growth-rate estimate enough to make
"positive with margin" a robust call.

---

## Figures

### `infinite_variance_control.png` — variance growth + heavy-tailed QQ departure

![Two panels. Left: log-log plot of mean Var(S_n/√n) over 20 replicates vs walk length n=[50..800] for
the Pareto (γ=1.5) increment — a purple curve rising from ~850 at n=50 to ~2e4 at n=800 with
batch-mean slope 0.423, tracking a dashed reference line of slope 2/γ-1=0.333, and sitting far above
the gray dotted "finite-variance target Var≡1" line at the bottom. Right: a QQ plot of the
standardized endpoint at n=800 against N(0,1) quantiles, the empirical points forming a steep S-curve
that departs sharply from the dashed y=x Gaussian reference in both tails (reaching ±60 at the ±2.5
theoretical quantiles), the signature of heavy tails.](../../../experiments/05_bm_scaling_limit/figures/infinite_variance_control.png)

**Left panel (Gate 04a).** Mean `Var(S_n/√n)` (over the 20 replicate groups) vs `n`, log-log. The
purple Pareto curve climbs from `~850` at `n = 50` to `~2×10⁴` at `n = 800` — the diverging scale
made visible — tracking the dashed `2/γ − 1 = 0.333` reference and standing *orders of magnitude*
above the gray dotted `Var ≡ 1` line that every finite-variance law (commits 01–03) would hug. The
legend records the fitted batch-mean slope `0.423`. The last rung jumps well above the reference
line, the "one big jump" of a heavy tail catching an extreme draw — a visual reminder of why the raw
variance is noisy and the gate is on the *slope*, not the level.

**Right panel (no Gaussian limit).** A QQ plot of the standardized endpoint at the largest rung
`n = 800` against `N(0,1)` quantiles. The empirical points trace a steep S-curve that departs sharply
from the dashed `y = x` Gaussian reference in *both* tails (reaching `±60` at the `±2.5` theoretical
quantiles) — the unmistakable signature of heavy tails, and the marginal counterpart of Gate 04b's
"KS does not decay." The QQ panel uses its own deterministic draw from `StableRNG(SEED_CTRL + 1)` — a
distinct stream, decoupled from the gate numbers, matching Phase 2's `marginal_qq.png` convention —
and picks the largest rung because the diverging scale has had the most room to separate from a fixed
Φ there.

The figure was inspected directly (not merely rendered) — titles, axes, legends fully visible,
nothing clipped.

---

## Verified gate output

```
GATE 04a [pareto     ] Var(S_n/√n)-vs-n slope: 0.4235  (batch SE 0.1414)  vs margin +0.10 -> PASS
GATE 04b [pareto     ] KS(S_n/√n,Φ)-vs-n slope: 0.0607  (batch SE 0.0023)  vs floor -0.35 -> PASS   (contrast: commit 02's laws ~ -0.50)
GATE 04 (variance grows AND no Gaussian limit) -> PASS
ALL GATES: PASS
```

Gate 04a's slope `0.4235` clears the `+0.10` margin with room to spare (and sits below the `+0.667`
theory exactly as the downward-biased sample-variance estimator predicts); its batch SE `0.1414`
reflects the residual heavy-tail noise the batch-means averaging tames but cannot erase. Gate 04b's
slope `0.0607` is flat-to-positive with a tiny SE `0.0023`, decisively separated from commit 02's
`≈ −0.50` regime by the `−0.35` floor. The falsifier fails in *both* predicted directions, so GATE 04
passes — meaning the finite-variance hypothesis is confirmed load-bearing.

`julia --project -e 'using Pkg; Pkg.test()'` stays green at **177/177** (unchanged — this commit
touches no `src/` or `test/`, so the suite is a pure regression sanity check; the gates above are this
commit's real verification).

---

## Code review

`/code-review` is not programmatically invokable in this environment (it errors with
`disable-model-invocation`), so a rigorous manual self-review was substituted — the same substitution
commits 01–03 used. The full diff was re-read line by line. Findings and disposition:

- **Verified the tail is untruncated** — the single most important correctness property. `U =
  rand(rng, dims...)` uses the full open support; neither `U` nor `mag` is clamped or capped
  anywhere. This is the property Gate 04a structurally protects.
- **Verified `:pareto` does not extend `LAW_ORDER`** — it is added only to the `increment_samplers`
  Dict (mutation of contents, not the `const` binding) and used solely from this phase's own code, so
  no already-committed number from commits 01–03 shifts.
- **Verified the control stream is independent** — `SEED_CTRL = 20260725` and its `+1` QQ derivative
  are fresh streams, drawn after every earlier gate's draws, so they cannot perturb any prior number.
- **Verified the gate is a conjunction** — `gate_04 = gate_04a && gate_04b`, and the top-level
  `ALL GATES` line now ANDs `gate_04` in, so a half-failure fails the unit.
- **Verified `empirical_cov` reuse** — reshaping the endpoint row to `1 × N` and reading `[1,1]` is a
  legitimate variance estimator via the library's own covariance routine (respecting the `n_grid × N`
  orientation), not a hand-rolled formula.
- **Confirmed** no `src/` or `test/` file is touched, and `Pkg.test()` remains 177/177.

No findings survived.

---

## Deviations from plan

**None material.** The plan's design — two contrasting falsifier gates, a symmetric-Pareto increment
with `γ < 2`, added in its own block outside `LAW_ORDER` — is realized as written. The one point worth
recording is a *deliberate* gating choice, not a departure: Gate 04a's slope target is gated as
**"positive with margin" (`> +0.10`)** rather than pinned to the `+2/3` theoretical exponent, because
the infinite-variance sample-variance estimator is too noisy and downward-biased to gate against the
exact rate. This matches the falsifier's intent exactly — the *sign* of the slope, not its value, is
what distinguishes infinite from finite variance.

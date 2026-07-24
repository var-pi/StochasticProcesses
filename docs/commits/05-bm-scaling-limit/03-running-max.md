# Unit 5 · Commit 03 — The running-maximum path functional → half-normal

## TL;DR

Commit 02 verified Donsker at the level of the **one-time marginal** `W_n(1) = S_n/sqrt(n) → N(0,1)`.
But Donsker is a statement about the **whole path**, not just its endpoint — so a functional that
depends on the path's *shape* should converge too. This commit adds that check with the natural next
functional: the **running maximum** `M^(n) = sup_{[0,1]} W_n`. By the reflection principle,
`sup_{[0,1]} B =_d |B_1| =_d |N(0,1)|`, the **half-normal** law with CDF `2·Φ(x) − 1` for `x ≥ 0`.

The change is **experiment-only**: ~140 lines appended to the *same* file,
`experiments/05_bm_scaling_limit/run.jl` (from the Phase 3 banner at line 521 to EOF), plus one
committed figure `figures/running_max.png`. No `src/` or `test/` code — the suite stays green at
**177/177** (a pure regression check, since nothing under test changed).

Unlike commit 02's headline *rate* gate, GATE 03 is a **consistency + wrong-target-control** gate at a
single fixed `n` per law: `KS(M, half-normal)` must be **small** (below an absolute bound) **and**
`KS(M, Φ)` — the deliberately *wrong*, full-normal target — must be **large**. The first passes only
if the functional converged to the right shape; the second closes the loophole where "KS is small"
could mean "converged to *some* law that happens to sit near both."

The engineering weight here is a **threshold mis-design that failed on first pass and taught a real
lesson**: the finite-`n` deviation of `KS(M, half-normal)` is a genuine `~c/sqrt(n)` convergence term
(a max-functional analogue of commit 02's Edgeworth corrections), **not** the ~0-plus-sampling-noise
that the first threshold budgeted for. That war story is the most valuable thing in this doc and gets
the most space.

For the shared conventions this commit inherits (RNG rules, the `n_grid × N` orientation, `normcdf`,
`ks_statistic`, `rescaled_walk`, `LAW_ORDER` / `increment_samplers` / `colors_by_law`), see the sibling
doc `docs/commits/05-bm-scaling-limit/01-foundation-and-covariance.md` — not restated here.

---

## Background — the reflection principle and why running max is exact on the lattice

**The limit law.** For standard Brownian motion `B` on `[0, 1]`, the reflection principle gives the
running supremum's law in closed form:

```
sup_{[0,1]} B  =_d  |B_1|  =_d  |N(0,1)|
```

This is the **half-normal** distribution. Its CDF, defined here as `halfnormcdf`, is

```
halfnormcdf(x) = 2·Φ(x) − 1   for x ≥ 0,   and 0 for x < 0
```

(`Φ` is the standard-normal CDF, the `normcdf` from commit 01.) The `x < 0` branch is exactly 0: a
path started at 0 has a running max that includes `S_0 = 0`, so `M ≥ 0` always, at finite `n` and in
the limit.

**Why the discrete running max is the *exact* continuous supremum.** `rescaled_walk` returns the
`(n+1) × N` lattice `W`, one path per column, and the block takes

```julia
M = vec(maximum(W; dims = 1))   # running max per path (per column) — one scalar per sample
```

This is a subtle correctness point worth stating: the max over the `n+1` lattice rows **is** the true
continuous-path supremum, no fine grid needed. Donsker's limit path is the *piecewise-linear
interpolation* of the lattice points; a straight segment attains its maximum at one of its two
endpoints, so no excursion can hide *between* lattice points. (This is what makes the running max
cheaper to check than, say, a local-time or occupation functional, which genuinely would need
sub-lattice resolution.)

**Why this is a consistency check, not a rate.** Commit 02 asked *how fast* `KS → 0` and gated the
log-log slope. Here the claim is different and simpler: *at one large fixed `n`, the running-max law
already looks like the half-normal — specifically, and not merely "some limit."* So the gate is a
single-`n` bound, not a slope over a ladder.

---

## What changed

One file, appended; one new figure. No library or test changes.

- **`experiments/05_bm_scaling_limit/run.jl`** — ~140 lines appended (Phase 3 block, lines 521–EOF).
  Structure: the `halfnormcdf` helper → the per-law consistency loop computing `KS(M, half-normal)`
  and the `KS(M, Φ)` control → the figure → the recorded-parameters line → the updated
  `ALL GATES` line (now `gate_01a && gate_02 && gate_03`).
- **`experiments/05_bm_scaling_limit/figures/running_max.png`** — empirical CDFs of `M^(n)` per law
  overlaid on the half-normal target (solid) and the wrong-target `Φ` (dashed).

### The new helper and the loop core

```julia
halfnormcdf(x) = x >= 0 ? 2 * normcdf(x) - 1 : 0.0
```

The loop threads **one** `StableRNG(SEED_MAX)` stream across `LAW_ORDER` in its pinned order (the
repo's standard shared-stream discipline — reordering the loop would silently change every committed
number), and per law:

```julia
W = rescaled_walk(sampler, n_steps, N_mc, rng_max)
M = vec(maximum(W; dims = 1))            # running max per column
ks_half  = ks_statistic(M, halfnormcdf)  # correct-target KS: should be small
ks_wrong = ks_statistic(M, normcdf)      # wrong-target control: should be large (~0.5)
gate_law = (ks_half < MAX_KS_BOUND) && (ks_wrong > MAX_WRONG_MIN)
```

An `@assert all(law -> haskey(N_STEPS_MAX, law) && haskey(N_MC_MAX, law), LAW_ORDER)` guards the
per-law Dicts so a missing law fails loudly rather than `KeyError`-ing (or silently skipping) inside
the loop.

### Per-law `n` and `N` — sized independently, not shared

`n` is a **per-law Dict**, not a shared scalar:

```julia
const N_STEPS_MAX = Dict(:rademacher => 3_000, :uniform => 1_200, :exponential => 1_200)
const N_MC_MAX    = Dict(:rademacher => 30_000, :uniform => 50_000, :exponential => 50_000)
```

Rademacher's running max is **lattice-valued** (only `n+1` possible values), so `KS(M, half-normal)`
carries the same `O(n^(-1/2))` discreteness term (`~0.4/sqrt(n)`) seen in commit 02's Rademacher
marginal, layered **on top** of the shared `~c/sqrt(n)` deviation. It therefore needs a larger `n`
(3000 vs 1200) for its true deviation to land under the same bound. `N` is kept **modest** — large
enough that single-draw sampling noise is a minor contributor, not "as large as memory allows" —
because (per the war story below) the deterministic deviation, not sampling, dominates the KS distance
at these `n`. Rademacher's `N` is capped lower (30k vs 50k) to keep its materialized `(n+1) × N`
lattice to a few hundred MB.

---

## The gate design — consistency bound + wrong-target control

GATE 03 runs for all three laws in `LAW_ORDER` at their fixed `n`, and each law must clear **both**
halves:

- **Correct-target bound** — `KS(M, half-normal) < MAX_KS_BOUND = 0.05`. This budgets the functional's
  own deterministic finite-`n` deviation (empirically ~0.017–0.024 at the chosen `n`) with real
  margin. A FAIL here means a genuinely mis-scaled or wrongly-shaped running max, not noise.
- **Wrong-target control** — `KS(M, Φ) > MAX_WRONG_MIN = 0.3`. The half-normal is emphatically **not**
  `Φ`: at `x = 0`, `Φ(0) = 0.5` while `halfnormcdf(0) = 0`, so the analytic sup-distance
  `sup|halfnormcdf − Φ|` approaches **0.5**, independent of `n` or `N`. The `0.3` threshold leaves
  enormous room (>6× the `0.05` correct-target bound). This half carries no real risk of its own; its
  job is to make the "specifically half-normal, not just *some* limit" claim a **checked fact** rather
  than an assumption.

The two together give a **>6× discrimination ratio** between the largest acceptable correct-target KS
and the smallest acceptable wrong-target KS — the gate cannot pass by converging to a garbage law that
merely sits near both targets.

### The war story — the threshold that budgeted the wrong quantity

The first threshold design **failed for all three laws on first pass**, and the reason is the
transferable lesson.

**The mis-design.** The initial gate was `KS(M, half-normal) < floor(N) + 0.01`, where `floor(N)` is
the classical Kolmogorov finite-`N` **sampling** floor `E[D_N] ~ 0.87/sqrt(N)`. This implicitly
assumed the true `KS(M, half-normal)` at finite `n` is `≈ 0` and that anything above zero is sampling
noise to be budgeted by the floor.

**Why it is wrong.** `KS(M, half-normal)` at finite `n` is **not** ~0-plus-noise. It is a genuine,
**deterministic** convergence deviation of size `~c/sqrt(n)` — the running max is itself only a
finite-sample approximation to `sup_{[0,1]} B`, exactly analogous to the Edgeworth corrections that
governed commit 02's marginal *rate*. For Rademacher there is additionally the lattice-discreteness
term `~0.4/sqrt(n)` stacked on top. This deviation surfaced first, painfully, at a much smaller `n`
(uniform/exponential at `n = 300`): there the true deviation `~0.033–0.046` came in **2.5–3.6× over**
a `floor(N) + margin` threshold of `~0.013–0.015`, and **every law failed**. The mistake was
conflating two different regimes — a **rate-scale** quantity (`~c/sqrt(n)`, an exact property of the
finite-`n` law) with a **fixed absolute sampling margin** (`~0.87/sqrt(N)`) — that differ by an order
of magnitude at these `N`.

**The fix.** Gate against a single **principled absolute bound** `MAX_KS_BOUND = 0.05`, sized to the
deterministic `~c/sqrt(n)` deviation at moderate `n`, and raise `n` (to {rademacher 3000, uniform
1200, exponential 1200}) so that deviation lands comfortably under it. This is deliberately **not**
"floor + margin" — there is no meaningful floor-shrinking effect to chase here (the deviation, not
sampling, dominates), so folding a floor term back in would just reintroduce the same conflation.

**The road *not* taken — a deliberate decision.** One could instead push `n` into the many thousands
to force the deterministic deviation *below* the sampling floor and recover a floor-budgeted gate. This
is deliberately avoided: for Rademacher the lattice term `~0.4/sqrt(n)` decays slowly, so driving it
under the floor needs a multi-GB `(n+1) × N` lattice for **no added rigor** — the discrimination
against the wrong target is already decisive by >6× at the modest `n` used. A **realistic absolute
margin plus a structural wrong-target control** is the correct design; chasing the sampling floor here
is a memory/compute trap.

### A crash fixed along the way

The recorded-parameters `@printf` originally passed the per-law Dicts through a `%d` conversion, which
crashes (a `Dict` is not an integer). Fixed to `%s` with `string(...)`:

```julia
@printf("\nrecorded: seed_max=%d, n_steps_max=%s, N_mc_max=%s, max_ks_bound=%.3f, max_wrong_min=%.2f\n",
        SEED_MAX, string(N_STEPS_MAX), string(N_MC_MAX), MAX_KS_BOUND, MAX_WRONG_MIN)
```

---

## Figures

### `running_max.png` — the discrimination made visible

![Empirical CDFs of the running max M^(n) for rademacher (n=3000, KS=0.0181), uniform (n=1200,
KS=0.0166), and exponential (n=1200, KS=0.0239), all three lying essentially on the solid black
half-normal target curve; the dashed black Phi (wrong-target) curve sits well above and to the left,
starting at 0.5 at x=0 while the half-normal starts at 0](../../../experiments/05_bm_scaling_limit/figures/running_max.png)

Empirical CDFs of `M^(n)` per law (colored, `LAW_ORDER` colors), overlaid on the **half-normal**
target (solid black, the correct limit) and **Φ** (dashed black, the wrong-target control). All three
empirical curves lie essentially *on top of* the half-normal — the KS values in the legend
(0.0181 / 0.0166 / 0.0239) are visually imperceptible. The dashed `Φ` control runs clearly separated:
it starts at `Φ(0) = 0.5` while the half-normal (and every empirical curve) starts at 0, the exact
structural gap the `KS(M, Φ) ≈ 0.5` control exploits. The plot makes the gate's whole logic visible —
the empirical laws hug the correct target and stand far from the wrong one, in a single frame.

The ECDF is subsampled (`stride = max(1, n_ms ÷ 400)`) for a legible plot, since `N_MC_MAX` points
would be far too dense to render. The figure was inspected directly (not merely rendered) — title,
axes, and legend fully visible, nothing clipped.

---

## Verified gate output

```
GATE 03 [rademacher ] n=3000    KS(M,half-normal)=0.01813  (bound 0.05000) -> PASS   |   KS(M,Φ)=0.5000  (> 0.30) -> PASS
GATE 03 [uniform    ] n=1200    KS(M,half-normal)=0.01662  (bound 0.05000) -> PASS   |   KS(M,Φ)=0.5000  (> 0.30) -> PASS
GATE 03 [exponential] n=1200    KS(M,half-normal)=0.02387  (bound 0.05000) -> PASS   |   KS(M,Φ)=0.5000  (> 0.30) -> PASS
GATE 03 (all laws) -> PASS
ALL GATES: PASS
```

Every law clears the correct-target bound with margin (0.018 / 0.017 / 0.024 vs the 0.05 bound), and
the wrong-target control pins at exactly 0.5000 for all three — the structural `Φ(0) − halfnormcdf(0)`
gap, independent of `n` and `N`, as predicted.

`julia --project -e 'using Pkg; Pkg.test()'` stays green at **177/177** (unchanged — this unit touches
no `src/` or `test/`, so the suite is a pure regression sanity check; the gate above is this commit's
real verification).

---

## Code review

`/code-review` is not programmatically invokable in this environment (it errors with
`disable-model-invocation`), so a rigorous manual self-review was substituted — the same substitution
commits 01 and 02 used. The full diff was re-read line by line. Findings and disposition:

- **Fixed a real crash:** the `@printf` recorded-parameters line passed the `N_STEPS_MAX` /
  `N_MC_MAX` Dicts through `%d`, which throws. Corrected to `%s` + `string(...)`. This is a genuine
  defect that would have aborted the run before printing the final gate line.
- **Threshold re-design (the war story above):** the `floor(N) + margin` budget was replaced with the
  principled absolute `MAX_KS_BOUND` after it failed all three laws. Documented in-code with a pointer
  to this doc.
- **Verified** the running-max-on-a-lattice equals the continuous supremum (piecewise-linear
  interpolation argument) — so `vec(maximum(W; dims = 1))` needs no sub-lattice grid.
- **Verified** the loop threads one shared `StableRNG(SEED_MAX)` across `LAW_ORDER` in pinned order,
  and that the `@assert` guards both per-law Dicts against a `KeyError` / silent skip.
- **Verified** no `src/` or `test/` file is touched (matches the unit's "everything lives in the
  experiment" design), and that `Pkg.test()` remains 177/177.

No other findings survived.

---

## Deviations from plan

1. **The gate threshold was corrected from the plan's sketch.** The plan described the correct-target
   threshold as "below the finite-N floor + a small margin." Implementation corrected that to a
   **principled absolute bound** (`MAX_KS_BOUND = 0.05`) budgeting the deterministic `~c/sqrt(n)`
   finite-`n` deviation — because the floor-only budget was an **order of magnitude too tight** and
   failed for all three laws on first pass. Same intent (consistency + wrong-target discrimination),
   sounder threshold.
2. **Per-law `n` sizing.** `n` is a per-law Dict (rademacher 3000, uniform/exponential 1200) rather
   than a shared scalar, so Rademacher's extra lattice-discreteness term lands under the same bound as
   the two continuous laws. This emerged from the threshold analysis, not the plan.
3. **A `%d`-on-a-Dict `@printf` crash** in the recorded-parameters line was found and fixed during
   implementation — not anticipated by the plan.

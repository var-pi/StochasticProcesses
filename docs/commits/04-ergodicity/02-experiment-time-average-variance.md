# Commit 2 — the ergodicity experiment (Unit 4 "ergodicity", feature `04-ergodicity`)

## TL;DR

Adds Unit 4's gallery experiment: the new script `experiments/04_ergodicity/run.jl`
plus the three committed PNG figures it writes under
`experiments/04_ergodicity/figures/` (`variance_vs_T.png`, `msd_vs_t.png`,
`running_average_band.png`). It consumes the `Ergodic` module shipped in commit 1
(`running_time_average`, `time_average_variance`, `mean_square_displacement`,
`green_kubo`, `time_average_variance_exact`). No `src/` or `test/` file is touched —
a pure gallery commit, nothing added to CI (Monte-Carlo experiments stay out of CI
per `CLAUDE.md`).

Unit 4's job is to **close the methodological loop** the whole project rests on. Units
0–3 each justified checking *one sampled path against an analytic result* — but that
methodology is only licensed if the time-average of one long path actually converges
to the ensemble mean (the ergodic / L²-law-of-large-numbers claim, Pavliotis Prop.
1.16). This experiment demonstrates that convergence and its rate on one seeded OU
ensemble, via **two stochastic gates** plus a diagnostic:

- **(a)** the ensemble variance `Var(A_T)` of the time-average decays as `T^{-1}` —
  fitted log-log slope lands on `-1`;
- **(b)** the *constant* in that decay is `2·D*` where `D* = D/α²` is the Green–Kubo
  transport coefficient (Pavliotis Example 1.18) — **not** `2·R(0) = 2D/α`. These
  two differ precisely because `α = 2 ≠ 1` here, so gate (b) genuinely pins `α²`
  rather than just `α`.

Both gates PASS with margin. **No deviations from the plan** — `run.jl` was
implemented byte-for-byte from the commit plan and reproduced its stated pass
conditions exactly (deterministic given the seed: two independent runs gave
bit-identical GATE lines).

---

## Background: the ergodic loop and the two constants

Commit 1's doc covers the `Ergodic` estimators in depth; this recaps only what the
experiment leans on. Nothing below is an unexplained symbol.

- **The methodology being validated.** Throughout Units 0–3 the pattern is: draw
  sampled paths, form an empirical statistic, compare to a closed-form analytic
  target. That is only sound if a *finite* sample (or a *finite* time-average of one
  path) genuinely concentrates on the analytic truth. Prop. 1.16 is that guarantee:
  for a stationary process with integrable correlation `C ∈ L¹`, the time-average
  `A_T = (1/T)∫₀ᵀ X_s ds → μ` in mean square as `T → ∞`. Since every sampler here
  draws a **zero-mean** law, `μ = 0`, and the mean-square of `A_T` **is** its
  variance about the true mean.

- **The finite-T variance (Lemma 1.17).** Before the limit,
  ```
  Var(A_T) = (2/T²) ∫₀ᵀ (T − u) C(u) du  →  2·D*/T   as T → ∞.
  ```
  So `Var(A_T) ∝ T^{-1}` (gate a, the *slope*) with constant `2·D*` (gate b, the
  *rate constant*).

- **The two easily-confused constants (coincide only at α = 1).** For the OU
  correlation `C(u) = (D/α)·exp(-α|u|)`:
  ```
  R(0) = C(0) = D/α        the variance at zero lag
  D*   = ∫₀^∞ C = D/α²     the Green–Kubo transport coefficient (Example 1.18)
  ```
  The decay constant is `2·D* = 2D/α²`, so it **carries α², not α**. With the
  experiment's `D = 1, α = 2` these are `2·D* = 0.5` vs `2·R(0) = 1.0` — a clean
  factor-of-2 apart. Choosing `α ≠ 1` is deliberate: at `α = 1` the two constants
  collapse and gate (b) could not distinguish "the code computes `D/α²`" from "the
  code computes `D/α`". Here it must land on `0.5`, not `1.0`, to pass.

- **The MSD identity.** The integrated mean-square displacement `E[(∫₀ᵗ X_s ds)²]`
  grows like `2·D*·t`. It is *algebraically* `t²·Var(A_T)` on the same paths (the
  running average is the integral divided by `t`), so its log-log slope is exactly
  gate (a)'s slope `+2`. It is a restatement, not an independent measurement —
  hence a diagnostic, not a gate.

---

## What changed

One new script and its three committed figures. No library or test file changes.

### `experiments/04_ergodicity/run.jl` (new, 146 lines)

Opens with the mandated reproducibility setup: `ENV["GKSwstype"] = "100"` **before**
`gr()` (headless plotting), and a single recorded `StableRNG(SEED_OU)` stream. Every
stochastic call receives that explicit RNG — no bare `randn()`.

**The constants (each, and why its value).**

| const | value | meaning / why |
|-------|-------|----------------|
| `SEED_OU` | `20260718` | the single recorded `StableRNG` seed. |
| `D` | `1.0` | OU noise strength. |
| `ALPHA` | `2.0` | OU relaxation rate — **deliberately ≠ 1** so `2·D* = 0.5` differs from `2·R(0) = 1.0` (gate b then pins `α²`). |
| `DT` | `0.05` | time step; Nyquist `π/DT ≈ 63 ≫ α`, so the OU correlation is well-resolved. |
| `N_GRID` | `2^14 = 16384` | one ensemble of length-16384 records → `T_max ≈ 819` (well over 1600 correlation times). |
| `N_ENS` | `4000` | ensemble size; per-point relative scatter of `Var` is `~√(2/N) ≈ 2%`. |
| `TMIN_FIT` | `10.0` | fit the asymptote only at `T ≥ 10`, ~20 correlation times past `1/α = 0.5`, so finite-T curvature is negligible vs the gate's noise budget. |
| `NLADDER` | `14` | geometric T-ladder points (subsampled from nested prefixes). |
| `TPLATEAU` | `20.0` | gate (b) reads `T·Var` on the plateau `T ≥ 20`. |
| `NGROUP` | `20` | disjoint sub-ensembles for the honest MC slope SE (see gate a). |

`@assert N_ENS % NGROUP == 0` guards the group-SE partition (else the last columns
silently drop out).

**Two self-contained helpers (no `Statistics` import).** The script is deliberately
`Statistics`-free — same discipline as commit 1's module and the sibling
`00_covariance_core` experiment:

- `ols_slope_se(x, y)` — inline OLS returning the fitted slope *and* its residual
  standard error `√(s²/Sxx)`. Textually the same helper already used in
  `experiments/00_covariance_core/run.jl`.
- an inline sorted-median for gate (b): `cvals[div(length(cvals)+1, 2)]` on the
  sorted plateau values.

**The one ensemble (drives everything).** A single seeded circulant-embedding OU
ensemble is drawn once into an `N_GRID × N_ENS` matrix, one path per **column** (the
load-bearing `CLAUDE.md` orientation; a transpose estimates the wrong object):

```julia
r_seq = [exponential_kernel(0.0, k*DT; D=D, alpha=ALPHA) for k in 0:N_GRID-1]
Dstar = green_kubo(r_seq, DT)                    # library, not hard-coded
rng   = StableRNG(SEED_OU)
paths = Matrix{Float64}(undef, N_GRID, N_ENS)
for j in 1:N_ENS
    paths[:, j] = sample_circulant_embedding(r_seq, rng)
end
```

Crucially `D*` is obtained from the library's own `green_kubo(r_seq, DT)` (realized
`0.250208`), matching the analytic `D/α² = 0.25` to 5 decimals — the analytic value
is *checked*, not assumed. All downstream statistics are then read off this one
matrix: `time_average_variance`, `mean_square_displacement`, and the exact
`time_average_variance_exact(r_seq, DT)` curve.

**Gate (a) — variance slope → −1.** The point estimate is the OLS slope over a
14-point geometric T-ladder subsampled from the grid (`argmin` snap to actual grid
times), fitting only `T ≥ TMIN_FIT`. Realized slope `-0.9926`. The MC uncertainty is
**not** the OLS residual SE — the ladder points are nested prefixes of one path
matrix, so their residuals are autocorrelated and the OLS SE under-estimates. Instead
the honest slope SE comes from re-fitting the same ladder on `NGROUP = 20` **disjoint**
sub-ensembles and taking the spread of those slopes:
```julia
gate_a = abs(slope_v + 1) < 2.5 * se_v
```
Realized `|slope+1| = 0.0074 < 2.5·SE_grp = 0.0164` → **PASS**, ~1.1 SE of headroom.
For contrast the two SEs are printed: `SE_grp = 0.0066` vs `SE_ols = 0.0023` — the
OLS SE is ~3× too small, exactly the under-estimate the group method avoids.

**Gate (b) — constant → 2·D* (pins α²).** Slope-free and robust: on the plateau
`T ≥ TPLATEAU` the product `T·Var(A_T) → 2·D*`, so the gate reads the **median** of
those plateau products (robust to the growing tail scatter of the estimator) and
checks it against `2·D*` within a 5% relative tolerance:
```julia
gate_b = abs(cmedian - 2*Dstar) / (2*Dstar) < 0.05
```
Realized `median(T·Var) = 0.49481` vs `2·D* = 0.50042`, relative error `0.0112 < 0.05`
→ **PASS**. The printed line also shows `2·R(0) = 1.0` for contrast — "would be wrong"
— making explicit that the gate lands on the transport coefficient, not the zero-lag
variance.

**MSD diagnostic (not a gate).** `ols_slope_se` on the MSD ladder gives slope `1.0074`
against target `+1`. Because `msd ≡ t²·tav` exactly (a deterministic identity from the
shared `_cumulative_integral`), this is gate (a) shifted by `+2`, not an independent
check — labelled a DIAGNOSTIC accordingly.

### The three figures (committed artifacts)

`experiments/04_ergodicity/figures/{variance_vs_T,msd_vs_t,running_average_band}.png`,
regenerated by `run.jl` and committed per the `CLAUDE.md` "commit their figures" rule:

1. **`variance_vs_T.png`** — the MC `Var(A_T)` on log-log, overlaid on **two**
   references: the exact Lemma-1.17 curve `time_average_variance_exact` (the full
   closed-form trapezoid identity, valid at *all* T, not just the asymptote) and the
   `2·D*/T` asymptote (dashed). The 14-point fit ladder is marked as scatter. The MC
   curve tracks the exact curve and settles onto the asymptote.
2. **`msd_vs_t.png`** — the MC integrated MSD on log-log against the `2·D*·t`
   reference line, title showing the realized `+1.007` slope.
3. **`running_average_band.png`** — six individual running-time-average paths
   (columns 1–6 of the same ensemble) settling into the shrinking `±√(2·D*/T)`
   envelope as `T` grows, funneling toward `0`. This is the direct visual of Prop.
   1.16: one path's time-average converging in L² to the true mean (0, since the law
   is zero-mean).

---

## Why these design choices

- **One ensemble drives all three artifacts (no re-draw).** Gates (a), (b), the MSD
  diagnostic, and all three figures read the single `paths` matrix. This keeps the
  RNG stream a single fixed sequence — reordering or re-drawing would change every
  committed number and figure (the `CLAUDE.md` "do not reorder RNG draws" rule) — and
  makes the MSD/variance identity hold on *the same data*.

- **Disjoint-sub-ensemble SE, not the OLS residual SE (gate a).** The ladder is built
  from nested prefixes of one path matrix, so successive points are strongly
  autocorrelated; the textbook OLS residual SE assumes independent residuals and so
  under-reports the true slope uncertainty (here by ~3×). Splitting the ensemble into
  20 disjoint groups and measuring the spread of independently-fitted slopes gives an
  honest SE — the only defensible denominator for a "within a small multiple of the
  statistic's own noise" gate (`CLAUDE.md`'s Monte-Carlo convention). Fitting deep in
  the asymptotic regime (`T ≥ 10`) further ensures the remaining bias (finite-T
  curvature) is far below that noise, so the slope really targets `-1`.

- **A slope-free median for gate (b).** The constant gate deliberately avoids a second
  fit. On the plateau `T·Var` is a flat sequence with heavy tail scatter (the
  estimator's variance grows with T); the median of the plateau products is robust to
  that scatter in a way a mean or an endpoint read would not be. It pins the rate
  constant directly.

- **`α = 2`, not `α = 1` (the crux of gate b).** Spelled out in Background: only
  `α ≠ 1` separates `D* = D/α²` from `R(0) = D/α`, so this is what turns gate (b) into
  a real test of the `α²` in the Green–Kubo coefficient rather than a value that would
  pass whether the code squared `α` or not.

- **Two overlay references on the variance figure.** Plotting both the exact
  Lemma-1.17 curve *and* the `2D*/T` asymptote (not just the asymptote) lets a reader
  see the MC estimate agree with the *finite-T* truth in the pre-asymptotic region and
  only then merge onto the straight-line asymptote — distinguishing "the estimator is
  right everywhere" from "the estimator happens to hit the tail slope".

---

## Every gate / figure, and why it bites

Unlike the deterministic `test/` tier, these are Monte-Carlo artifacts: they "bite" by
moving measurably under a real defect, calibrated to the sampling noise, not by exact
equality.

1. **Gate (a), variance slope −0.9926.** A sampler regression that broke the
   stationarity or correlation structure of the OU paths would change the decay
   exponent away from `-1`; the `2.5·SE_grp` band (`0.0164` wide) is tight enough to
   catch a real slope error yet calibrated to the ensemble's own noise so it does not
   false-fail on the committed seed (a seed sweep in the plan found headroom
   `≤ 2.1·SE_grp` across seeds, so this seed's 1.1-SE margin is unremarkable/robust).

2. **Gate (b), constant 0.49481 vs 0.50042.** The discriminating check. Any code path
   that computed the rate constant as `2·R(0) = 1.0` instead of `2·D* = 0.5` — i.e.
   dropped the second `α` — would land at ~1.0, a 100% relative error, blowing far past
   the 5% gate. This is the assertion that pins `D/α²` specifically, and it can only do
   so because `α ≠ 1`.

3. **MSD diagnostic, slope 1.0074.** Confirms the `msd = t²·tav` identity holds on the
   real ensemble (slope exactly `+2` above gate a). A discretization drift between the
   two estimators — if they did *not* share `_cumulative_integral` — would break the
   exact `+2` offset.

4. **`variance_vs_T.png`.** The MC curve must hug the exact Lemma-1.17 curve through
   the bend and lie on the `2D*/T` asymptote at large T. A wrong divisor, a transpose,
   or a mis-scaled `D*` would separate the MC curve from the exact overlay visibly.

5. **`running_average_band.png`.** All six running averages must funnel into the
   shrinking `±√(2D*/T)` envelope and toward 0. If Prop. 1.16 convergence failed (or
   the paths carried a spurious mean), some path would escape the band or fail to
   settle — the qualitative signature of a broken ergodic loop.

---

## Empirical / runtime verification

Running `julia --project=experiments experiments/04_ergodicity/run.jl` produced (exit
code 0, twice, bit-identical):

```
Green-Kubo D* = green_kubo(r,dt) = 0.250208  (analytic D/alpha^2 = 0.250000)
GATE (a) variance slope: -0.9926;  |slope+1| = 0.0074  vs  2.5*SE = 0.0164  (SE_grp 0.0066, SE_ols 0.0023) -> PASS
GATE (b) constant: median(T*Var) = 0.49481  vs  2 D* = 0.50042  (rel 0.0112, 2R(0)=1.000 would be wrong) -> PASS
DIAGNOSTIC MSD slope: 1.0074 +/- ... (integrated restatement of gate (a); target +1)

recorded: seed=20260718, D=1.0, alpha=2.0, dt=0.050, n_grid=16384, n_ens=4000, Dstar=0.25021
ALL GATES: PASS
```

Both gates PASS, the library `green_kubo` matches the analytic `D/α² = 0.25` to 5
decimals, and the run is deterministic given the seed (two independent runs identical
to 6 decimals). All three figures were generated under headless plotting and **visually
inspected**: titles, axis labels and legends all present and inside frame; the variance
figure shows the MC curve tracking the exact Lemma-1.17 curve onto the `2D*/T`
asymptote; the band figure shows all six running averages funneling into the shrinking
envelope toward 0.

The deterministic suite is untouched (no `src/`/`test/` change) — this experiment is
intentionally outside CI, run locally with its figures committed, per the two-tier
convention.

## Trade-offs and known limitations

- **Gate (b)'s "median" takes the lower-middle element on an even-length array**
  (`cvals[div(length(cvals)+1, 2)]`), rather than averaging the two central values —
  not the textbook median. Left verbatim as the plan's pinned, already-run code (its
  exact output `0.49481` *is* the plan's stated pass condition), and immaterial under
  the 5% relative tolerance. Changing it would diverge from the validated plan for no
  gain.

- **The `NGROUP` sub-ensemble loop recomputes `time_average_variance` over work already
  done in the full-ensemble pass** (~2× redundant compute). Accepted: the project's
  working agreement deprioritizes efficiency relative to correctness, the run finishes
  in seconds, and the script is verbatim/finalized.

- **`ols_slope_se` is duplicated verbatim** from `00_covariance_core/run.jl`. Consistent
  with the repo's existing self-contained-experiment-script pattern (there is no shared
  `experiments/common.jl`), so not a new violation — left as-is.

## Code review

Ran `/code-review` (high effort) on the diff. **No correctness bug found.** Two
non-trivial candidate findings surfaced — the even-length median and the redundant
sub-ensemble recompute (both above) — and were **consciously declined**: each attacks
code the plan pins as already-run-and-confirmed, and neither affects correctness under
the gate tolerances. A third, low-severity nit (the duplicated `ols_slope_se` helper)
was noted as consistent with the repo's existing self-contained-script pattern, not a
new violation. No action taken; the script is transcribed verbatim from the finalized
plan.

## Deviations from plan

**None.** `run.jl` was implemented byte-for-byte from the commit plan and its realized
numbers match the plan's stated pass conditions exactly. The only new filesystem
objects are the `experiments/04_ergodicity/` directory and its `figures/` subdirectory,
both implied by the plan's file list.

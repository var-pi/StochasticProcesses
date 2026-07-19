# Commit 3 — the non-integrable negative control (Unit 4 "ergodicity", feature `04-ergodicity`)

## TL;DR

Unit 4's finale: a **falsifier**. Commits 1–2 showed the ergodic theorem (Pavliotis
Prop. 1.16) *holding* — the time-average of one OU path converges in L² to the mean, and
its variance decays like `2·D*/T`. But Prop. 1.16 requires an **integrable** correlation
(`C ∈ L¹`), and the OU correlation is integrable, so commits 1–2 never tested whether
that hypothesis is actually load-bearing. This commit does: it feeds the **exact same**
`time_average_variance` estimator a *different* synthetic stationary process whose
correlation `C(u) = (1+u)^{-1/2}` is a valid covariance but is **non-integrable**
(`∫₀^∞ C = ∞`). With the L¹ hypothesis broken, the variance-vs-T slope no longer lands on
`-1` — it comes in near `-0.41`. The 1/T rate was the theorem *doing work*, not decoration.

Mechanically this is small and surgical: it **appends one block** to the end of the
already-committed `experiments/04_ergodicity/run.jl` (replacing only the previous final
two lines — the `recorded:` printf and the `ALL GATES` line) and adds **one committed
figure**, `nonintegrable_control.png`. No `src/` or `test/` change; nothing enters CI
(Monte-Carlo stays out of CI per `CLAUDE.md`). The new block uses a fresh, independent
`StableRNG(13579)` drawn strictly *after* the main OU stream is fully consumed, so gate
(a)/(b)'s committed numbers are provably unperturbed — confirmed bit-identical.

The new gate (c) PASSES; all three gates PASS.

---

## Background: why a non-integrable control, and what makes it valid

Only the two facts the control rests on; commit 2's doc covers the rest of the loop.

- **The hypothesis under test.** Prop. 1.16 (ergodic theorem for stationary processes)
  guarantees `A_T = (1/T)∫₀ᵀ X_s ds → μ` in L² *provided* `C ∈ L¹`. Lemma 1.17's exact
  identity `Var(A_T) = (2/T²)∫₀ᵀ (T−u) C(u) du` collapses to the clean `2·D*/T` — a
  `T^{-1}` slope — precisely because `D* = ∫₀^∞ C` is **finite**. Kill integrability and
  `D*` diverges; the collapse to `2·D*/T` cannot happen, and the slope must change.

- **A correlation that is PSD but not integrable.** The control needs a `C` that is a
  *bona fide* covariance (else it cannot be sampled) yet violates `C ∈ L¹`. The choice is
  ```
  C(u) = (1 + u)^{-1/2}
  ```
  It is positive-definite by **Pólya's criterion**: a real, even function that is convex
  and monotonically decreasing to 0 on `[0,∞)` with `C(0)=1` is the characteristic
  function of some distribution, hence a valid PSD kernel — so its Toeplitz covariance
  matrix is (numerically) factorable and the process is Cholesky-samplable. But its tail
  decays too slowly to integrate: `∫₀^∞ (1+u)^{-1/2} du = ∞`. So it is a valid
  stationary process to which Prop. 1.16 **does not apply** — exactly the negative control.

- **What the broken rate actually is.** Lemma 1.17 is still finite at finite T even when
  `C ∉ L¹` (the inner integral is over a bounded window), so there is still an *exact*
  finite-T variance curve — it just no longer decays like `1/T`. For this `C` the true
  asymptotic slope is `-1/2`, but the finite-T curve reaches it only very slowly: a
  correction of the form `slope ≈ -1/2 + 0.75/√T` keeps the *local* slope well **above**
  `-1/2` at any T this control can reach. This finite-T subtlety is the crux of how gate
  (c) is designed (below).

---

## What changed

One appended block in `run.jl`, one new committed figure. No library or test change.

### `experiments/04_ergodicity/run.jl` — appended negative-control block (~65 new lines)

Everything above the previous final two lines is untouched; the block is inserted where
the old `recorded:` printf used to be, and the printf/`ALL GATES` lines are rewritten to
add the control's recorded params and fold `gate_c` into the pass tally.

**Reused, not redefined.** The block leans on names already in scope from the main block:
`ols_slope_se`, `time_average_variance`, `time_average_variance_exact`, `NLADDER`,
`NGROUP`. No shadowing — the control introduces only new names (`_c`/`_ctrl` suffixed).

**Its own constants (fresh seed, modest grid):**

| const | value | why |
|-------|-------|-----|
| `SEED_CTRL` | `13579` | fresh, independent `StableRNG`, consumed only after the main OU stream — cannot perturb gate (a)/(b). |
| `N_C` | `4096` | modest: this control Cholesky-factors an `N_C × N_C` Toeplitz matrix, O(n³) (vs the main experiment's cheap circulant embedding at `2^14`). |
| `DT_C` | `0.1` | grid step → `T_max ≈ N_C·DT_C ≈ 410`. |
| `N_ENS_C` | `2000` | ensemble size (`@assert N_ENS_C % NGROUP == 0` guards the sub-ensemble partition). |
| `JITTER` | `1e-10` | the repo's standard Cholesky nugget, added before factoring and **reported** in the final printf. |

**Building and sampling the process (factor once):**

```julia
Cnon(u) = 1 / sqrt(1 + u)                                  # non-integrable, PSD (Pólya)
r_c   = [Cnon(k * DT_C) for k in 0:N_C-1]
Σc    = [r_c[abs(i - j) + 1] for i in 1:N_C, j in 1:N_C]   # stationary → Toeplitz from r_c
Lc    = cholesky(Symmetric(Σc) + JITTER * I).L             # factor ONCE (Σ is fixed)
paths_c = Lc * randn(StableRNG(SEED_CTRL), N_C, N_ENS_C)   # N_C × N_ENS_C, one path per COLUMN
```

Because the covariance is fixed, the whole ensemble is one `L·Z` matrix multiply on a
single seeded draw — no per-column resampling. Orientation is the load-bearing
`CLAUDE.md` one: `n_grid × N`, one path per **column**, matching what
`time_average_variance` consumes.

**The measured vs exact slopes.** The MC variance curve, the exact Lemma-1.17 curve for
this same `r_c`, and the disjoint-sub-ensemble slope SE are all built exactly as gate (a)
was in commit 2 — same 14-point geometric `argmin`-snapped ladder (here from `T = 20` to
`0.95·T_max`), same `NGROUP = 20` disjoint groups for the honest SE:

```julia
slope_c,   _ = ols_slope_se(log10.(Tk_c), log10.(tav_c[idx_c]))
slope_cex, _ = ols_slope_se(log10.(Tk_c), log10.(exact_c[idx_c]))   # deterministic exact-curve slope
# ... gsl_c: refit the ladder on NGROUP disjoint sub-ensembles; se_c = spread/√NGROUP
```

### `experiments/04_ergodicity/figures/nonintegrable_control.png` (new, committed)

Log-log MC `Var(A_T)` overlaid with the exact Lemma-1.17 curve for this `C`, plus **two
reference (not fitted) slope lines**, both anchored at the same point `(T0, a0)` on the
exact curve:

- a **dashed** slope `-1/2` — the true asymptote (not yet reached at this `T_max`);
- a **dotted** slope `-1` — what an *integrable* `C` would give.

The MC and exact curves both sit far above the `-1` line across the entire plotted range:
the integrable-case rate is visibly, unambiguously falsified. Committed per the
`CLAUDE.md` "commit their figures" rule.

---

## Gate (c) — the falsifier, and why it has two parts

Gate (c) is deliberately **not** the naive check "does the slope hit `-1/2`?". That naive
gate would *fail* on honest data, because the finite-T curve has not reached `-1/2` at
`T_max ≈ 410` (see Background). So the gate is split:

- **(c1) tracking.** The MC-fitted slope must track the **exact finite-T Lemma-1.17
  slope** computed from `time_average_variance_exact(r_c, DT_C)` on the same ladder — not
  the asymptotic `-1/2`. This asks "does the Monte-Carlo estimator reproduce the true
  finite-T behavior of *this* process?", the honestly reachable claim.
  ```julia
  gate_c1 = abs(slope_c - slope_cex) < 2.5 * se_c
  ```
  Uses the same disjoint-sub-ensemble SE as gate (a) — the ladder points are nested
  prefixes and hence autocorrelated, so the sub-ensemble spread, not the OLS residual SE,
  is the honest slope uncertainty.

- **(c2) the falsifier proper.** The slope must be clearly shallower than `-1`:
  ```julia
  gate_c2 = slope_c > -0.75
  ```
  This is the headline claim — with the L¹ hypothesis broken, the rate an integrable `C`
  would produce (`-1`) is *falsified*. The threshold `-0.75` sits comfortably between the
  realized `-0.41` and the falsified `-1`.

`gate_c = gate_c1 && gate_c2`. It bites in two directions at once: a regression that
silently restored a `1/T` decay (e.g. a divisor bug, or the estimator ignoring the
supplied covariance) would drive the slope toward `-1` and trip (c2); a regression that
warped the finite-T shape would break the (c1) tracking against the deterministic exact
curve.

---

## Why these design choices

- **Direct Cholesky on the literal Toeplitz matrix, not `sample_circulant_embedding`.**
  The circulant-embedding sampler used by the main block has no size parameter to
  auto-pad the embedding, and a slowly-decaying non-integrable `C` is **not guaranteed
  PSD under the naive even-extension embedding** it performs (its own docstring warns of
  this for fBm-like kernels). Factoring the literal Toeplitz covariance is the safe
  choice — at the cost of O(n³), which is why `N_C = 4096` here rather than the main
  experiment's `2^14`.

- **Factor once, sample as `L·Z`.** The control covariance is fixed, so the entire
  ensemble is a single `Lc * randn(...)` matrix product on one seeded draw — cheaper and
  simpler than per-column resampling, and it keeps the RNG stream a single fixed sequence.

- **Fresh independent seed, appended after the main stream.** `SEED_CTRL = 13579` is its
  own `StableRNG`, consumed strictly after the main block's `rng = StableRNG(SEED_OU)`
  stream is exhausted. This is what lets the control be *added* without perturbing any
  committed number — verified: `Dstar`, gate (a) slope, gate (b) median are all
  bit-identical to commit 2's committed values with the block present.

- **Gate the tracking, not `-1/2`.** Reaching the `-1/2` asymptote numerically would need
  `T_max` in the thousands — infeasible under O(n³) Cholesky. Gating (c1) against the
  exact finite-T curve, plus (c2) against the falsified `-1`, makes a rigorous claim that
  is actually reachable at `T_max ≈ 410`.

---

## Empirical / runtime verification

Ran `julia --project=experiments experiments/04_ergodicity/run.jl` end-to-end twice (exit
code 0 both times, bit-identical output — deterministic given the seeds):

```
Green-Kubo D* = 0.250208  (analytic D/alpha^2 = 0.250000)
GATE (a) variance slope: -0.9926;  |slope+1| = 0.0074  vs 2.5*SE = 0.0164 -> PASS   [unchanged vs commit 2]
GATE (b) constant: median(T*Var) = 0.49481  vs 2 D* = 0.50042  (rel 0.0112) -> PASS  [unchanged vs commit 2]
CONTROL slope: MC -0.4092  exact -0.4207;  |MC-exact| = 0.0116  vs 2.5*SE = 0.0342  (c1 PASS);
  slope > -0.75 (c2 PASS; -1 would be FALSE) -> PASS
ALL GATES: PASS
```

- The realized control slope `-0.4092` is nowhere near `-1` — the falsifier fires as
  intended, and (c1) tracks the deterministic exact slope `-0.4207` to within `0.0116`,
  well inside `2.5·SE = 0.0342`.
- Gate (a)/(b)'s numbers are **byte-identical** to commit 2's committed values, confirming
  the fresh-seed-after-stream isolation empirically.
- The extended recorded line now carries the control params:
  `... | control: seed=13579, n_c=4096, dt_c=0.10, n_ens_c=2000, jitter=1.0e-10`.
- `nonintegrable_control.png` was visually inspected: title, axis labels and legend all
  present and inside frame; both the MC and exact curves sit far above the dotted `-1`
  reference across the whole range.

No deterministic unit test was added — the library functions the block calls
(`time_average_variance`, `time_average_variance_exact`, plus `cholesky`/`Symmetric`)
already ship and are exercised by commits 1–2. `test/runtests.jl` is untouched; this
Monte-Carlo experiment stays out of CI per the two-tier convention.

## Trade-offs and known limitations

- **The `-1/2` asymptote is not reached numerically.** By design — the finite-T curve is
  still climbing toward it at `T_max ≈ 410`, and reaching it would demand an infeasible
  O(n³) grid. The control claims what it can rigorously show (finite-T tracking + a clearly
  non-`-1` slope), not the asymptote.

- **O(n³) Cholesky caps the reachable T.** The literal-Toeplitz factoring is the price of
  guaranteeing PSD-ness for a non-integrable `C`; it is what forces the modest `N_C = 4096`.
  Accepted: the run finishes in seconds and the gate is calibrated to what that grid can show.

## Code review

Ran `/code-review` (medium level, 1-vote verify) on the diff. An independent fresh-context
reviewer checked: off-by-one in the new Toeplitz construction and ladder subsampling, name
collisions with the untouched main block, RNG-stream isolation (fresh `StableRNG` consumed
only after the main stream is exhausted), the Cholesky-nugget convention (`JITTER` added
before factoring, `Symmetric` wrap present, reported in the final printf), path-matrix
orientation, dead code/efficiency, and `CLAUDE.md` convention violations. **Zero findings
survived** — nothing confirmed or plausible.

One purely stylistic nit was raised and **consciously declined**: the control ladder's
lower T-bound is a bare `20.0` literal rather than a named constant like the main block's
`TMIN_FIT`. The commit plan pins this block's text verbatim (it was run end-to-end against
the committed library before being finalized into the plan), so introducing a new named
constant would be an unrequested deviation from a pre-resolved decision, not a correctness
or coverage fix.

## Deviations from plan

**None.** The block was applied byte-for-byte from the plan's "Exact edit" section,
replacing only the previous final two lines of `run.jl`. The only new filesystem object
besides this doc is `experiments/04_ergodicity/figures/nonintegrable_control.png`.
Unit 4's CLAUDE.md / README updates are deliberately deferred to a later finalization
commit (mirroring the Unit 3 precedent of a dedicated final "README, gallery row,
CLAUDE.md" phase); this commit's plan explicitly scoped "no other file changes".

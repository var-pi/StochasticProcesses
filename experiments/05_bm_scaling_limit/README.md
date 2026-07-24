# 05 · Random walk → Brownian motion — Donsker's invariance principle

Units 0–4 treated Brownian motion as **one object**: the covariance operator with kernel
`R(t,s) = min(t,s)` — assembled, diagonalized, square-rooted, sampled. This unit steps off that
spine to show the *other* face of the same process. Brownian motion is also the **weak limit of a
rescaled random walk**: form the partial sums `S_k` of iid mean-0, variance-1 increments and
rescale space by `√n`, time by `n`,

```
W_n(t) = S_{⌊n·t⌋} / √n,    t ∈ [0, 1],
```

and as `n → ∞` the path `W_n` converges in law to standard Brownian motion — **for any increment
law whatsoever**, as long as it has mean 0 and finite variance. That universality is Donsker's
invariance principle, and it is a statement about laws on *path space*, not an operation on a
covariance operator.

Because of that, this unit adds **no new `src/` code** — the rescaled-walk builder lives entirely
in the experiment file. Its only library calls are `empirical_cov` (Unit 0) and `ks_statistic`
(Unit 3's `src/gof.jl`), reused here to interrogate the limit from four angles: its covariance,
its marginal *rate*, a path functional, and a deliberately broken hypothesis.

## The result

The headline asks the question Donsker leaves open — *at what **rate** does the marginal
`W_n(1) = S_n/√n` reach its Gaussian limit `N(0,1)`?* — and the answer is more interesting than
the textbook two-term story. It splits into **three** mechanisms, and the split is legible in a
single log–log plot:

![Log–log KS distance to Φ versus walk length n. Exponential Monte-Carlo points (green, slope −0.502) and Rademacher points (blue, slope −0.497) sit on their dashed exact curves along a −1/2 guide; uniform's exact curve (orange diamonds, slope −1.009) runs a full power steeper over its small-n window; dotted Kolmogorov-floor lines and −1/2 and −1 guide slopes are drawn](figures/ks_rate.png)

> **Key insight —** two roads lead to `n^(−1/2)`, one road to `n^(−1)`. A **skewed** law
> (exponential) and a **lattice** law (Rademacher, `±1`) both converge at `n^(−1/2)` — for
> *different* reasons — while a law that is **smooth *and* symmetric** (uniform) converges a full
> power faster, at `n^(−1)`. This corrects the master-plan brief, which wrongly grouped Rademacher
> with the fast laws.

Nine gates, across four phases, pin every claim. Here is the block from the actual run:

```
GATE 01a [rademacher ] cov-vs-N slope: -0.5261 -> PASS   [uniform -0.5585 PASS, exponential -0.5163 PASS]  (all laws) -> PASS
GATE 02-exp [exponential] MC slope: -0.5016 -> PASS
GATE 02-rad [rademacher ] MC slope: -0.4975 -> PASS
GATE 02-uni-exact: Irwin-Hall exact slope = -1.0088  (|slope+1|=0.0088 vs margin 0.05) -> PASS
GATE 02-sep: |uniform 1.0088| > |exp 0.5016| and > |rad 0.4975| -> PASS
GATE 03 [rademacher n=3000] KS(M,half-normal)=0.01813 (bound 0.05) -> PASS | KS(M,Φ)=0.5000 (>0.30) -> PASS   [uniform n=1200 0.01662 PASS, exponential n=1200 0.02387 PASS]
GATE 04a [pareto] Var(S_n/√n)-vs-n slope: 0.4235 (>+0.10) -> PASS
GATE 04b [pareto] KS(S_n/√n,Φ)-vs-n slope: 0.0607 (>-0.35, vs commit 02's ~-0.50) -> PASS
ALL GATES: PASS
```

## Concept

Donsker's theorem needs almost nothing from the increment law — **only two moments**: mean 0 and
variance 1. The limit is *universal*: always the same Brownian motion, regardless of the law's
shape. So every phase re-runs the same machinery against **three qualitatively different laws**, so
a shape-specific bug cannot hide behind the other two passing:

| Law | Support | Shape | Marginal rate | Why |
|-----|---------|-------|---------------|-----|
| **Rademacher** `±1` | 2-point (lattice) | symmetric, discrete | `n^(−1/2)` | Esseen lattice/discreteness term |
| **uniform** on `(−√3, √3)` | bounded continuous | symmetric, smooth | `n^(−1)` | both `n^(−1/2)` terms vanish → kurtosis leads |
| **exponential** `Exp(1) − 1` | unbounded | skewed (skewness 2) | `n^(−1/2)` | Edgeworth skewness term |

Two facts make the limit cheap to check without ever materializing a continuous path:

- **Only the lattice values matter.** Donsker builds a continuous path by linear interpolation
  between the lattice points `(k/n, S_k/√n)` — but every functional this unit checks (the two-time
  covariance, the KS-vs-n rate, the running maximum) is a function of the lattice values *alone*.
  So `rescaled_walk` returns just the `(n+1)`-vector per path (one path per column, the repo's
  `n_grid × N` orientation), and no interpolation is ever computed.
- **The covariance is exact and n-free.** For iid variance-1 increments, linearity of covariance
  gives `Cov(S_i/√n, S_j/√n) = min(i,j)/n = min(t_i, t_j)` *exactly*, at every lattice time, for
  every `n` — no CLT, no limit. This is the identity Phase 1 pins.

The qualitative content of the theorem — a coarse zig-zag tightening onto a continuous trace as
`n` grows — is visible before any gate:

![One rescaled Rademacher walk at each of n = 4, 16, 64, 256, 1024 overlaid on [0,1]; the n=4 path is a coarse 4-segment zig-zag while the n=1024 path is a visibly continuous, Brownian-like trace](figures/rescaled_paths.png)

## Phase 1 — the covariance lands on min(s, t)

The foundation gate checks that the rescaled walk is a genuine discrete Brownian motion at the
level of second moments. For each law it subsamples 8 interior lattice rows, builds the analytic
target `M[i,j] = min(t_i, t_j)`, reuses `empirical_cov` rather than hand-rolling a covariance, and
verifies the Frobenius error `‖Ĉ_N − M‖_F` decays at the Monte-Carlo rate `N^(−1/2)` as the
ensemble size grows.

![Left: log–log mean Frobenius error versus N for all three laws, each tracking a dashed −1/2 reference across two decades of N. Right: heatmap of the residual |Ĉ_N − M| for the Rademacher law at N=10000, values everywhere below 0.01](figures/covariance_vs_min.png)

Because the covariance identity holds *exactly for every n*, there is no convergence-in-n to gate —
the only thing that can be wrong is whether the builder produces correctly-normalized, independent
increments. And since `empirical_cov` demeans its input and covariance is mean-invariant, this is
**purely a variance-normalization check**, blind to the increment mean by construction. Running it
once per law is what gives it teeth: a wrong variance scale (e.g. dropping the uniform law's `√3`
factor) is caught here at the foundation.

The error bar is the subtle part, and it took three designs to make honest. A single N-ladder fit
badly understates the true noise (at `N = 100` one draw's relative scatter is ~55%, since the 8×8
target has only 36 independent entries to average over). The fix is **classical batch means**: draw
`NGROUP = 20` fully independent replicate ladders, fit a slope per replicate, and gate the *mean*
slope against the standard error *of that mean* — `slope = −0.526`, `|slope + 0.5| = 0.026 <
2.5·SE = 0.065`, PASS. This batch-means SE, reused in every later phase, is the direct analogue of
Unit 4's sub-ensemble slope SE, adapted to an axis where "more data" means more independent
replicates rather than a longer time series.

## Phase 2 — the headline: three roads to the Gaussian marginal

This is the unit's centerpiece. The CLT rate is governed by the Edgeworth expansion of the
standardized CDF around Φ:

```
F_n(x) ≈ Φ(x) − φ(x)·[ (skewness/6)·(x²−1)/√n  +  (kurtosis term)/n  +  … ]
```

The textbook "two-term" reading is: skewed laws converge at `n^(−1/2)` (the skewness term),
symmetric laws at `n^(−1)` (skewness vanishes, kurtosis leads). That is what the brief assumed —
and it misses a **third road** the smooth Edgeworth series cannot see:

- **exponential** (skewness 2) — the skewness term dominates → `n^(−1/2)`.
- **Rademacher** (symmetric, so skewness is *exactly* 0 — but a **lattice** law, only 2 atoms) —
  the skewness term vanishes, yet the Esseen **discreteness correction**, also `O(n^(−1/2))` and
  skewness-*independent*, takes over. Intuitively: a histogram of a coin-flip sum never quite looks
  continuous no matter how much you average — the distribution lives on a grid of spacing `2/√n`,
  and no `n` erases that jump structure. So Rademacher converges at `n^(−1/2)`, **not** `n^(−1)`.
- **uniform** (symmetric **and** smooth — no lattice, no skew) — both `n^(−1/2)` roads are closed,
  leaving the next Edgeworth term (kurtosis), `O(n^(−1))`, a full power faster.

The lattice discreteness is even visible directly — look at the **comb pattern** in the Rademacher
histogram, the exact grid structure the Esseen term quantifies, while all three marginals sit
cleanly on the Gaussian:

![2×3 grid. Top row: histograms of S_n/√n with the N(0,1) density overlaid, for rademacher n=360, uniform n=128, exponential n=320 — all closely Gaussian, with a visible comb pattern in the Rademacher histogram. Bottom row: QQ-plots against standard-normal quantiles, all lying on the y=x line](figures/marginal_qq.png)

### Hybrid gating — and why it has to be

The gating is **hybrid and asymmetric**, for a load-bearing reason. The two `n^(−1/2)` laws have
large signals (~`0.4/√n` for Rademacher, ~`0.13/√n` for exponential), so they are gated by **Monte
Carlo**: the batch-means slope must land within `2.5·SE` of `−0.5`. Uniform's `n^(−1)` signal is
tiny (`≈ 9×10⁻⁴` at `n = 32`) — resolving it by sampling would need `N ~ 10⁸` and still leave under
a decade of usable `n`. It is **physically un-gateable by Monte Carlo**, so uniform alone is gated
against its **exact, deterministic Irwin–Hall curve** (computed in BigInt/BigFloat to dodge the
catastrophic cancellation of the naive alternating sum), which carries no sampling noise at all. All
three exact curves are overlaid on the MC clouds in the headline figure, so the plot shows the
sampled points tracking theory down toward — but not into — the Kolmogorov floor.

<details>
<summary>The war story — why "large signal" did not mean "easy," and why more Monte Carlo made it worse</summary>

Getting the two MC gates to pass *honestly* was a large parameter search with a counter-intuitive
resolution, worth recording because it transfers. The raw KS-vs-n slope is biased away from `−0.5`
by two effects that move in **opposite directions** as the sample size `N` grows:

1. **A fixed, N-independent curvature bias in the *true* rate curve.** Even the exact
   Rademacher-to-Φ curve is not a perfect power law at finite `n`; it carries a `1/n`-type
   correction scaling like ~`0.1/n_min`. This is irreducible by more sampling — the only lever is a
   higher `n_min`.
2. **A finite-N sampling bias in the KS estimator** (the classical Kolmogorov floor
   `E[D_N] ~ 0.87/√N`), which *shrinks* as `N` grows — but it is a **bias, not noise**, so it does
   **not** average away across batch-means replicates.

The counter-intuitive consequence: naively raising the replicate count `NGROUP` to get a "more
reliable" SE makes the gate **worse**. More replicates just narrow the SE around the *same* biased
mean, making a real-but-small discrepancy look *more* statistically significant. Verified at
`N = 1,000,000`: `NGROUP = 20` gave a z-score of 5.66; `NGROUP = 60` gave **10.6**. The resolution
is a genuine per-law sweet spot — `n_min` high enough that the curvature bias is negligible, `N` in
the regime where the floor excess is also small relative to a still-honest SE — tuned separately per
law because their signal amplitudes and curvature-decay rates differ. This is why exponential and
Rademacher use *different* ladders and *different* `N` (and independent RNG streams). The chosen
Rademacher seed passes at a theory-derived configuration; seed-sensitivity is stated plainly, on the
same precedent as Unit 3's route-equivalence gate.

</details>

The verified numbers: exponential slope **−0.5016**, Rademacher **−0.4975** (both within `2.5·SE`
of `−0.5`); uniform's exact curve **−1.0088** (within a fixed `0.05` margin of `−1`); and a
**separation gate** makes the headline claim a checked fact — `|−1.0088|` is decisively steeper
than both `|−0.5016|` and `|−0.4975|`.

## Phase 3 — a path functional: the running maximum → half-normal

Donsker is a statement about the *whole path*, not just its endpoint — so a functional that depends
on the path's shape should converge too. The natural next check is the **running maximum**
`M^(n) = sup_{[0,1]} W_n`. By the reflection principle,

```
sup_{[0,1]} B  =_d  |B_1|  =_d  |N(0,1)|,
```

the **half-normal** law with CDF `2·Φ(x) − 1` for `x ≥ 0`. (The running max over the `n+1` lattice
rows *is* the exact continuous supremum: a straight interpolation segment attains its max at an
endpoint, so no excursion hides between lattice points — no fine grid needed.)

![Empirical CDFs of the running maximum for rademacher (n=3000, KS=0.018), uniform (n=1200, KS=0.017), and exponential (n=1200, KS=0.024), all three lying essentially on the solid black half-normal target; the dashed black Φ wrong-target control sits well above and to the left, starting at 0.5 at x=0 while the half-normal starts at 0](figures/running_max.png)

This is a **consistency + wrong-target** gate, not a rate: at one large fixed `n`, `KS(M,
half-normal)` must be **small** (below `0.05`) **and** `KS(M, Φ)` — the deliberately wrong,
full-normal target — must be **large** (above `0.3`). The second half closes the loophole where
"KS is small" could mean "converged to *some* law near both targets": the half-normal is
emphatically not Φ, since `Φ(0) = 0.5` while the half-normal's CDF is 0 at `x = 0`, so their
sup-distance pins at exactly **0.5000** for all three laws, independent of `n` or `N`. That gives a
>6× discrimination ratio between the largest acceptable correct-target KS and the smallest
acceptable wrong-target KS.

Walk length is sized **per law**: Rademacher's max is lattice-valued, so it carries the same
`~0.4/√n` discreteness term seen in Phase 2 *on top of* the functional's own `~c/√n` deviation, and
needs `n = 3000` versus `1200` for the two continuous laws to land under the same bound.

<details>
<summary>The threshold that budgeted the wrong quantity</summary>

The first gate design failed for all three laws on first pass. It budgeted `KS(M, half-normal) <
0.87/√N + margin` — the finite-N *sampling* floor — implicitly assuming the true finite-n KS is
`≈ 0` plus noise. But `KS(M, half-normal)` at finite `n` is a genuine **deterministic** convergence
deviation of size `~c/√n` (a max-functional analogue of Phase 2's Edgeworth corrections), several
times larger than the sampling floor at these `N`. Conflating a rate-scale quantity with a fixed
sampling margin — quantities that differ by an order of magnitude — is what failed. The fix: gate
against a **principled absolute bound** (`0.05`) sized to that deterministic deviation, and raise
`n` so the deviation lands comfortably under it. Chasing the sampling floor instead would need a
multi-GB lattice for Rademacher and buy no added rigor, since the wrong-target discrimination is
already decisive by >6×.

</details>

## Negative control — break finite variance, break Donsker

Every phase so far used increments with mean 0 **and finite variance** — and the variance-1
normalization only fixes the scale; the *finiteness* is the load-bearing hypothesis. The falsifier
drops it: feed the *same* `rescaled_walk` machinery a symmetric, mean-0, but **genuinely
infinite-variance** increment and show the `n^(−1/2)` picture breaks — not vaguely "worse," but in
the *opposite* direction of every earlier gate.

The increment is a **symmetric Pareto**, `sign · U^(−1/γ)` with `U ~ Uniform(0,1)` drawn
**untruncated** and tail index `γ = 1.5`. Sitting in the critical window `(1, 2)`, it has an honest
mean of exactly 0 (odd symmetry, `γ > 1`) but a provably infinite second moment
(`E[X²] = γ ∫₁^∞ x^(1−γ) dx` diverges at `γ ≤ 2`) — infinite variance, not merely large.

> **The load-bearing correctness point —** the tail must be left **untruncated**. A "numerical
> safety" clamp on `U` or a cap on the magnitude would silently restore a *finite* variance and
> defeat the whole falsifier. Gate 04a is the structural guard against exactly that: a secretly
> finite variance would settle instead of grow, and the gate would fail.

![Two panels. Left: log–log mean Var(S_n/√n) over 20 replicates versus walk length n = 50…800 for the Pareto (γ=1.5) increment — a purple curve rising from ~850 at n=50 to ~2×10⁴ at n=800 with batch-mean slope 0.423, tracking a dashed 2/γ−1=0.333 reference and sitting orders of magnitude above the gray dotted "finite-variance target Var≡1" line. Right: QQ plot of the standardized endpoint at n=800 against N(0,1), the empirical points forming a steep S-curve reaching ±60 at the ±2.5 theoretical quantiles, the signature of heavy tails](figures/infinite_variance_control.png)

Two contrasting gates pin the failure against Phase 2's baseline, both testing a *direction*, not a
rate:

- **Gate 04a — variance grows.** Under a finite-variance law, `Var(S_n/√n)` is *identically 1* for
  every `n` (slope 0). Under the Pareto, `S_n/√n = n^(1/γ − 1/2)·Y_n` with `Y_n` converging to a
  `γ`-stable law, so the scale **diverges** as `n^(1/3)`. The batch-means slope of `Var` vs `n` must
  be positive — observed **+0.4235**, clearing the `+0.10` margin. (It is gated on *sign*, not the
  `+2/3` theoretical exponent, because a finite-N sample variance of an infinite-variance quantity
  is a downward-biased, high-variance estimator — the point is that positive growth is impossible
  for a finite-variance law.)
- **Gate 04b — no Gaussian limit.** Because the scale diverges, at any fixed `x` the marginal CDF
  `F_n(x) → 1/2` (a diverging-scale symmetric law puts vanishing mass in any fixed window), so
  `KS(S_n/√n, Φ)` trends *toward* the same ~0.5 uninformative ceiling as Phase 3's wrong-target
  control. Reusing Phase 2's estimator *verbatim*, the slope must **not** decay like the
  finite-variance `n^(−1/2)`: observed **+0.0607**, flat-to-positive, decisively above the `−0.35`
  floor and far from commit 02's `≈ −0.50`.

The falsifier fails in **both** predicted directions, so the finite-variance hypothesis is
confirmed load-bearing.

## Recorded configuration

Reproducibility conventions (why an explicit seed, the RNG rule) live in the
[top-level README](../../README.md#conventions); this unit's concrete values, in draw order:

- **Phase 1 — covariance.** Increment laws Rademacher / uniform `U(−√3, √3)` / exponential
  `Exp(1) − 1`, each iid mean-0 variance-1. Lattice `n = 64`, 8 subsampled interior rows; MC
  sample-count ladder `N = [100, 320, 1000, 3200, 10000]`; `NGROUP = 20` batch-means replicates,
  gate multiple `2.5·SE`. Seeds `20260722` (Donsker overlaid-paths figure), `20260723` (GATE 01a,
  one stream threaded across the pinned `LAW_ORDER`).
- **Phase 2 — KS-vs-n rate.** Exponential: ladder `n = [40, 80, 160, 320]`, `N = 1,000,000` per
  rung, seed `12345`. Rademacher: ladder `n = [150, 200, 270, 360]`, `N = 500,000` per rung, seed
  `999` (independent streams — deliberately, so each law is tuned to its own `N`/ladder). Both
  `NGROUP = 20`, gate `2.5·SE`. Uniform exact Irwin–Hall gate over `n = [8, 16, 32, 64, 128]`, fixed
  margin `0.05`, sanity-checked at `n = 8/16/32` first.
- **Phase 3 — running maximum.** Per-law walk length `n =` {rademacher 3000, uniform 1200,
  exponential 1200}; per-law sample count `N =` {rademacher 30000, uniform 50000, exponential
  50000}; correct-target bound `0.05`, wrong-target control `0.3`; seed `20260724` (one stream
  across `LAW_ORDER`).
- **Negative control — infinite variance.** Symmetric Pareto, tail index `γ = 1.5`, tail left
  **untruncated**. Walk-length ladder `n = [50, 100, 200, 400, 800]`, `N = 2000` per rung,
  `NGROUP = 20`; variance-slope margin `+0.10`, KS-slope floor `−0.35`; seed `20260725` (own stream,
  drawn last; `:pareto` added to the sampler Dict *without* extending `LAW_ORDER`, so no earlier
  number shifts).

This experiment is fully Monte-Carlo (four seeded phases) — run it locally
(`julia --project=experiments experiments/05_bm_scaling_limit/run.jl`); it is **not** part of CI.
The six figures above are committed artifacts. This unit deliberately adds no `src/` code: the
library routines it drives (`empirical_cov`, `ks_statistic`) are covered by their own Unit 0 / Unit
3 testsets in `test/runtests.jl`, and the suite stays green at 177/177.

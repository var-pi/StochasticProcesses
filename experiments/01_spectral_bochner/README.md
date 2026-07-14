# 01 · Spectral / Bochner: the OU Lorentzian, estimated two ways

Run: `julia --project experiments/01_spectral_bochner/run.jl`

## Concept

A stationary Ornstein–Uhlenbeck process with covariance `R(t) = (D/α)·e^{−α|t|}` has, by the
Wiener–Khinchin / Bochner theorem (Pavliotis Example 1.15), the **Lorentzian** spectral density

```
S(ω) = (1/2π) ∫ e^{−iωt} R(t) dt = D / (π (ω² + α²))     (two-sided)
```

This experiment synthesizes one long *exact* stationary OU record, estimates `S` from it two
independent ways — a **Welch** (averaged, windowed) periodogram and a **Bochner FFT** of the
estimated autocorrelation — and checks both against that closed form, plus the total-variance
normalization `∫ S dΩ = R(0) = D/α`.

## Why this check

`welch_psd`, `raw_periodogram`, `bochner_forward`/`spectral_power`, and
`sample_circulant_embedding` are the Unit-1 library primitives. This experiment exercises all of
them together end-to-end on a single stochastic object and checks that the estimated spectrum
matches theory in **both shape and total power** — not just that the code runs, but that the
numbers land where the Wiener–Khinchin theorem says they must. It is also the durable home of the
repository's **2π normalization lesson** and the **one-sided reporting convention** (below).

## What it proves

1. **Synthesis (circulant embedding, Route 4).** One `sample_circulant_embedding` draw of length
   `N_GRID = 2^17` is an *exact* stationary Gaussian record — no seam discontinuities, unlike
   concatenating short independent records. It is the unit's stationary synthesis route and the
   sole stochastic object here (one fixed seed).
2. **Two estimators, one truth.** Welch (50% Hann overlap) and Bochner-FFT of the biased ACF both
   overlay the analytic Lorentzian across the resolved band (`figures/psd_vs_lorentzian.png`).
3. **Gate (a) — normalization.** `|∫Ŝ dΩ / R(0) − 1| < 0.05`, computed with the library
   **`spectral_power`** (rectangular Parseval sum). See *Integrator split* below.
4. **Gate (b) — shape.** Resolved-band relative-L² error of the one-sided `Ŝ` **vs the one-sided
   density `2·S(ω)`** is below `3/√nseg`. Comparing the folded/doubled estimate to the *two-sided*
   `S` would be a clean 2× bug — see *One-sided reporting convention* below.
5. **Gate (c) — circulant faithfulness (deterministic).** Two conditions, both required: the
   covariance reconstructed from the circulant eigenvalues matches the analytic Toeplitz sequence to
   floating-point scale (`‖r_recon − r‖/‖r‖ < 1e-10`, an FFT identity that pins the even-extension
   convention), **and** every eigenvalue is non-negative (`min(λ) ≥ −1e-10·max(1,max|λ|)`) — the PSD
   precondition that makes the embedding an *exact* stationary covariance. The reconstruction alone
   would be a can't-fail round-trip; the eigenvalue check is what makes the gate bite (a spurious
   negative eigenvalue is exactly what an invalid embedding produces). Deterministic, not an
   empirical covariance — the honest reading of a floating-point-scale tolerance.
6. **Negative control (i) — inconsistency.** The raw (single-shot) periodogram is jagged and its
   variance does not shrink with record length (`figures/raw_vs_welch.png`); the printed per-bin
   roughness makes `raw ≫ welch` a number, normalized per-bin so it reflects genuine jaggedness,
   not raw's finer frequency grid.
7. **Negative control (ii) — the 2π lesson.** Dropping the `1/2π` in the density sends the variance
   ratio to `2π` instead of `1` (`figures/dropped_2pi.png`).

## The one-sided reporting convention

`bochner_forward` and `welch_psd` return **one-sided** spectra by default: folded onto `ω ≥ 0` with
every interior bin doubled (the DC bin is *not* doubled), so the discrete `∫Ŝ dΩ` still recovers the
full-line `R(0)`. Two consequences the gates must respect:

- **Shape (gate b)** compares to the **one-sided density `2·S(ω)`** at interior `ω`, *not* the
  two-sided `S(ω)`. A factor of 2 here is a convention error, never a constant to fudge.
- **Integral (gates a, ii)** targets `R(0)` directly — the doubling is exactly what makes the
  one-sided sum equal the two-sided integral.

## The integrator split (why `spectral_power`, not `spectral_variance`)

The library ships **two** integrators, each genuinely better in its regime, both CI-tested:

- `spectral_variance` — **trapezoidal**. The un-doubled DC bin gets weight ½, so it under-integrates
  DC-dominated / coarse-grid spectra (the Phase-1 paired test exhibits a 25% DC gap on a 4-point
  grid: trapezoid 7.5 vs rectangular 10). Use it for **smooth analytic** densities.
- `spectral_power` — **rectangular Parseval** (`dΩ · Σ Ŝ`). Counts the DC bin in full, so it is
  **unbiased (exact in expectation) at any grid resolution**. Use it for **discrete periodograms**.

Gate (a) uses `spectral_power`, so it carries **zero integrator bias**; the trapezoid's DC-halving
lesson lives in the Phase-1 paired test, not here. Gate (a)'s only residual is **Monte-Carlo
scatter** — see *Tolerance regime*.

## 50% Hann overlap (textbook Welch)

A Hann window tapers each segment's edges to zero, so `noverlap = 0` down-weights ~half the record;
50% overlap (`NOVERLAP = L÷2`) recovers that tapered data. Its real payoff is **gate (b)** — the
per-bin PSD estimate, where the tapered-edge waste matters. For **gate (a)** (total power) it helps
only modestly (~23% variance trim, not a halving), because total-power variance is a global
functional set by the record's integrated autocorrelation (∝ `1/N_GRID`). So gate (a)'s ~3σ margin
is carried by **`N_GRID = 2^17`**, not by overlap. The overlap path is asserted numerically in CI
(Phase-2 hand-segmentation test), so it is load-bearing, not run through untested.

## Recorded configuration

- **Seed:** `StableRNG(20240611)`, never the global RNG. One draw — the only stochastic object.
- **Process:** OU, `D = 1.0`, `α = 1.0`, so `R(0) = D/α = 1.0`.
- **Grid:** `DT = 0.05` (Nyquist `ω = π/DT ≈ 62.8`, well above `α`); `N_GRID = 2^17 = 131072`.
- **Welch:** `nseg = 64` → segment length `L = 2048`; `noverlap = L/2 = 1024`; `window = :hann`.
- **Bochner-ACF:** biased (`/N`) autocovariance out to `maxlag = round(12/(α·DT)) = 240` lags
  (~12 correlation lengths), then `bochner_forward`.
- **Resolved band:** `0 < ω ≤ OMEGA_MAX = 20.0`.
- **Gates:** (a) `|∫Ŝ dΩ/R(0) − 1| < 0.05` via `spectral_power`; (b) resolved-band rel-L² of the
  one-sided `Ŝ` vs `2·S(ω)` below `3/√nseg = 0.375`; (c) circulant faithfulness — reconstruction
  `< 1e-10` **and** non-negative eigenvalues `min(λ) ≥ −1e-10·max(1,max|λ|)`.
- **No jitter** (unlike the Unit-0 Cholesky sampler): circulant embedding is exact and the OU PSD
  precondition holds, so there is no nugget to report.

## Expected outcome

A dry run on Julia 1.10 prints (deterministic — the seed is fixed):

```
GATE (a) normalization: int Shat dOmega / R0 = 1.0016  -> PASS
GATE (b) shape L2: rel = 0.0752  vs  3/sqrt(nseg) = 0.3750  -> PASS
GATE (c) circulant faithfulness: ||r_recon - r|| / ||r|| = 6.149e-16, min(lambda) = 2.499e-02  -> PASS
CONTROL dropped-1/2pi: int (2pi*Shat) dOmega / R0 = 6.2931  (target 2pi = 6.2832)
CONTROL raw-vs-Welch per-bin roughness: raw=3.2927  welch=0.0087 (raw >> welch)
ALL GATES: PASS
```

- **Gate (a)** lands at `1.0016` — 0.16% above 1 at this seed. The *sampling* standard deviation of
  this ratio is ~1.7% at `N_GRID = 2^17`, so the 5% gate holds at ~3σ; this particular draw happened
  to land well inside 1σ. A FAIL is real breakage (a normalization/window-power regression), not an
  unlucky seed.
- **Gate (b)** `0.0752 ≪ 0.375` — a ~5× margin. The threshold `1/√nseg` is a *per-bin* SE scale used
  against an *aggregate* relative-L² error: a rough proxy, **not** a rigorous confidence statement
  like Unit-0's fitted-slope SE. With 50% overlap ~2× more segments are averaged than `nseg`, so the
  real per-bin SE is *below* `1/√nseg` — the proxy is **conservative** (the gate is, if anything,
  harder on itself). **A ~2× offset here would not be a tuning target — it would be the one-sided /
  two-sided convention bug: compare to `2·S(ω)`, not `S(ω)`.**
- **Gate (c)** reconstruction `~6e-16` (floating-point round-off of the FFT/IFFT identity) and
  `min(λ) = 0.025 > 0` (all circulant eigenvalues non-negative — the OU embedding is a valid,
  exact stationary covariance). A negative `min(λ)` would flag an invalid embedding.
- **Control (ii)** lands on `2π = 6.2832` (realized `6.2931`), not 1 — the signature of a dropped
  `1/2π`.

## Tolerance regime

Monte-Carlo for gates (a) and (b): each is a small multiple of the estimator's own scatter/SE, not a
fixed absolute tolerance. Gate (a)'s 5% budget is sized (via `N_GRID = 2^17`) to sit at ~3σ of the
integrator's ~1.7% scatter; the lever for that margin is `N_GRID` (sd ∝ 1/√N_GRID), **not** loosening
the 5% threshold. Gate (c) is deterministic at floating-point scale (`< 1e-10`).

## Not run in CI

This experiment is Monte-Carlo and is not part of `test/runtests.jl` or CI. Run it locally; the four
figures below are committed artifacts. The deterministic identities it relies on are covered by the
Phase 1–3 testsets.

## Figures

- `figures/psd_vs_lorentzian.png` — log–log Welch and Bochner-ACF estimates overlaid on the dashed
  one-sided Lorentzian `2D/(π(ω²+α²))` across the resolved band; both track it.
- `figures/raw_vs_welch.png` — the inconsistency made visible: raw periodogram jagged over many
  decades, Welch smooth and consistent.
- `figures/dropped_2pi.png` — two bars, `∫Ŝ/R(0) ≈ 1` (kept) vs `∫2π·Ŝ/R(0) ≈ 2π` (dropped), with
  dashed references at 1 and 2π: the 2π lesson.
- `figures/circulant_cov_error.png` — reconstructed vs analytic covariance sequence over the first
  ~60 lags; the two are indistinguishable (err ~1e-16).

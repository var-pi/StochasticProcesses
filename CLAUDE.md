# CLAUDE.md

Guidance for agents working in this repo. For the project's purpose and gallery, see `README.md`.
For the through-line, the Unit 0–7 feature briefs, and the design rationale, see the master plan
`docs/plan/pavliotis_ch1_project_plan_3.tex` (master-plan altitude: no call signatures or
tolerances — those live in the code and the per-commit plans).

## What this is

A computational companion to Pavliotis, *Stochastic Processes and Applications*, Ch. 1
(`StochasticProcesses.jl`, Julia ≥ 1.10). Each **unit** adds one self-contained vertical slice — a
type/algorithm from the text plus a numerical experiment that checks it against a known analytic
result — building a gallery of reproducible stochastic-process demos.

**Current state:** Units 0 (covariance core + Cholesky sampling), 1 (spectral/Bochner: the
`spectral.jl` core, Welch/raw periodogram estimators, circulant-embedding sampler, and the
`01_spectral_bochner` experiment), 2 (Karhunen–Loève), 3 (process zoo / reconciliation), 4
(ergodicity / loop-closer), and 5 (random-walk → BM scaling limit) are complete. Unit 2 comprises five commits: `periodic_kernel` (torus contrast kernel), the `kl.jl`
module (symmetrized Nyström eigenproblem), the `sample_kl` KL-truncation sampler, the
`02_kl_quadrature` experiment, and its READMEs. Unit 3 comprises six commits: `brownian_bridge_kernel`
(a catalogue kernel), the `gof.jl` module (`ks_statistic` — deliberately the one `src/` module
that is *not* an operation on the covariance operator; a shared goodness-of-fit utility reused by
the future Unit 5), and the `03_process_zoo` experiment across three phases (route equivalence via a
split-half bootstrap null; the two-step distributional identities — Gaussianity by construction →
App. B.5 → Cramér–Wold, plus the KL-coefficient independence check; and the Toeplitz/Szegő
cross-check against the analytic un-normalized symbol `2π·S`), plus its READMEs. Unit 4 comprises
four commits: the `ergodic.jl` module (the path-side time-average estimators `running_time_average`,
`time_average_variance`, `mean_square_displacement`; the Green–Kubo coefficient `green_kubo`
(`D*=D/α²`); and the exact Lemma-1.17 finite-T variance identity `time_average_variance_exact`), the
`04_ergodicity` experiment (the variance-vs-T `1/T` slope and its `2D*` constant, gated against
independent sub-ensemble slope SEs), its non-integrable-`C` control (the falsifier showing
Prop. 1.16's `L¹` hypothesis is load-bearing), and its READMEs. Unit 5 comprises four experiment-only
commits (no new `src/` code — it reuses `gof.jl`'s `ks_statistic` and `empirical_cov`): the
`rescaled_walk` lattice builder + the three increment-law samplers and the two-time
covariance→`min(s,t)` MC-rate gate; the headline KS-vs-`n` marginal-rate plot with **hybrid** gating
(MC stochastic `−½` for exponential/skewness and Rademacher/lattice-discreteness, exact-Irwin–Hall
deterministic `−1` for uniform/smooth-symmetric — a **three-mechanism** story that corrects the brief,
which had grouped Rademacher with the `n⁻¹` laws); the running-maximum→half-normal functional; and the
infinite-variance (symmetric-Pareto, `γ=1.5`) falsifier. `05_bm_scaling_limit/` is the first experiment
with **no** `src/` or `test/` change — its verification is entirely the experiment's Monte-Carlo plus
deterministic exact-CDF gates. Units 6–7 are planned but not yet implemented — do not pre-stub them.

## Architecture

The central design decision is a split between a small, tested **library** and a narrative
**experiment gallery**:

- `src/` — the library, organized **by operation on the covariance operator** (kernels, the
  diagonalizations, the square roots), *not* by named process. A process is just a choice of kernel.
  - `StochasticProcesses.jl` — top-level module; grows by one `include` + re-export per phase.
  - `kernels.jl` (`Kernels`) — `brownian_motion_kernel`, `exponential_kernel` (OU),
    `periodic_kernel` (torus, Unit 2), `brownian_bridge_kernel` (Unit 3).
  - `gaussianprocess.jl` (`GaussianProcesses`) — `GaussianProcess`, `assemble_cov`,
    `assemble_mean`, `empirical_cov`.
  - `sampling.jl` (`Sampling`) — `sample_cholesky(Σ, rng; jitter=1e-10)`,
    `sample_circulant_embedding(r, rng)`, `sample_kl(lambdas, eigfuncs, rng)` (Unit 2).
  - `spectral.jl` (`Spectral`) — `bochner_forward`, `spectral_variance`, `spectral_power`,
    `welch_psd`, `raw_periodogram` (public); `_onesided`, `_sorted_by_omega`, `_raw_transform`,
    `bochner_inverse` (private helpers).
  - `kl.jl` (`KL`, Unit 2) — `quad_nodes_weights`, `nystrom_eigen` (symmetrized: solves
    `W^{1/2}KW^{1/2}g=λg`, `e=W^{-1/2}g`), `trace_diag`, `kl_tail_energy`.
  - `gof.jl` (`GOF`, Unit 3) — `ks_statistic(samples, cdf)` (KS sup-distance to a target CDF).
    The one module *not* organized as an operation on the covariance operator — a deterministic-
    tested goodness-of-fit utility, shared with Unit 5.
  - `ergodic.jl` (`Ergodic`, Unit 4) — `running_time_average`, `time_average_variance`,
    `mean_square_displacement` (path-side time-average estimators over an `n_grid × N` path matrix);
    `green_kubo` (`D*=∫₀^∞ C = D/α²`); `time_average_variance_exact` (the exact Lemma-1.17 finite-T
    variance identity, O(n)).
- `test/runtests.jl` — deterministic analytic identities, tight tolerances; grows by one testset
  per phase. This is what CI runs.
- `experiments/NN_name/` — the gallery, **one folder per unit**, each with `run.jl`, `README.md`,
  and committed `figures/`. Currently `00_covariance_core/`, `01_spectral_bochner/`,
  `02_kl_quadrature/` (Unit 2), `03_process_zoo/` (Unit 3), and `04_ergodicity/` (Unit 4). The
  `experiments/` folder carries its own committed `Project.toml` + `Manifest.toml` (a shared env
  that `dev`s the package) — run scripts under `--project=experiments`.
- `docs/plan/` — the master plan (`.tex` source + tracked compiled PDF).

## Commands

- Run the test suite (what CI runs): `julia --project -e 'using Pkg; Pkg.test()'`
- Instantiate the pinned environment: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- Run an experiment locally (writes figures, prints the slope gate — **not** run in CI):
  `julia --project=experiments experiments/00_covariance_core/run.jl` (experiments have their own
  env that `dev`s the package — see the Environment convention below)

## Non-negotiable conventions

These are load-bearing for reproducibility; violating one silently corrupts every downstream number.

- **RNG:** `StableRNGs.StableRNG(seed)` only — **never the global RNG / bare `randn()`** (the
  default stream is not stable across Julia versions). Pass an explicit `StableRNG(seed)` to every
  stochastic routine and **record the seed** (in the experiment's `run.jl` consts + README).
- **Cholesky nugget:** always factor `Σ + εI` and **report `ε`**. Too small → indefinite throw; too
  large → biased covariance. Default `jitter = 1e-10`.
- **CI vs. local:** deterministic tests run in CI (`Pkg.test()` only); Monte-Carlo experiments stay
  out of CI — run them locally and **commit their figures**.
- **Environment:** `Manifest.toml` is intentionally committed — do not gitignore it. This holds for
  *both* environments: the root package env, and the separate `experiments/` env
  (`experiments/Project.toml` + `experiments/Manifest.toml`), which `dev`s the package via a
  relative `path = ".."` so experiment scripts can `using StochasticProcesses` as a real dependency
  (this is also what makes the editor's LanguageServer resolve the package instead of flagging
  "Missing reference"). Run experiments with `--project=experiments`. Likewise
  `experiments/**/figures/` are committed artifacts.
- **Headless plotting:** experiments set `ENV["GKSwstype"] = "100"` before `gr()` so figures render
  with no display (CI/agent shells).

## Testing conventions (two tiers)

- **`test/` — deterministic.** Analytic identities (kernel symmetry/PSD, closed-form Σ entries)
  with tight absolute tolerance. Tests must **bite**: prefer hand-computed targets over re-running
  the function's own formula, and non-square fixtures so shape/orientation bugs cannot hide.
- **`experiments/` — Monte-Carlo.** The headline artifact of each unit is a **log–log slope** (e.g.
  `‖Σ̂_N − Σ‖_F ∝ N^{-1/2}`). Gate a *stochastic* slope against theory within a small multiple
  (2–3×) of the fitted slope's own standard error — never a fixed absolute tolerance. Seed so the
  result is deterministic given the seed.

## Gotchas specific to this code

- **`paths` orientation is `n_grid × N` — one sample path per COLUMN.** A transpose silently
  estimates the wrong matrix; `empirical_cov` returns `n_grid × n_grid`, not `N × N`.
- **`sample_cholesky` samples the ZERO-MEAN law** `X = L·z` and deliberately ignores
  `meanfn`/`assemble_mean` (Unit 0 is zero-mean only). A future non-zero-mean unit must add the
  mean explicitly: `X = assemble_mean(gp, t_grid) .+ L·z`.
- **Do not reorder RNG draws** in an experiment: the demo paths and the N-ladder pull from the same
  `StableRNG` stream, so reordering changes every committed number and figure.
- **`brownian_motion_kernel` is both the main check and the singular negative control;**
  `exponential_kernel` is well-conditioned and is the shipped OU example, *not* a control.

## Roadmap (planned units, `experiments/NN_*`)

0 covariance core (done) · 1 spectral/Bochner (done) · 2 Karhunen–Loève (done) ·
3 process zoo (done) · 4 ergodicity (done) ·
5 random-walk → BM (done) · 6 fractional BM · 7 SDE bridge. `src/` accretes modules; existing files are
not restructured. Do not implement a unit until its phase-plan is being worked.

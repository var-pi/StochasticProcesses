# CLAUDE.md

Guidance for agents working in this repo. For the project's purpose and gallery, see `README.md`.
For the full design rationale and Unit 0–7 spec, see `docs/plan/pavliotis_ch1_project_plan_3.tex`.

## What this is

A computational companion to Pavliotis, *Stochastic Processes and Applications*, Ch. 1
(`StochasticProcesses.jl`, Julia ≥ 1.10). Each **unit** adds one self-contained vertical slice — a
type/algorithm from the text plus a numerical experiment that checks it against a known analytic
result — building a gallery of reproducible stochastic-process demos.

**Current state:** Unit 0 (covariance core + Cholesky sampling) is complete. Units 1–7 are planned
but not yet implemented — do not pre-stub them.

## Architecture

The central design decision is a split between a small, tested **library** and a narrative
**experiment gallery**:

- `src/` — the library, organized **by operation on the covariance operator** (kernels, the
  diagonalizations, the square roots), *not* by named process. A process is just a choice of kernel.
  - `StochasticProcesses.jl` — top-level module; grows by one `include` + re-export per phase.
  - `kernels.jl` (`Kernels`) — `brownian_motion_kernel`, `exponential_kernel` (OU).
  - `gaussianprocess.jl` (`GaussianProcesses`) — `GaussianProcess`, `assemble_cov`,
    `assemble_mean`, `empirical_cov`.
  - `sampling.jl` (`Sampling`) — `sample_cholesky(Σ, rng; jitter=1e-10)`.
- `test/runtests.jl` — deterministic analytic identities, tight tolerances; grows by one testset
  per phase. This is what CI runs.
- `experiments/NN_name/` — the gallery, **one folder per unit**, each with `run.jl`, `README.md`,
  and committed `figures/`. Currently only `00_covariance_core/`.
- `docs/plan/` — the master plan (`.tex` source + tracked compiled PDF).

## Commands

- Run the test suite (what CI runs): `julia --project -e 'using Pkg; Pkg.test()'`
- Instantiate the pinned environment: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- Run an experiment locally (writes figures, prints the slope gate — **not** run in CI):
  `julia --project experiments/00_covariance_core/run.jl`

## Non-negotiable conventions

These are load-bearing for reproducibility; violating one silently corrupts every downstream number.

- **RNG:** `StableRNGs.StableRNG(seed)` only — **never the global RNG / bare `randn()`** (the
  default stream is not stable across Julia versions). Pass an explicit `StableRNG(seed)` to every
  stochastic routine and **record the seed** (in the experiment's `run.jl` consts + README).
- **Cholesky nugget:** always factor `Σ + εI` and **report `ε`**. Too small → indefinite throw; too
  large → biased covariance. Default `jitter = 1e-10`.
- **CI vs. local:** deterministic tests run in CI (`Pkg.test()` only); Monte-Carlo experiments stay
  out of CI — run them locally and **commit their figures**.
- **Environment:** `Manifest.toml` is intentionally committed — do not gitignore it. Likewise
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

## Workflow (from the agent working agreement)

- **One phase = exactly one git commit.** A phase is an independently-verifiable increment that
  leaves the package loadable with green tests and introduces nothing a later phase depends on.
  Commit only when asked; the commit message names the phase and the pass conditions verified.
- **TDD with a mutation gate.** Write tests first; run them against the *unimplemented* feature — if
  a test passes before the feature exists it is vacuous, so rewrite it to fail. Then implement and
  fix-until-green. A test that stays green no matter what the code does is noise.
- **Every feature ships a negative control** — a test that is *supposed* to fail, demonstrating a
  load-bearing hypothesis. Unit-0 exemplar: the *same* singular Brownian-motion Σ (all-zero `t=0`
  row, `R(0,s)=0`) throws `PosDefException` at `jitter = 0` and succeeds once `ε ≳ 1e-10` — proving
  the nugget matters for the real matrix, not a contrived one.
- **READMEs are the durable evidence of understanding.** Each `experiments/NN_*/README.md` must
  state the concept, why the check is the right one, what it proves, the recorded config (seed,
  jitter, grid), and the expected numeric outcome — treat it as a first-class deliverable.
- **Code quality.** Every variable is self-explanatory or carries a comment. Let docstrings carry
  the *why* (the physics, the convention, why a jitter is honest not cosmetic).

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

0 covariance core (done) · 1 spectral/Bochner · 2 Karhunen–Loève · 3 process zoo · 4 ergodicity ·
5 random-walk → BM · 6 fractional BM · 7 SDE bridge. `src/` accretes modules; existing files are
not restructured. Do not implement a unit until its phase-plan is being worked.

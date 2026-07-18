# Unit 3 Phase 6 — process-zoo README, gallery row, CLAUDE.md

The dedicated documentation commit for Unit 3 (the "process zoo"). It closes the feature: it adds
the experiment's narrative README, the root-README gallery row, and brings CLAUDE.md's written
record into step with the five code commits (C1–C5) that precede it.

## Why this is its own commit, landing last

README documentation describes the feature *as a whole*, and its final shape settles only once
every commit's contract does — so it is not smuggled into any single code commit. It is authored
last, once every gate number is final and every figure is committed, so the prose can quote the
real run rather than a prediction. This mirrors how Unit 2 closed (a dedicated README commit plus
a separate CLAUDE.md "mark complete" step); here the CLAUDE.md update is folded into this same
docs commit, consistent with the repo convention that CLAUDE.md changes are version-controlled
(cf. Unit 2's `52300dc`).

## What it contains

- **`experiments/03_process_zoo/README.md` (new).** The narrative, modeled on
  `experiments/02_kl_quadrature/README.md`. It does not merely assert the unit's conclusions — it
  spells out the load-bearing reasoning:
  - *Route equivalence*: why `‖Σ̂_A − Σ̂_B‖_F` concentrates at `√2σ₁ > 0` (a norm over `n_grid²`
    entries), so the gate is calibrated against an empirical split-half **bootstrap** null band, not
    a zero-centred SE; the `√2` rescale placing the full-`N` cross statistic and the half-`N` null on
    the same `2σ₁` scale; and the honest ~1/8-seed false-failure of a 95%-band AND-gate over three
    correlated pairs.
  - *Distributional identity* (the heart): the three-step licensing chain — (1) `c^{-1/2}W(ct)` is
    Gaussian **by construction** (a deterministic linear map of a Gaussian process — a proof, not a
    sample-tested claim); (2) Appendix B.5 upgrades a full-covariance match to equality in law; (3)
    the Cramér–Wold projections catch the non-Gaussian impostor that a covariance match cannot,
    supplying the Gaussianity evidence B.5 requires but does not itself provide. Plus the
    KL-coefficient check (`ξ_k ~ N(0,λ_k)` exactly; uncorrelated + jointly Gaussian ⇒ independent).
  - *Cross-check*: the operator eigenvalues converge to the **un-normalized** symbol
    `R̂(ω)=2π·S(ω)=2D/(α²+ω²)`, NOT the `1/2π` density `S` (Unit 2's `λ_k=R̂(k)` anchor); the
    resolved-bulk gap (excludes the ω→0 edge and the noise floor, never `max_k`); the deterministic
    fixed-margin slope gate (−0.89 < −0.5, exponent reported not claimed); and the Welch overlay on
    one-sided `2·S` as pedagogy, not the gate.
  - The wrong-exponent (`c^{-1/3}`) negative control and the null's own calibration guard, and the
    full recorded configuration (three independent seeds and their roles).
- **`README.md` (root): the Unit 3 gallery row**, after the Unit 2 row, matching the table's
  three-column format.
- **`CLAUDE.md`**: "Current state" now lists Units 0–3 complete and describes Unit 3's six commits;
  the `src/` inventory gains `gof.jl` (`GOF` / `ks_statistic`, the one non-`𝒞`-operation module) and
  `brownian_bridge_kernel` in the `kernels.jl` entry; the experiment gallery lists `03_process_zoo/`;
  the roadmap line reads `3 process zoo (done)`.

## Verification

- `julia --project -e 'using Pkg; Pkg.test()'` — green at 146/146 (this commit touches no
  `src/`/`test/` code).
- Every gate number pasted into the README matches the committed `run.jl` output (deterministic
  given the three recorded seeds); all nine `figures/*.png` links resolve to committed files.

## Note on authorship

The experiment README was drafted by a `commit-plan-implementer` subagent that was interrupted by a
session limit before committing; the coordinator finished the remaining docs (gallery row,
CLAUDE.md, this commit doc), fixed one mojibake in the README, verified the numbers and figure
links, and made the commit.

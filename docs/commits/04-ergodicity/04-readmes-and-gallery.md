# Commit 4 — READMEs, gallery row, CLAUDE.md (Unit 4 "ergodicity", feature `04-ergodicity`)

## TL;DR

Unit 4's outward-facing documentation, closing the unit. This is a **docs-only** commit —
no `src/`, no `test/`, no experiment code changed. Commits 1–3 (`00a233a` the `ergodic.jl`
module, `03e2d9a` the main variance-vs-T experiment, `075ade5` the non-integrable control)
already landed the code, the `Ergodic` testset, and the four committed figures. This commit
adds the showcase `experiments/04_ergodicity/README.md`, inserts the Unit-4 gallery row into
the root `README.md`, and updates `CLAUDE.md` to mark Unit 4 complete. It mirrors the Unit-3
finalization precedent (`5ea0089`, "process-zoo README, gallery row, CLAUDE.md"): a dedicated
final phase whose only job is documentation.

Nothing enters CI — no code diff exists. The numbers quoted throughout the new docs are the
real committed run's console output, verified byte-for-byte.

## What changed

Four files: one new experiment README, two edited repo-level docs, and this doc.

### `experiments/04_ergodicity/README.md` (new) — the showcase

Authored by the `feature-readme-writer` subagent (this repo's standing protocol for feature
READMEs), following the `03_process_zoo/README.md` skeleton. Its spine:

- A **loop-closer hook** — every prior unit compared *one* Monte-Carlo path against a
  closed-form target and called a match success; that move is only legitimate if a single
  long path's time-average sees the whole distribution. Prop. 1.16 is the theorem that
  licenses it. Unit 4 measures the rate at which the loop closes and then breaks the
  hypothesis on purpose.
- **`## The result`** — the `variance_vs_T.png` figure plus the real console block quoted
  verbatim, ending `ALL GATES: PASS`.
- **`## Concept`** — Prop. 1.16 (L² LLN for stationary processes with `C ∈ L¹`), Lemma 1.17's
  exact finite-T identity `Var(A_T) = (2/T²)∫₀ᵀ(T−u)C(u)du → 2·D*/T`, and the Green–Kubo
  coefficient `D* = ∫₀^∞ C`. The `D` vs `D*` caution is a blockquote callout: for the OU
  correlation `R(0) = D/α` but `D* = D/α²`, equal only at `α = 1`; the experiment runs at
  `α = 2` precisely so `2·D* = 0.5` sits a clean factor of two below `2·R(0) = 1.0`.
- Sections for **gate (a)** (the `1/T` slope, plus the sub-ensemble-SE subtlety — nested-prefix
  ladder points are autocorrelated, so OLS residual SE under-reports by ~3×; the honest SE
  comes from `NGROUP = 20` disjoint sub-ensembles), **gate (b)** (the plateau-median constant
  check pinning `α²`), the **MSD diagnostic** (not a gate — an exact algebraic identity, so its
  slope is mechanically gate (a)'s `+2`), and the running-average-band figure.
- **`## Negative control`** — the non-integrable `C(u)=(1+u)^{-1/2}` falsifier (valid PSD by
  Pólya's criterion, but not in L¹), the slope landing near `-0.41` rather than the true
  asymptotic `-1/2` (slow finite-T convergence), and gate (c)'s two parts (c1 tracks the exact
  Lemma-1.17 curve for this same `C`; c2 confirms the slope is well above `-1`).
- **`## Recorded configuration`** — seeds/params and the standard Monte-Carlo boilerplate
  (run locally, not in CI, figures committed, deterministic pieces covered by the `Ergodic`
  testset).

### `README.md` (root, edited) — gallery row

One row inserted after the Unit-3 row, matching the existing 0–3 format (topic column +
headline-result column). Quotes slope **−0.993**, constant median `T·Var` **0.495** vs 0.500
(pinning the `α²`, not `α`, dependence), the integrated MSD growing as `2D*t`, and the control
slope **−0.41** breaking the rate.

### `CLAUDE.md` (edited) — five pinned edits, applied verbatim

1. The **Current state** paragraph now lists Unit 4 (ergodicity / loop-closer) as complete
   alongside 0–3.
2. The Unit-3 paragraph's tail sentence expands into a new **Unit-4 paragraph** naming its four
   commits (the `ergodic.jl` module, the main experiment, the non-integrable control, the
   READMEs), and the "planned but not implemented" disclaimer moves to Units 5–7.
3. A new **`ergodic.jl` (`Ergodic`, Unit 4)** bullet appended to the `src/` module list, naming
   its five functions (`running_time_average`, `time_average_variance`,
   `mean_square_displacement`, `green_kubo`, `time_average_variance_exact`).
4. The experiments-list sentence now names **`04_ergodicity/`** (Unit 4).
5. The Roadmap line changes `4 ergodicity ·` → `4 ergodicity (done) ·`.

All five were pinned verbatim in the commit plan and applied exactly — no paraphrasing.

## Verification performed

- `julia --project -e 'using Pkg; Pkg.test()'` → **177/177 green**, confirming nothing under
  `src/` or `test/` was touched (docs-only, suite unaffected by design).
- Re-ran `experiments/04_ergodicity/run.jl` and confirmed the console output reproduces the
  README's quoted numbers byte-for-byte: `Green-Kubo D* = 0.250208`, gate (a) slope `-0.9926`
  (`|slope+1| = 0.0074 < 2.5·SE = 0.0164`), gate (b) `median(T·Var) = 0.49481` vs `2D* =
  0.50042`, MSD diagnostic slope `1.0074`, control `MC -0.4092` vs `exact -0.4207`, ending
  `ALL GATES: PASS`. Deterministic given the seeds (`20260718` main, `13579` control).
- All four figure links in the new README (`variance_vs_T.png`, `msd_vs_t.png`,
  `running_average_band.png`, `nonintegrable_control.png`) resolve to files committed in
  commits 2–3.
- Manually diffed `CLAUDE.md` and root `README.md` against the plan's pinned edit blocks —
  each edit applied verbatim.

No `/code-review` was run: there is no code diff to review, matching the Unit-3 docs-commit
precedent (`5ea0089`).

## Deviations from plan

None. The plan pinned the `CLAUDE.md` edits and the `README.md` gallery row verbatim and
specified delegating the experiment README to `feature-readme-writer` with a brief; all was
followed exactly. The subagent noted two small self-reconciliations — using `+/- 0.0023` for
the MSD diagnostic SE in prose to match the quoted block, and phrasing gate (b)'s `rel 0.0112`
as "1.1%" in prose — both harmless, no numbers altered, no gaps or defects flagged.

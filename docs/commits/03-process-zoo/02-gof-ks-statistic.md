# Commit 2 — `ks_statistic` (Unit 3 "process zoo", feature `03-process-zoo`)

## TL;DR

Adds one pure function, `ks_statistic(samples, cdf)`, the Kolmogorov–Smirnov
sup-distance between the empirical CDF of `samples` and a target CDF `cdf`, inside a
**brand-new `GOF` submodule** (`src/gof.jl`). Plus its export/include/re-export wiring in
`src/StochasticProcesses.jl`, a one-line addition to that module's docstring, and a
five-assertion deterministic testset. No experiment, no new type, no change to any
existing behavior — a strictly additive utility commit. The test suite goes from 140 to
146 passing tests: 5 new `@test`s in the KS testset plus 1 more from extending the
`public surface` regression-guard tuple by one symbol. **No deviations from the plan** —
the reference implementation was specified verbatim and used verbatim; only the docstring
was expanded (which the plan explicitly permitted).

`GOF` is the first `src/` module that is *not* organized "by operation on the covariance
operator" — a deliberate, documented architectural exception explained below.

---

## Background: the terms you need

If you already know KS statistics you can skip to "What changed". This section is here so
a reader who knows basic probability but not this specific statistic never has to tab away.

- **Empirical CDF (ECDF).** Given a sample `x_1, ..., x_n`, sort it into order statistics
  `x_(1) ≤ x_(2) ≤ ... ≤ x_(n)`. The empirical CDF is the step function

  ```
  F_n(x) = (number of samples ≤ x) / n
  ```

  It starts at `0` below `x_(1)`, and **jumps up by exactly `1/n` at each order
  statistic**, reaching `1` at and above `x_(n)`. Between consecutive order statistics it
  is flat. It is the sample's own estimate of the true distribution function.

- **Target CDF `F`.** A callable `x -> F(x)` giving the distribution you are testing the
  sample *against* — e.g. `x -> x` is the CDF of `Uniform(0, 1)`; a Gaussian CDF is
  `x -> cdf(Normal(μ, σ), x)`. Goodness-of-fit asks: does the sample look like it was
  drawn from `F`?

- **Kolmogorov–Smirnov (KS) statistic.** The single number

  ```
  D_n = sup_x |F_n(x) - F(x)|,
  ```

  the largest vertical gap anywhere between the empirical and target CDFs. It is the
  classical distribution-free goodness-of-fit distance: small `D_n` means the sample's
  ECDF hugs `F` everywhere; a large `D_n` is evidence the sample is not from `F`. Because
  both `F_n` and `F` are non-decreasing and `F` is continuous, the sup is always attained
  *at one of the order statistics* — so we never need to search a continuum, only check
  the `n` jump locations.

- **Why both sides of each jump must be checked.** This is the one subtle point. `F_n`
  has a **jump discontinuity** at each `x_(i)`: just below it, `F_n = (i-1)/n`; at and
  above it, `F_n = i/n`. The target `F(x_(i))` sits somewhere in that interval, and the
  supremum of `|F_n - F|` near that jump can be realized on **either** side of the jump:

  - the **upper gap** `i/n - F(x_(i))` — `F_n` just *after* the jump minus the target;
  - the **lower gap** `F(x_(i)) - (i-1)/n` — the target minus `F_n` just *before* the jump.

  A correct KS statistic takes the max over **both** gaps at **every** order statistic. An
  implementation that only checks one side (the "D+"-only or "D−"-only one-sided statistic)
  silently returns a too-small answer whenever the true sup happens to live on the other
  side — a bug that a symmetric test fixture cannot detect, but a skewed one can. The two
  "bunched" tests below are exactly that skewed pair.

---

## What changed

Three files, all additive.

### 1. `src/gof.jl` — the new module (whole file, 54 lines)

A brand-new file defining `module GOF`, exporting `ks_statistic`, and implementing it:

```julia
function ks_statistic(samples, cdf)
    n = length(samples)
    n == 0 && throw(ArgumentError("ks_statistic: empty sample"))
    xs = sort(samples)
    d = 0.0
    for (i, x) in enumerate(xs)
        Fx = cdf(x)
        d = max(d, i / n - Fx, Fx - (i - 1) / n)   # both sides of the jump
    end
    return d
end
```

The body is a single pass over the sorted sample: for each order statistic it folds both
the upper gap `i/n - Fx` and the lower gap `Fx - (i-1)/n` into a running max. It sorts
internally (so callers need not pre-sort) and throws `ArgumentError` on empty input (the
sup over zero order statistics is undefined). The docstring spells out the ECDF, both
gaps, the both-sides rationale, and points at the skewed-fixture tests.

The file opens with a prominent **architectural-exception preamble comment** (lines 1–14)
— see "Why these design choices" for its verbatim rationale.

### 2. `src/StochasticProcesses.jl` — the wiring

Two edits:

- **The include/using/export block** (lines 44–46) appended at the end, symmetric with
  every other submodule:

  ```julia
  include("gof.jl")
  using .GOF
  export ks_statistic
  ```

  This is the second half of the export pair: `ks_statistic` is exported from *both*
  `GOF` (in `gof.jl`) and the umbrella module here, or it will not resolve for a
  `using StochasticProcesses` user.

- **The module docstring's submodule bullet list** (lines 16–17) gained one line naming
  `GOF` as "the one deliberate exception: goodness-of-fit statistics (shared across units,
  not an operation on the covariance operator)". This keeps the docstring's own inventory
  of submodules accurate — see "Deviations from plan" for why this minor addition is
  noted.

### 3. `test/runtests.jl` — the new testset + surface guard

A new `@testset "KS statistic, hand-computed"` (lines 537–577) with five assertions, plus
`:ks_statistic` added to the `public surface` regression-guard tuple (line 588). Detailed
below.

### How it composes (control flow)

There is no new control flow to trace and nothing downstream consumes `ks_statistic`
*yet* — it is a leaf utility built ahead of its first callers (Unit 3's Cramér–Wold /
KL-coefficient checks and Unit 5's random-walk-to-BM checks, neither built). It takes a
vector and a callable and returns a `Float64`; no RNG, no covariance machinery, no state.

---

## Why these design choices (restating the planner's pre-resolved decisions)

- **`GOF` is a deliberate architectural exception, stated plainly rather than
  rationalized away.** Every other `src/` module is organized *by operation on the
  covariance operator* `C` — kernels, its diagonalizations (`Spectral`, `KL`), its square
  roots (`Sampling`) — the central design principle in `CLAUDE.md`'s Architecture section
  and this package's module docstring. `ks_statistic` has *nothing* to do with `C`. The
  preamble comment in `src/gof.jl` (lines 1–14) states the exception outright and gives
  two concrete reasons it lives in `src/` anyway rather than being stretched to fit the
  "operation on `C`" framing or exiled to `experiments/`:

  1. **Two separate units need it**, so it belongs to neither on its own: Unit 3's
     Cramér–Wold / KL-coefficient goodness-of-fit checks (not yet built) and Unit 5's
     random-walk-to-BM convergence checks (not yet built). A shared utility used across
     units is library code, not experiment code.
  2. **It is deterministic and hand-computable**, which is *exactly* the guard the
     `src/`-vs-`test/` split exists to provide (`CLAUDE.md`'s two-tier testing convention:
     the `test/` tier is for deterministic analytic identities with tight tolerances). A
     Monte-Carlo-only utility would belong in `experiments/`; a hand-testable one belongs
     in `src/` with a biting deterministic testset.

- **Two-argument `ks_statistic(samples, cdf)`, not a samples-only version against a fixed
  target.** A convenience signature that tested only against, say, `Uniform(0,1)` or a
  fixed standard normal would not serve the actual callers. Unit 3's Cramér–Wold projection
  test compares a projected sample against `N(0, aᵀΣa)` — a target whose variance depends
  on the projection direction `a` and the covariance `Σ`, i.e. a *different* CDF on every
  call. So the target CDF must be a caller-supplied argument. The two-argument form is the
  minimum general enough to serve both future units.

---

## Every test, individually (and why it bites)

All five new assertions live in `@testset "KS statistic, hand-computed"`. The repo's
convention (`CLAUDE.md`) is that tests must *bite*: hand-computed targets over re-running
the function's own formula, and fixtures chosen so one-sided/orientation bugs cannot hide.
Every gap below is worked out independently from `F(x) = x` on `Uniform(0, 1)`.

1. **Symmetric case: sup = 7/30** — `ks_statistic([0.1, 0.5, 0.9], x->x) ≈ 7/30`.
   Sorted `xs = [0.1, 0.5, 0.9]`, `n = 3`:
   - `i=1, x=0.1`: upper `1/3 − 0.1 = 7/30 ≈ 0.2333`; lower `0.1 − 0 = 3/30`
   - `i=2, x=0.5`: upper `2/3 − 0.5 = 1/6`; lower `0.5 − 1/3 = 1/6`
   - `i=3, x=0.9`: upper `1 − 0.9 = 1/10`; lower `0.9 − 2/3 = 7/30`

   The max over all six gaps is `7/30 ≈ 0.23333`, attained at the upper gap of `i=1` *and*
   the lower gap of `i=3`. This pins the basic arithmetic and the `1/n`-per-jump structure.
   Note it is deliberately **symmetric**: because the sup is attained on both an upper and
   a lower gap, this fixture *alone cannot distinguish* an upper-only from a lower-only
   implementation — both would still return `7/30`. That is why it is not sufficient on its
   own and the bunched pair below exists. A buggy off-by-one on the jump index (using `i` vs
   `i-1`) would drift this off `7/30`.

2. **Unsorted input guard** — `ks_statistic([0.9, 0.1, 0.5], x->x) ≈ 7/30`. The same
   multiset fed out of order. It must return the *same* `7/30`, proving the internal
   `sort` runs. If `ks_statistic` forgot to sort, the `i/n` and `(i-1)/n` bounds would
   attach to the wrong order statistic — e.g. pairing the `i=1` bound `1/3` with `x=0.9`
   gives `1/3 − 0.9 < 0` and the `i=3` bound with `x=0.5` gives `1 − 0.5 = 0.5` — and the
   result would drift well away from `7/30` (an unsorted pass here computes `0.5`, not
   `7/30`). This is a shape/ordering bug the symmetric value alone would not surface.

3. **Bunched left: the UPPER gap attains the sup = 0.8** — `ks_statistic([0.1, 0.2], x->x)
   ≈ 0.8`. `xs = [0.1, 0.2]`, `n = 2`:
   - `i=1, x=0.1`: upper `1/2 − 0.1 = 0.4`; lower `0.1 − 0 = 0.1`
   - `i=2, x=0.2`: upper `2/2 − 0.2 = 0.8`; lower `0.2 − 1/2 = −0.3`

   Sup `= 0.8`, the **upper** gap at `i=2`. A **D−-only** (lower-gap-only) implementation
   would compute `max(0.1, −0.3) = 0.1`, missing the true sup entirely. This fixture pins
   down the upper side: it is only correct if the upper gap is actually checked.

4. **Bunched right (mirror): the LOWER gap attains the sup = 0.8** —
   `ks_statistic([0.8, 0.9], x->x) ≈ 0.8`. The mirror of case 3. `xs = [0.8, 0.9]`,
   `n = 2`:
   - `i=1, x=0.8`: upper `1/2 − 0.8 = −0.3`; lower `0.8 − 0 = 0.8`
   - `i=2, x=0.9`: upper `2/2 − 0.9 = 0.1`; lower `0.9 − 1/2 = 0.4`

   Sup `= 0.8`, the **lower** gap at `i=1`. A **D+-only** (upper-gap-only) implementation
   would compute `max(−0.3, 0.1) = 0.1`, missing the true sup. This fixture pins down the
   lower side.

   **Cases 3 and 4 are a genuine negative-control pair.** Each is uniquely caught by
   checking the side the *other* omits: case 3 dies under a lower-only bug, case 4 dies
   under an upper-only bug. Neither one alone certifies both-sidedness — you need one
   fixture where the upper gap is *solely* responsible for the max and another where the
   lower gap is *solely* responsible. (The symmetric case 1, where both sides tie, could
   not do this job for either.) Together they lock the `max(..., i/n − Fx, Fx − (i−1)/n)`
   both-sides fold in place.

5. **Empty input throws** — `@test_throws ArgumentError ks_statistic(Float64[], x->x)`.
   The sup-distance over zero order statistics is undefined; the function must reject it
   rather than silently return `0.0` (which is what the loop-with-`d=0.0`-initializer would
   return if the `n == 0 && throw` guard were removed — a misleadingly "perfect fit"). This
   pins the guard.

6. **Public-surface regression guard** — `:ks_statistic` added to the `public surface`
   tuple, which loops asserting `isdefined(StochasticProcesses, f)`. This catches the
   specific failure mode of a name exported from `GOF` but not re-exported from the
   umbrella module (or vice versa) — the two-half export pair (`src/gof.jl` `export` +
   `src/StochasticProcesses.jl` line 46) falling out of sync. Without it a dropped
   re-export would compile fine and only surface as a confusing `UndefVarError` downstream.

## Mutation-gate evidence (this actually happened)

To confirm the tests bite rather than merely pass, the both-sides line

```julia
d = max(d, i / n - Fx, Fx - (i - 1) / n)   # correct
```

was temporarily mutated to the plausible one-sided bug

```julia
d = max(d, i / n - Fx)                      # upper-side only
```

and the **full** suite was re-run. Result: **145 passed / 1 failed out of 146**. The single
failure was the **"bunched right (mirror)"** case — the one where the *lower* gap attains
the sup, so an upper-only statistic returns `0.1` instead of `0.8`, exactly as predicted.

Crucially, the **"bunched left"** case did **not** fail under this mutation: for that
fixture the upper gap (`0.8` at `i=2`) is the one that attains the sup, so an upper-only
implementation still gets it right by luck. This is the concrete demonstration of *why the
negative-control pair is necessary*: a single skewed fixture cannot, by itself, distinguish
an upper-only bug from a lower-only bug — you need one fixture where each side is uniquely
responsible for the max. (The symmetric and unsorted cases also survived this mutation,
since their sup is attained on the upper side too.) After restoring the correct line, the
suite returned to **146 / 146**.

## Empirical / runtime verification

Beyond the unit tests, `ks_statistic` was exercised directly under `using
StochasticProcesses`:

- `isdefined(StochasticProcesses, :ks_statistic)` returns `true` — the export pair
  resolves.
- `ks_statistic([0.1, 0.5, 0.9], x -> x)` returns `0.2333333333333334`, i.e. `7/30` to
  floating-point, confirming the exported symbol computes the hand-verified value through
  the real `using` namespace, not just inside the test module.

There is no CLI/experiment flow to drive for this commit: no experiment consumes
`ks_statistic` yet (its first callers are future units). The "real flow" here is the direct
library call above, which is what a downstream unit's test will make.

## Trade-offs and known limitations

- **No first caller yet (built ahead of need, deliberately).** `ks_statistic` ships with
  no in-repo consumer — its planned users (Unit 3 Cramér–Wold / KL-coefficient checks,
  Unit 5 random-walk-to-BM checks) are not implemented. This is intentional: the utility is
  small, deterministic, and hand-testable now, and placing it in `src/` with a biting
  testset means the future units inherit a verified primitive rather than re-deriving it.
  The risk of building ahead is mitigated by the two-argument signature being fixed by a
  concrete known requirement (`N(0, aᵀΣa)` targets), not guessed.

- **Architectural exception, accepted and documented.** `GOF` breaks the "organized by
  operation on the covariance operator" invariant. Rather than distort that principle to
  cover a goodness-of-fit statistic, the exception is stated plainly in the module
  preamble and the top-level docstring. The cost is one module that does not fit the
  taxonomy; the alternative (forcing it into a covariance-operator framing, or duplicating
  it across two future experiment folders) was judged worse.

- **No sample-value validation.** The function accepts any numeric vector and any callable;
  it does not check that `cdf` is monotone in `[0,1]` or that samples are finite. Consistent
  with the rest of `src/` (kernels do not validate their domain either); a mis-specified
  `cdf` yields a meaningless distance rather than an error.

## Pass conditions verified

1. **Full suite green at 146/146.** `julia --project -e 'using Pkg; Pkg.test()'` passes all
   146 tests (up from 140 before this commit: 5 new `@test`s in the KS testset + 1 from
   extending the `public surface` loop = 6 new assertions).

2. **The new testset bites.** Mutation gate: the upper-side-only version fails exactly the
   "bunched right (mirror)" assertion (1 of 146), and the correct version restores 146/146.
   See the mutation-gate section.

3. **Exported and resolvable.** `isdefined(StochasticProcesses, :ks_statistic)` is `true`
   and `ks_statistic([0.1, 0.5, 0.9], x -> x)` returns `0.2333333333333334 ≈ 7/30` under
   `using StochasticProcesses`.

## Code review

A `/code-review` pass (high effort, 3 parallel finder agents: correctness/cross-file
tracing; reuse/simplification/efficiency; altitude/`CLAUDE.md`-conventions) was run against
this diff. **Zero findings survived across all three angles.** The reference implementation
was specified FINAL/verbatim by the commit plan — its structure and both-sided gap logic
were required to match exactly — so the review's role here was not hunting for an algorithm
bug but confirming (a) correct wiring: the `export` pair, the include/using block, the
`public surface` tuple extension; and (b) that the tests bite. It confirmed both. There
were no findings to report.

## Deviations from plan

**None** on the algorithm: the plan specified the `ks_statistic` reference implementation
verbatim, and it was used verbatim (signature and body semantics identical). The docstring
was expanded, which the plan explicitly permitted ("Expand the docstring as you see fit but
keep the signature and body semantics identical").

One small addition beyond the plan's literal file list is noted for completeness: the
plan's file list for `src/StochasticProcesses.jl` called out only the include/using/export
block, but the module's own docstring (the submodule bullet list, lines 16–17) also gained
a one-line `GOF` bullet, mirroring how `KL` is listed there. This is a minor, harmless edit
squarely within "modify `src/StochasticProcesses.jl`" and is required to keep the
docstring's inventory of submodules accurate — not a scope change.

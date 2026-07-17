# Commit 1 — `brownian_bridge_kernel` (Unit 3 "process zoo", feature `03-process-zoo`)

## TL;DR

Adds one pure function, `brownian_bridge_kernel(t, s) = min(t, s) - t*s`, to the
existing `Kernels` module: the covariance of the standard Brownian bridge on `[0, 1]`.
Plus its docstring, its two `export` lines, and a seven-assertion deterministic testset.
No experiment, no new type, no change to any existing behavior — a strictly additive
"catalogue" commit. The test suite goes from 132 to 140 passing tests: 7 new `@test`s in
the kernel testset plus 1 more from extending the `public surface` regression-guard loop
by one symbol. There were **no deviations from the plan**: the commit implements exactly
what was specified.

This is the first commit doc written in the repo, so it also sets the pattern: explain the
math, the mechanics, every test individually, the empirical end-to-end check, the
trade-offs, and any forward dependencies.

---

## Background: the terms you need

If you already know this codebase you can skip to "What changed". This section is here so a
reader with only moderate stochastic-processes background never has to tab away.

- **Gaussian process (GP).** A random function `X(t)` whose values at any finite set of
  times `t_1, ..., t_n` are jointly Gaussian (normally distributed). A GP is fully
  determined by two things: its **mean function** `m(t) = E[X(t)]` and its **covariance
  kernel**. In this library everything is zero-mean, so the kernel is the whole story.

- **Covariance kernel `R(t, s)`.** The function `R(t, s) = Cov(X(t), X(s)) =
  E[X(t)·X(s)]` (zero-mean). It says how strongly the process value at time `t` co-varies
  with the value at time `s`. "Which stochastic process" is, in this codebase, literally
  just "which kernel" — that is the central design decision recorded in `CLAUDE.md` and in
  the header comment of `src/kernels.jl`. Every kernel here is a plain Julia function of
  two times.

- **Brownian motion (BM).** The canonical continuous random walk starting at `X(0) = 0`.
  Its kernel is `brownian_motion_kernel(t, s) = min(t, s)`. Note `R(0, s) = min(0, s) = 0`:
  BM is *pinned at the start only*. Its variance `R(t, t) = t` grows without bound.

- **Brownian bridge.** Brownian motion *conditioned to return to 0 at time 1* — a random
  path tied down at **both** ends, `X(0) = 0` and `X(1) = 0`, free to wander in between
  (Pavliotis §1.5). Its covariance kernel is

  ```
  R(t, s) = min(t, s) - t·s        on [0, 1].
  ```

  The `- t·s` correction term is exactly what conditioning-on-`X(1)=0` subtracts off the
  plain BM covariance. The variance profile is `R(t, t) = t - t² = t(1 - t)`: zero at both
  ends, maximal (`= 1/4`) at the midpoint `t = 1/2`. This "tied at both ends, fattest in
  the middle" shape is the signature of a bridge.

- **Positive-semi-definite (PSD).** A covariance matrix `Σ` (the kernel evaluated on a grid
  of times) must be PSD — all eigenvalues `≥ 0` — for it to be a legitimate covariance of
  *some* real random vector. It encodes "no linear combination of the `X(t_i)` can have
  negative variance". A kernel that produced a non-PSD `Σ` on some grid would not be a real
  covariance function. We check this numerically as `all(eigvals(Σ) .>= -1e-10)` (a tiny
  negative tolerance absorbs floating-point round-off around a true zero eigenvalue).

- **Cholesky sampling.** To draw a sample path from a zero-mean GP on a grid: assemble the
  covariance matrix `Σ`, factor it as `Σ = L·Lᵀ` (Cholesky factorization, `L` lower
  triangular), draw a standard normal vector `z`, and return `X = L·z`. Then
  `Cov(X) = L·E[z·zᵀ]·Lᵀ = L·Lᵀ = Σ`, so `X` has exactly the desired covariance. This is
  the `sample_cholesky(Σ, rng; jitter=1e-10)` routine in `src/sampling.jl`; the
  **jitter** (a.k.a. nugget) adds `ε·I` before factoring so a singular/degenerate `Σ` does
  not throw — see the trade-offs section for why that matters here.

- **`GaussianProcess` / `assemble_cov` / `empirical_cov`.** The glue in
  `src/gaussianprocess.jl`: `GaussianProcess(kernel)` wraps a kernel; `assemble_cov(gp,
  grid)` evaluates it into a `Symmetric` matrix `Σ` (one entry per pair of grid times);
  `empirical_cov(paths)` estimates the covariance back from a matrix of sample paths.
  **Path orientation is `n_grid × N` — one sample path per column** (a load-bearing
  convention from `CLAUDE.md`; a transpose silently estimates the wrong matrix).

- **KL / Nyström.** Mentioned only in passing below: the Karhunen–Loève expansion is a
  different (eigenbasis) way to represent and sample a GP, built in Unit 2's `src/kl.jl`.
  It is **not** used by this commit — this commit is a kernel, and kernels are upstream of
  every sampler.

---

## What changed

Three files, all additive:

### 1. `src/kernels.jl` — the new function + docstring

```julia
brownian_bridge_kernel(t, s) = min(t, s) - t * s
```

Placed after the three existing kernels (`brownian_motion_kernel`, `exponential_kernel`,
`periodic_kernel`) and added to the module's `export` line. The docstring follows the
house style already set by the other three: a one-line signature, the closed-form formula,
a sentence tying it to the Pavliotis reference (§1.5), and a short "key values worth having
in mind" block listing `R(t,t) = t(1-t)`, `R(0,s) = 0`, and `R(1,s) = 0`. The docstring
also states, in prose, the exact reasoning the tests encode: *"A kernel that only implements
`min(t,s)` and forgets the `-t*s` term would still pass the t=0 pin but silently fail the
t=1 one — both endpoints must be checked to catch it."* The docstring and the test comments
are deliberately consistent so the two never drift.

### 2. `src/StochasticProcesses.jl` — the re-export

The top-level module re-exports each submodule's public names so `using
StochasticProcesses` yields a flat namespace. The one-line re-export for `Kernels` gained
`, brownian_bridge_kernel`. This is the second half of a symmetric pair: the name must be
exported from *both* `Kernels` (line 15) and the umbrella module (line 24) or it will not
resolve for a downstream user. The "public surface" test (below) guards exactly this pair
staying in sync.

### 3. `test/runtests.jl` — the new testset + the surface guard

A new nested `@testset "brownian_bridge_kernel"` under the existing `@testset "Kernels"`,
plus one new symbol added to the `public surface` regression-guard loop. Detailed below.

### How it composes (control flow)

There is no new control flow — that is the point of a kernel commit. `brownian_bridge_kernel`
is a leaf: a pure `(t, s) -> Float64` with no branches beyond the `min`, no state, no RNG,
no allocation. Everything downstream picks it up *for free* through the existing pipeline,
because the whole library is designed so a process is just a kernel argument:

```
brownian_bridge_kernel                       # the new leaf function
   │  passed to
   ▼
GaussianProcess(brownian_bridge_kernel)      # existing wrapper, unchanged
   │  passed to
   ▼
assemble_cov(gp, grid)  ->  Σ (Symmetric)    # evaluates kernel on every (t_i, t_j) pair
   │  passed to
   ▼
sample_cholesky(Σ, rng; jitter=1e-10) -> X   # existing sampler, unchanged
   │  many draws summarized by
   ▼
empirical_cov(paths)  ->  Σ̂                  # should reproduce Σ
```

No existing file was modified structurally; the accretion is exactly "one function + two
export edits + one testset", matching the `CLAUDE.md` rule that `src/` grows by addition and
existing files are not restructured.

---

## Why these design choices (restating the planner's pre-resolved decisions)

These were decided before implementation; they are recorded here so a future reader does not
re-litigate them.

- **Plain function, not a struct or closure.** The three existing kernels are all plain
  functions of `(t, s)` (two of them with keyword params, one without). A bridge has no
  parameters at all, so a struct would carry no state and only add indirection. Keeping it a
  plain function makes it drop straight into `GaussianProcess(kernel)` exactly like the
  others.

- **No `D` / `alpha` parameters.** Those belong to the OU/`exponential_kernel` (noise
  strength and relaxation rate) and to `periodic_kernel`. The *standard* Brownian bridge has
  no free constant — its variance profile `t(1-t)` is fixed by the pinning at 0 and 1. Adding
  `D`/`alpha` here would invent parameters the object does not have.

- **Domain fixed to `[0, 1]`; a parameterized `[0, T]` bridge was rejected.** Pavliotis fixes
  `T = 1` in §1.5 and the bridge is canonically stated on `[0, 1]`. A `T` parameter would be
  unused complexity for no proven need (YAGNI): it would widen the signature, demand its own
  tests, and complicate the docstring, all to support a generalization no current or planned
  unit calls for. The rejected alternative is recorded here so the narrowness is a deliberate
  choice, not an oversight.

---

## Every test, individually (and why it bites)

All seven new assertions live in `@testset "brownian_bridge_kernel"` inside
`test/runtests.jl`. The repo's testing convention (`CLAUDE.md`) is that tests must *bite*:
prefer hand-computed targets over re-running the function's own formula, and choose fixtures
that make shape/orientation/sign bugs impossible to hide. Each assertion below is mapped to
the specific correctness claim it defends and the specific bug it would catch.

1. **Symmetry** — `brownian_bridge_kernel(0.3, 0.7) == brownian_bridge_kernel(0.7, 0.3)`.
   A covariance kernel must satisfy `R(t,s) = R(s,t)`. The formula `min(t,s) - t·s` is
   symmetric in `t, s` by construction, but this pins it: a typo that broke symmetry (e.g.
   `t - t·s`) would fail here. `0.3`/`0.7` are chosen distinct and unequal so the two
   argument orders are genuinely different inputs (a fixture like `0.5, 0.5` could not detect
   an asymmetry).

2. **Hand-computed closed form** — `brownian_bridge_kernel(0.3, 0.7) == 0.09`. Worked by
   hand: `min(0.3, 0.7) - 0.3·0.7 = 0.3 - 0.21 = 0.09`. This is the single strongest
   assertion for *the arithmetic being right*, and it is deliberately **not** a re-run of the
   function's own formula — it is an independently computed number. `0.3`/`0.7` is a good pair
   because `min` picks the *first* argument (so a `min`-argument-swap bug would be visible if
   it interacted with an asymmetric error) and because both the `min` term (`0.3`) and the
   product term (`0.21`) are non-trivial, non-equal, and land on an exactly representable
   decimal (`0.09`), letting us use `==` rather than `≈`. Remove this test and a wrong
   coefficient on the `t·s` term (say `-2·t·s`) could pass everything else.

3. **Both endpoints pinned** — `brownian_bridge_kernel(0.0, 0.4) == 0.0` **and**
   `brownian_bridge_kernel(1.0, 0.4) == 0.0`. **This is the load-bearing pair.** The bridge's
   defining feature is that it is tied to zero at *both* ends, which is exactly the `- t·s`
   term's job. Consider the most likely implementation bug: writing `min(t, s)` and forgetting
   `- t·s` (i.e. accidentally re-implementing Brownian motion). That buggy version still
   returns `0` at `t = 0` (because `min(0, 0.4) = 0` regardless), so the `t = 0` assertion
   *alone would not catch it*. But at `t = 1` the correct kernel gives `min(1, 0.4) - 1·0.4 =
   0.4 - 0.4 = 0`, whereas the buggy `min`-only version gives `min(1, 0.4) = 0.4 ≠ 0`. So the
   `t = 1` assertion is precisely the one that distinguishes a real bridge from Brownian
   motion. **Both** are kept: `t = 0` documents the "same as BM" pin, `t = 1` documents the
   "new" pin that the correction term buys. Dropping either half weakens the test — dropping
   the `t = 1` half would let the whole "it's actually a bridge" claim go unchecked.

   > **Forward dependency (flag).** This two-endpoint pin is deliberately over-specified
   > relative to what *this* commit strictly needs, because a **later, not-yet-implemented
   > commit in Unit 3 will rely on the `R(1, s) = 0` property for a bridge-endpoint check**
   > (an experiment/assertion that the bridge returns to zero at `t = 1`). Testing both
   > endpoints now — not just `t = 0` — is what makes that future check's premise already
   > guaranteed and legible. A future reader wondering "why assert `t = 1` when BM only ever
   > asserts `t = 0`?" should read this paragraph: the `t = 1` pin is the entire reason this
   > kernel exists.

4. **Variance at the midpoint** — `brownian_bridge_kernel(0.5, 0.5) == 0.25`. Hand value:
   `R(0.5, 0.5) = 0.5·(1 - 0.5) = 0.25`. This checks the diagonal (the variance profile) at
   its maximum, the most information-rich single point of `t(1-t)`. Combined with the two
   endpoint zeros, three points of the parabola `t(1-t)` are now pinned (`0`, `1/4`, `0` at
   `t = 0, 0.5, 1`), which fixes the whole quadratic. `0.25` is exactly representable, so
   `==`.

5. **Distinguished from Brownian motion** — `brownian_bridge_kernel(0.3, 0.7) !=
   brownian_motion_kernel(0.3, 0.7)`. An explicit "this is not just BM" guard: BM gives
   `min(0.3, 0.7) = 0.3`, the bridge gives `0.09`, so they differ. This is the interior-point
   companion to the `t = 1` endpoint test: it certifies the `- t·s` term is materially present
   *away* from the endpoints too, not merely at the boundary. If someone "optimized" the
   kernel back to `min(t,s)`, this fails immediately.

6. **PSD on an interior grid** — `all(eigvals(Matrix(Σb)) .>= -1e-10)` where `Σb =
   assemble_cov(GaussianProcess(brownian_bridge_kernel), range(0.05, 0.95; length=12))`.
   This is the "it is a genuine covariance" check, and crucially it runs through the *real*
   `GaussianProcess`/`assemble_cov` assembly path, not the raw function — so it also guards
   that the kernel composes correctly into a matrix. The grid `range(0.05, 0.95; length=12)`
   deliberately **excludes the endpoints 0 and 1**. Why: at `t = 0` (and `t = 1`) the kernel
   is identically zero for every `s` (`R(0, s) = 0`, `R(1, s) = 0`), so including an endpoint
   would put a literally-zero row and column into `Σ`, making it exactly singular with a
   structural zero eigenvalue. The PSD check would then pass for an uninformative reason (the
   zero row) and could mask a genuine near-indefiniteness among the interior modes. Restricting
   to `[0.05, 0.95]` gives a strictly interior, non-degenerate covariance whose PSD-ness is a
   real statement about the bridge's correlation structure. `length=12` is a non-tiny grid
   (12×12), enough that a sign or assembly bug would show as a clearly negative eigenvalue
   rather than round-off. This mirrors the existing `periodic_kernel` PSD test, which likewise
   drops a degenerate grid point.

7. **Public-surface regression guard** — `:brownian_bridge_kernel` was added to the tuple of
   symbols in `@testset "public surface"`, which loops asserting `isdefined(StochasticProcesses,
   f)` for each. This catches the specific failure mode of a name exported from the submodule
   but not re-exported from the umbrella module (or vice versa) — i.e. it guards the
   two-file export pair (`src/kernels.jl` line 15 + `src/StochasticProcesses.jl` line 24)
   staying in sync. Without it, a dropped re-export would compile fine and only surface as a
   confusing `UndefVarError` for an end user.

### Mutation-gate evidence (this actually happened)

To confirm the tests bite rather than merely pass, the correct one-liner was temporarily
mutated to the plausible-bug version

```julia
brownian_bridge_kernel(t, s) = min(t, s)      # deliberately drop the - t*s term
```

and the **full** suite was re-run. Result: **136 passed / 4 failed out of 140**, with all 4
failures inside the new testset. The failing assertions were exactly the ones that depend on
the `- t·s` term:

- `R(1.0, 0.4) == 0.0` — the endpoint pin (bug gives `0.4`); **this is the assertion that
  specifically catches the "min-only" mutation**, and it fired as designed.
- the hand-computed `R(0.3, 0.7) == 0.09` (bug gives `0.3`),
- the midpoint variance `R(0.5, 0.5) == 0.25` (bug gives `0.5`),
- the "distinguished from Brownian motion" inequality (bug makes it *equal* to BM).

The three assertions that did **not** fail under the mutation were the symmetry test (both
formulas are symmetric), the `R(0.0, 0.4) == 0.0` pin (`min(0, 0.4) = 0` either way — the
exact reason a single-endpoint test is insufficient), and the PSD check (BM's `min(t,s)` is
also PSD on the interior grid). That the `t = 0` pin survives the mutation while the `t = 1`
pin catches it is the concrete demonstration of why **both** endpoints are asserted. The
correct implementation was then restored and the full suite returned to **140 passed / 140**.

---

## Empirical / runtime verification (drove the real flow, end-to-end)

Unit tests exercise the raw function; this section is the separate "drove the real flow"
evidence required before committing. Because the package has no CLI/app front end yet, the
"real flow" is the downstream pipeline a user or experiment would actually use: build a
`GaussianProcess`, assemble the covariance, draw many paths with the *shipped* Cholesky
sampler, and estimate the covariance back. Concretely: `GaussianProcess(brownian_bridge_kernel)`
on a 9-point grid `t = 0, 1/8, ..., 1` over `[0, 1]`, `Σ = assemble_cov(...)`, then **2000**
draws via `sample_cholesky(Σ, StableRNG(42); jitter=1e-10)` (the mandated `StableRNG` — never
the global RNG — with the seed recorded here), summarized by `empirical_cov`. Observations:

- **Analytic diagonal is exact.** `Σ`'s diagonal reproduced the hand formula `t(1-t)`
  exactly: `[0, 0.1094, 0.1875, 0.2344, 0.25, 0.2344, 0.1875, 0.1094, 0]` for
  `t = 0, 1/8, ..., 1`. Symmetric about the midpoint, zero at both ends, peak `0.25` at
  `t = 1/2` — the bridge variance profile.

- **Empirical diagonal tracks it within Monte-Carlo noise.** From the 2000 Cholesky-sampled
  paths: `[0, 0.115, 0.185, 0.231, 0.255, 0.242, 0.188, 0.111, 0]` — matching the analytic
  diagonal to the scatter expected at `N = 2000`. This certifies the kernel is a real,
  sampleable covariance through the actual sampler, not just a formula.

- **Pinned at both ends, empirically, end-to-end.** Sampled path values at `t = 0` and
  `t = 1` were `~1e-5` in magnitude across draws — consistent with the Cholesky nugget
  `jitter = 1e-10` (`sqrt(1e-10) = 1e-5`), i.e. the only residual motion at the endpoints is
  the deliberate jitter, nothing else. The bridge really does return to zero at both ends
  when you sample it, not merely in the closed form.

- **Non-degenerate in the interior.** Sample values at `t = 0.5` varied substantially draw to
  draw (e.g. `0.76, -0.49, -0.02, -0.20, 0.13`), confirming genuine variance away from the
  pinned ends — the process is tied down at the boundary but free in the middle, exactly the
  bridge picture.

- **Out-of-domain behavior probed (see trade-offs).** `brownian_bridge_kernel(1.5, 0.5) =
  -0.25` and `brownian_bridge_kernel(-0.2, 0.5) = -0.1`: outside `[0, 1]` the function does
  not clamp or validate — it just evaluates the algebra and can return a negative
  "covariance". Recorded as an accepted limitation below, not a bug.

---

## Trade-offs and known limitations

- **Plain function vs. struct** — resolved in favor of a plain function (see design choices).
  Trade-off: no place to hang parameters, but the standard bridge has none, so this costs
  nothing.

- **`[0, 1]` only, no `[0, T]` generalization** — resolved in favor of the canonical fixed
  domain (see design choices). Trade-off: a future need for a bridge on `[0, T]` or a bridge
  pinned to a nonzero endpoint would require a new function or a parameterization; that cost
  is deferred deliberately (YAGNI) rather than paid speculatively now.

- **No domain validation (accepted, unfixed, pre-existing pattern).** The function evaluates
  `min(t, s) - t·s` for *any* real inputs and will silently return a meaningless negative
  value for `t` or `s` outside `[0, 1]` (demonstrated above: `-0.25` at `(1.5, 0.5)`). This
  was **not** fixed, because it matches the existing convention in `src/kernels.jl` — none of
  the other three kernels validate their domain either (`brownian_motion_kernel`,
  `exponential_kernel`, `periodic_kernel` all just evaluate their algebra). Adding a bounds
  check to this one kernel alone would be an inconsistency, and the fix (if wanted) belongs at
  a higher layer for all kernels at once. The practical risk to note: an accidentally
  out-of-range grid in some *future* experiment — e.g. a typo `range(0, 1.5; length=...)`
  instead of `range(0, 1; ...)` — would produce meaningless numbers rather than erroring. Worth
  knowing; not a regression this commit introduces.

---

## Pass conditions verified

1. **Full suite green at 140/140.** `julia --project -e 'using Pkg; Pkg.test()'` passes all
   140 tests (up from 132 before this commit, confirmed directly by re-running the suite
   against the pre-commit tree: 7 new `@test`s in the `brownian_bridge_kernel` testset plus
   1 more from extending the `public surface` loop by one symbol = 8 new assertions total).

2. **The new testset bites.** Mutation gate: the deliberately broken `min(t, s)`-only version
   fails 4 of the new assertions (including the `R(1.0, 0.4) == 0.0` endpoint pin), and the
   correct version restores 140/140. See the mutation-gate section for which assertions fired
   and why.

3. **Exported and resolvable.** `isdefined(StochasticProcesses, :brownian_bridge_kernel)`
   returns `true`, and a direct call under `using StochasticProcesses` returns
   `brownian_bridge_kernel(0.3, 0.7) == 0.09`. The `public surface` testset guards this
   staying true.

## Code review

A `/code-review` pass (high effort, 8-angle) was run against this diff before this write-up.
**Zero findings survived verification.** The diff is small, mechanical, and follows the plan
exactly: correct formula, symmetric `export` additions in both `src/kernels.jl` and
`src/StochasticProcesses.jl`, a docstring in the same style as the three existing kernels, no
removed behavior, and no cross-file breakage (nothing else in the codebase references the new
symbol yet). Stated plainly: there were no findings to report.

## Deviations from plan

None. This commit implements exactly what was planned — a plain-function `[0,1]` bridge
kernel with no parameters, its docstring, its two exports, and a biting deterministic
testset — with no scope changes, no added parameters, and no structural edits to existing
files.

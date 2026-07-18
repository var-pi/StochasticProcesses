# Commit 5 — Toeplitz/Szegő cross-check ladder (Unit 3 "process zoo", feature `03-process-zoo`)

## TL;DR

Appends a third and final block to the Unit 3 experiment script,
`experiments/03_process_zoo/run.jl` — a **"Phase 5 — Toeplitz/Szegő cross-check +
Welch overlay + aggregate"** section (~70 new lines) that replaces what used to be
the file's closing `recorded:` printf line. It generates two new committed PNG
figures under `experiments/03_process_zoo/figures/` (`cross_check.png`,
`welch_overlay.png`), bringing that folder to **9 figures total**. No `src/` or
`test/` file is touched and no new dependency is added — this is a pure gallery
commit, so it changes no library behaviour and adds nothing to CI (Monte-Carlo
experiments are deliberately kept out of CI per `CLAUDE.md`). This is **Commit 5,
the last code commit** of Unit 3.

Commit 3 established Unit 3's core idea, **reconciliation** — the library's three
independent square roots of a covariance operator (Cholesky, Karhunen–Loève,
circulant embedding) all agree on the OU process, judged against a split-half
bootstrap null. Commit 4 added the **distributional-identities** gate (Brownian
self-similarity, Cramér–Wold projections, KL coefficients, bridge endpoints). This
commit adds the **third and final gate**: a *Toeplitz/Szegő cross-check* that
reconciles **Unit 2's KL machinery** (`nystrom_eigen`) with **Unit 1's spectral
machinery** (`welch_psd`) through a classical asymptotic theorem
(**Grenander–Szegő**) about how the eigenvalues of a large truncated
Toeplitz/convolution operator converge to values of the operator's **symbol**
(essentially its Fourier transform). With this commit in place, the script's final
line becomes `route_ok && dist_ok && xcheck_ok` — the point at which **all three**
of Unit 3's gates must jointly pass for `ALL GATES: PASS` to print.

**The central intellectual content of this commit is one normalization fact.** The
object the KL eigenvalues converge to is the **un-normalized symbol**
`R̂(ω) = 2D/(α²+ω²)`, which is **NOT** the same as Unit 1's normalized OU spectral
density `S(ω) = D/(π(α²+ω²))` — the two differ by a factor of exactly `2π`:
`R̂(ω) = 2π·S(ω)`. Comparing the eigenvalues against `S` directly instead of `R̂`
would produce **no convergence signal at all** (a fitted slope ≈ 0, a residual that
plateaus rather than shrinks), because you would be comparing two quantities that
differ by a large constant factor (~6.28×) rather than genuinely converging.
Getting this right was a **deliberate, verified correction**, not a footnote — it
is the whole reason this commit is nontrivial, and it is spelled out in full below.

There were **no deviations from the plan**: the plan's exact code block was
transcribed verbatim, with one small self-review refactor noted under "Code review"
(defining `R̂` as `2π · S_density(ω)` so the `2π` relationship is *derived in code*,
not duplicated across two hand-written formulas). All five pass conditions were
verified exactly as specified — route-equivalence numbers unchanged, cross-check
ladder `g(T)` values and slope `−0.8893` matching, `ALL GATES: PASS`, 9 figures
including the 2 new ones.

---

## Background: the terms you need

If you already read Commits 1–4 you know this codebase's Gaussian-process, kernel,
`nystrom_eigen`, `quad_nodes_weights`, `welch_psd`, `sample_circulant_embedding`,
and spectral-density vocabulary. This section recaps the pieces this commit
*leans on* in one or two sentences each with a pointer, and then goes deep on what
is genuinely new: the Toeplitz/Szegő (Grenander–Szegő) eigenvalue-convergence
theorem, the OU covariance operator and its symbol, and the crucial distinction
between the **un-normalized symbol `R̂`** and the **normalized spectral density
`S`**. Nothing below is left as an unexplained symbol.

### Recapped from earlier commits (brief, with pointers)

- **Gaussian process (GP) and covariance kernel `R(t, s)`.** A random function
  `X(t)` whose values on any finite set of times are jointly Gaussian; here always
  zero-mean, so it is fully described by its covariance kernel
  `R(t, s) = Cov(X(t), X(s))`. "Which process" is just "which kernel". Full
  treatment in Commit 1's Background.

- **The Ornstein–Uhlenbeck (OU) kernel.** The stationary process used throughout
  this experiment: `exponential_kernel(t, s; D, alpha) = (D/α)·exp(−α·|t−s|)`,
  wrapped locally as `ou(t, s)`. It depends only on the lag `τ = t − s`
  (stationary), decays exponentially with relaxation rate `α`, and has variance
  `R(0) = D/α`. At the parameters used here (`D = 1`, `α = 1`) that is `R(0) = 1`.
  The OU process is the canonical mean-reverting Gaussian process (Pavliotis §2.2).

- **`quad_nodes_weights(T; n)` (Unit 2).** Builds an `n`-point quadrature rule
  (nodes `t_i` and weights `w_i`) on the interval `[0, T]` — the trapezoid rule,
  which on a uniform grid gives equal interior weights. It is the discretization
  the Nyström eigenproblem integrates against.

- **`nystrom_eigen(R, nodes, weights)` (the KL engine, Unit 2).** Discretizes the
  **covariance-operator eigenproblem** `∫₀ᵀ R(t, s)·e(s) ds = λ·e(t)` by the
  quadrature rule and solves it *symmetrized*
  (`W^{1/2} K W^{1/2} g = λ g`, `e = W^{-1/2} g`, where `W = diag(weights)` and
  `K[i,j] = R(t_i, t_j)`), returning eigenvalues `λ_k` sorted descending plus a
  matrix of `W`-orthonormal eigenfunctions on the nodes. **Crucially for this
  commit:** the operator it diagonalizes is the *bare integral operator*
  `∫ R(t,s)·e(s) ds` — there is **no `2π` factor anywhere** in its construction;
  `R` is literally the OU covariance kernel. Unit 2's own anchor identity is
  `λ_k → R̂(k)` (the eigenvalues approach the *un-normalized* transform of the
  kernel). Detailed in Commit 3's and Commit 4's Backgrounds.

- **`welch_psd(x, dt; nseg, window)` (Unit 1).** Estimates the **power spectral
  density** of a single sampled path `x` by Welch's method: split the series into
  `nseg` overlapping segments, apply a taper (here a **Hann window** — a raised-
  cosine that smoothly zeroes each segment's ends to suppress spectral leakage),
  periodogram each segment, and average. Returns one-sided angular frequencies `ω`
  and the estimated density `Ŝ(ω)`. It is a *noisy, data-driven* estimate of the
  same spectral object `S` below. Full treatment in Unit 1's `01_spectral_bochner`.

- **`sample_circulant_embedding(r, rng)` (Unit 1).** Draws one stationary Gaussian
  sample path whose autocovariance sequence is `r = [R(0), R(dt), R(2dt), …]`, via
  the FFT of a circulant embedding of the Toeplitz covariance. It is Unit 1's
  "spectral route" sampler; the Welch overlay below uses it to produce the path it
  then spectrum-estimates.

- **The OU spectral density `S(ω)` (Unit 1, Bochner / Wiener–Khinchin).** Bochner's
  theorem says a stationary covariance `R(τ)` is the Fourier transform of a
  non-negative spectral measure; for OU the **normalized spectral density** is

  ```
  S(ω) = D / (π·(α² + ω²)).
  ```

  This is Unit 1's convention: it is the density defined through the inverse-
  Fourier/Bochner relation `R(τ) = ∫ S(ω)·cos(ωτ) dω` carrying its **own `1/2π`
  normalization** baked into the definition. `S` integrates/reconstructs `R` with
  that `1/2π` convention; it is *not* the raw transform of `R`. In the code it is
  `S_density(ω) = D / (pi * (ALPHA^2 + ω^2))`.

### New in this commit

- **Toeplitz operators and their symbol.** A *Toeplitz* matrix is one whose entries
  depend only on the difference of indices — `A[i,j] = a_{i−j}` — the discrete
  fingerprint of a **stationary** (convolution) operator. A stationary covariance
  kernel `R(t, s) = R(t − s)` discretized on a uniform grid produces exactly such a
  matrix (`R[i,j] = R(t_i − t_j)`). The **symbol** of a Toeplitz/convolution
  operator is the function whose Fourier coefficients (or transform) are the
  operator's entries — for a convolution operator with kernel `R(τ)`, the symbol is
  essentially the Fourier transform of `R`. The symbol is to a Toeplitz operator
  what an eigenvalue-generating function is: it *predicts* the operator's spectrum
  in the large-size limit. The one used here is the **un-normalized symbol**

  ```
  R̂(ω) = 2D / (α² + ω²),
  ```

  the raw (Fourier-cosine, no-`1/2π`) transform of the OU kernel `(D/α)e^{−α|τ|}`.

- **The Grenander–Szegő eigenvalue theorem (the "symbol governs the spectrum"
  theorem).** A classical result on large Toeplitz operators: as the operator's
  size (here the domain length `T`) grows, its eigenvalues become **distributed
  like the values of its symbol**. Concretely, the `k`-th eigenvalue of the OU
  integral operator on `[0, T]` approaches the symbol evaluated at the natural
  Toeplitz-theory frequency

  ```
  ω_k = k·π / T,       λ_k  →  R̂(ω_k) = 2D / (α² + ω_k²)   as T → ∞.
  ```

  Two properties of this convergence are load-bearing here. **(i) It is
  asymptotic** — it is a large-`T` statement, so at any finite `T` there is a gap
  that *shrinks* as `T` grows; the shrinkage rate, not the gap's absolute size, is
  the thing to test. **(ii) It is a *bulk / distributional* statement** — the
  theorem governs how the *bulk* of the eigenvalues distribute, and its convergence
  is known to be **weakest at the edges of the spectrum** (the extreme eigenvalues,
  especially the `k = 1` DC/zero-frequency mode, converge more slowly and
  erratically). A statistic that wants to see the theorem's real rate must
  therefore look at bulk eigenvalues and *exclude the edge*. Both properties drive
  the design of the gap statistic below.

- **The un-normalized symbol `R̂` vs. the normalized density `S` — the crux.**
  These two objects encode the *same* OU spectral shape `1/(α²+ω²)` but differ by a
  constant factor of exactly `2π`:

  ```
  R̂(ω) = 2D/(α²+ω²)  =  2π · [ D/(π(α²+ω²)) ]  =  2π · S(ω).
  ```

  The factor is not a coincidence — it is the difference between two *conventions*
  for "the spectrum of the same process":
  - `nystrom_eigen` diagonalizes the **bare integral operator** `∫ R(t,s) e(s) ds`.
    There is no `2π` in that construction, so its eigenvalues track the **raw
    transform** `R̂` of the kernel (Unit 2's anchor identity `λ_k → R̂(k)`).
  - `S(ω)` is Unit 1's **normalized spectral density**, defined through the
    Bochner/inverse-Fourier relation with its own **`1/2π`** convention built in.

  So the eigenvalues of the Nyström operator converge to `R̂(ω_k)`, *not* to
  `S(ω_k)`. **Operationally, this is the whole commit:** if you compared `λ_k`
  against `S(ω_k)` you would be comparing a quantity to something ~`2π ≈ 6.28×`
  smaller, so the "gap" would be dominated by that constant offset and would
  **plateau** as `T` grows rather than shrink — the fitted log–log slope would come
  out ≈ 0 (no convergence signal) instead of the strongly negative slope the
  theorem predicts. Comparing against the correct un-normalized `R̂` is what makes
  the gap actually vanish. (This is why the code writes `Rhat(ω) = 2π·S_density(ω)`
  — see "Code review" for why it is defined *derived* from `S` rather than as a
  second hand-typed formula.)

---

## What changed

One edit to one existing file, plus its two new figures. No library, test, or
dependency-manifest change.

### `experiments/03_process_zoo/run.jl` (+70 lines; the append)

**The edit.** The file's previous last two lines (Commit 4's final `recorded:`
printf) are replaced by a new `# Phase 5 — Toeplitz/Szegő cross-check + Welch
overlay + aggregate` section that runs the cross-check, draws two figures, prints
the three-gate aggregate, and ends with an **expanded** `recorded:` line.
Everything from the `Phase 5` comment banner to the end of the file is new; the
first ~257 lines (Commits 3 and 4) are unchanged.

**The new constants (every one, and why its value).**

| const | value | meaning / why |
|-------|-------|----------------|
| `T_LADDER` | `[4, 8, 16, 32, 64]` | The ladder of OU domain lengths `T` over which the eigenvalue-vs-symbol gap is measured. A geometric (×2) ladder is what a **log–log slope** fit wants — equally spaced in `log T`. |
| `NODES_PER_UNIT` | `32` | Quadrature nodes **per unit length**, so each `T` uses `nx = 32·T` nodes. Node density is held fixed as `T` grows (rather than a fixed grid size), so the discretization resolves the kernel equally well at every rung and does not confound "more domain" with "coarser grid". |
| `K_BULK` | `30` | Hard cap on how many leading eigenvalues enter the bulk gap statistic. |
| `RES_FLOOR` | `1e-3` | Relative noise-floor cutoff: eigenvalues below `RES_FLOOR·λ₁` are dropped from the statistic (numerical-noise tail, not signal). |
| `XCHECK_THRESH` | `-0.5` | The gate threshold — the fitted `log g` vs `log T` slope must be `< −0.5` (a clearly negative, i.e. genuinely shrinking, gap). |
| `SEED_XCHECK` | `141421` | `StableRNG` seed for the Welch-overlay path only; **recorded** in the final line. |
| `N_WELCH` | `4096` | Length of the single OU path drawn for the Welch overlay. |
| `DT_WELCH` | `0.05` | Time step of that path. |

It reuses earlier constants unchanged: `D = 1.0`, `ALPHA = 1.0` (the OU kernel
parameters), and the `ou(t, s)` kernel closure defined near the top of the file.

**The two new local helpers.**

```julia
S_density(ω) = D / (pi * (ALPHA^2 + ω^2))    # Unit-1 normalized 1/2π density S(ω)
Rhat(ω)      = 2 * pi * S_density(ω)          # un-normalized symbol R̂(ω) = 2π·S(ω)
```

`R̂` is defined **derived from `S`** (multiply by `2π`), not written out as a
second closed-form — so the `2π` relationship is enforced in code and cannot drift
if one formula is later edited (see "Code review").

**The cross-check ladder (the gate).** For each `Tx` in `T_LADDER`:

1. Build a quadrature rule on `[0, Tx]` at `nx = NODES_PER_UNIT*Tx` nodes:
   `nod, wt = quad_nodes_weights(float(Tx); n = nx)`. (Same Unit-2 machinery the
   `eig_decay` portraits earlier in the file use — but here the node count *scales
   with `T`*, not a fixed grid.)
2. Diagonalize the OU integral operator: `λx, _ = nystrom_eigen(ou, nod, wt)`,
   giving eigenvalues `λx` sorted descending.
3. Choose the **bulk upper index**
   `khi = min(K_BULK, findlast(k -> λx[k] > RES_FLOOR*λx[1], eachindex(λx)))`:
   the smaller of the hard cap `30` and the last eigenvalue still above the noise
   floor `1e-3·λ₁`.
4. Record the **bulk gap**
   `g(T) = maximum(abs(λx[k] − Rhat(k*π/Tx)) for k in 2:khi)` — the worst-case
   distance between an eigenvalue and the symbol at its Toeplitz frequency
   `ω_k = k·π/T`, **over `k = 2 … khi`** (note the range **starts at 2**, not 1).

Then fit a straight line to `log g` vs `log T` by least squares: build the
2-column design matrix `Xdes = [1 , log T]`, solve `xco = Xdes \ log.(gs)` (Julia's
backslash = least-squares), take the slope `xslope = xco[2]`, and gate
`xcheck_ok = xslope < XCHECK_THRESH` (i.e. `< −0.5`).

**Three subtleties of the bulk statistic, spelled out.**

- **`k = 1` (the DC / zero-frequency edge mode) is excluded** — the max runs over
  `2:khi`, never from `1`. Grenander–Szegő convergence is a *bulk* statement and is
  known to be **weakest at the spectrum's edges**; the `k = 1` mode does not shrink
  at the bulk rate, so including it would contaminate `g(T)` with a term that stays
  large and flatten the fitted slope toward 0. Excluding the edge is what lets the
  statistic see the theorem's genuine bulk rate.
- **The noise-floor cutoff `RES_FLOOR = 1e-3`** drops the tail of numerically-
  negligible eigenvalues (below `1e-3·λ₁`). Down there the eigenvalues are
  dominated by floating-point noise from the quadrature eigendecomposition, not by
  the asymptotic theorem, so comparing them against `R̂` would measure round-off,
  not convergence.
- **The max is deliberately NOT over all `k`** (never `maximum over 1:length`) —
  precisely because of the two points above: an unrestricted `max_k` would be
  edge-contaminated (by `k = 1`) *and* noise-contaminated (by the sub-floor tail),
  a strictly worse (noisier, non-shrinking) statistic.

  A documented **non-issue** (verified, not merely assumed): `khi ≥ 1` always,
  because the predicate `λx[k] > RES_FLOOR·λx[1]` is trivially true at `k = 1`
  (`λx[1] > 1e-3·λx[1]`), so `findlast` never returns `nothing`. And at the actual
  parameters (`D = 1`, `α = 1`) `khi` comes out to **30 for every `T` in the
  ladder**, so the `2:khi` range is `2:30` throughout and is never empty. This was
  checked at these constants; it is called out here (and the `findlast`/`min`
  fragility was a *declined* review finding — see "Code review") rather than left
  implicit.

**Why a fixed-margin gate, not the SE-multiple gate this repo usually prescribes.**
`CLAUDE.md`'s testing convention says a *stochastic* log–log slope must be gated
against theory within a small multiple (2–3×) of the fitted slope's own standard
error — never a fixed absolute tolerance. **That convention does not apply here,
and deliberately so:** `g(T)` is **fully deterministic**. There is **no RNG
anywhere in its computation** — it is built from the analytic formula `R̂` and a
deterministic quadrature eigendecomposition, not from Monte-Carlo sample paths. So
there is no sampling noise against which to size a standard error; a **fixed margin
below zero** is the correct gate shape for a deterministic asymptotic claim.
Correspondingly, the fitted slope value is **reported as an observation, not
"claimed vs theory"** — Grenander–Szegő does not hand you a specific rate to test
against, only the qualitative claim that the gap *shrinks*; the code comment says
exactly this (`reported, not claimed vs theory`). The gate asks only "does it
shrink, clearly?" — slope `< −0.5`.

**The Welch overlay (pedagogical figure, NOT part of the gate).** On its **own**
`StableRNG(SEED_XCHECK = 141421)` stream `rngx`, it draws **one** OU path via the
Unit-1 circulant-embedding route (`sample_circulant_embedding` of the OU
autocovariance sequence `r_welch = [ou(0, k·DT_WELCH) for k in 0:N_WELCH-1]`,
`N_WELCH = 4096` points at `DT_WELCH = 0.05`), estimates its PSD with
`welch_psd(xw, DT_WELCH; nseg = 16, window = :hann)`, and overlays the estimate
against `2 .* S_density.(ωw)`.

Note the `2×` here is a **different** factor of two from the `2π` in `R̂` — do not
conflate them. The `2×` is Unit 1's convention for a **one-sided** PSD estimate
(folding the two-sided density's negative-frequency half onto the positive axis),
whereas the `2π` in `R̂ = 2π·S` is the un-normalized-operator-vs-normalized-density
convention. Two conventions, two unrelated factors of two; they coincide in neither
origin nor value.

**Why this figure exists if it is not gated.** It is the **visual reconciliation** —
the human-readable demonstration that Unit 1's Welch/spectral route and Unit
2/this-commit's Nyström/Toeplitz route are looking at the *same* underlying
spectral object from two different angles. The *gate* only checks the Nyström-vs-
analytic-symbol convergence quantitatively; the overlay lets a reader *see* that
the estimated spectrum of an actual circulant-embedded path tracks the analytic
density. It is kept out of the gate on purpose — see "Trade-offs".

**The aggregate.** The final logic becomes

```julia
route_ok = all_in                                   # Commit 3 gate
dist_ok  = ss_ok && ctrl_out && cw_ok && kl_ok && br_ok   # Commit 4 gate
println((route_ok && dist_ok && xcheck_ok) ? "ALL GATES: PASS" : "ALL GATES: FAIL")
```

so **all three** of Unit 3's gates must jointly pass. This is the commit where the
script gains its final verdict line.

**The expanded `recorded:` line.** The old Phase-4 `recorded:` line is superseded by
one that **keeps every prior field** (`T_OU`, `N_GRID`, `D`, `alpha`, `N_ROUTE`,
`N_SPLIT`, `N_BRIDGE`, `c`, `K_KL`, `jitter`, `seed`, `seed_dist`) and **adds two
new ones**: `T_ladder=[4, 8, 16, 32, 64]` and `seed_xcheck=141421`. This preserves
the repo convention that the recorded line lists *every* reproducibility constant
that governs the run.

### The two new figures (committed artifacts)

`experiments/03_process_zoo/figures/{cross_check,welch_overlay}.png`, generated
headless (`ENV["GKSwstype"] = "100"`, set earlier in the file) and committed per
the `CLAUDE.md` "commit their figures" rule. They bring the folder to 9 figures.
Their content is described in "Every figure/statistic, individually".

---

## Why these design choices

### Compare `λ_k` against the un-normalized symbol `R̂`, not the density `S`

This is the intellectual core of the commit. As derived in the Background,
`R̂(ω) = 2π·S(ω)`: the Nyström eigenvalues converge to the **un-normalized** symbol
`R̂` because `nystrom_eigen` diagonalizes the bare integral operator
`∫ R(t,s) e(s) ds` with no `2π` in sight (Unit 2's anchor `λ_k → R̂(k)`), whereas
`S` carries Unit 1's own `1/2π` normalization. The **rejected alternative** —
comparing `λ_k` against `S(ω_k)` directly — is *wrong* not by a small bias but
categorically: the two quantities differ by a constant factor `2π ≈ 6.28`, so their
difference would be dominated by that offset and would **plateau** as `T` grows.
The fitted `log g` vs `log T` slope would come out ≈ 0 — *no convergence signal at
all* — masking the very phenomenon the check exists to demonstrate. Only against
the correct `R̂` does the gap actually vanish (observed slope `−0.8893`, strongly
negative). Getting this normalization right was the deliberate, verified correction
at the heart of this commit.

### A fixed-margin deterministic slope gate, not an SE-multiple gate

`CLAUDE.md` prescribes the standard-error-multiple gate for **stochastic** slopes,
because a Monte-Carlo slope carries sampling noise whose scale sets the tolerance.
`g(T)` here has **no randomness whatsoever** — analytic `R̂` and a deterministic
eigendecomposition — so there is no standard error to size a tolerance against, and
an SE-multiple gate would be meaningless (it would be dividing by an estimated zero
noise). A **fixed absolute margin below zero** (`slope < −0.5`) is the right shape
for a deterministic asymptotic claim: it asks only whether the gap clearly shrinks
with `T`, which is exactly (and only) what Grenander–Szegő promises. The slope
value itself is *reported*, not tested against a specific predicted rate, because
the theorem does not supply one.

### Bulk-restricted max, not `max_k` over everything

The statistic is `g(T) = max over k ∈ {2,…,khi} |λ_k − R̂(ω_k)|`, and the two
restrictions are both load-bearing. **Excluding `k = 1`** removes the spectral edge
where Grenander–Szegő convergence is weakest — the DC mode does not shrink at the
bulk rate, and folding it in would keep `g(T)` large and flatten the slope toward
0, hiding real convergence. **The `RES_FLOOR = 1e-3·λ₁` cutoff** removes the
sub-floor eigenvalue tail where floating-point noise (not the theorem) dominates,
so the max is never set by round-off. An **unrestricted `max_k`** would be strictly
worse: edge-contaminated by `k = 1` and noise-contaminated by the tail — a noisier
statistic that would understate the convergence the theorem actually produces. The
restriction is what makes the fitted slope reflect the bulk theorem cleanly.

### The Welch overlay is pedagogical-only, deliberately not gated

The overlay could in principle be folded into the gate (e.g. quantifying how
closely the Welch estimate tracks `2·S`). It is deliberately **not**, for a clean
reason rooted in `CLAUDE.md`'s rule against **conflating tolerance regimes**. The
overlay has its **own stochastic seed** (`SEED_XCHECK`) — it draws a random path —
so gating it would require its own *standard-error-based* tolerance (the Monte-Carlo
regime), mixed into the same gate as the **deterministic** fixed-margin cross-check.
Keeping the two apart keeps each check's tolerance regime pure: the cross-check
gate is 100% deterministic (fixed margin), the overlay is 100% illustrative (no
gate at all). The overlay's job is *visual reconciliation*, not a pass/fail
assertion; folding it in would muddy both.

---

## Every figure/statistic, individually (and why it bites)

Treating each artifact the way Commits 1–2 treat each `@test`: what specific bug
would move it, and how the artifact catches it. The cross-check gap is a
*deterministic* asymptotic quantity — it "bites" by its fitted slope failing to
clear the fixed margin, not by Monte-Carlo scatter.

1. **Printed — the ladder `g(T)` values** (`T=4 → 0.28917`, `T=8 → 0.17892`,
   `T=16 → 0.09686`, `T=32 → 0.04971`, `T=64 → 0.02516`). Each is the worst-case
   bulk eigenvalue-vs-symbol gap at that domain length. The sequence roughly halves
   each time `T` doubles — the visible signature of genuine convergence. A wrong
   symbol normalization (comparing against `S` not `R̂`) would leave these values
   pinned near the constant offset `~2π·S`, not shrinking; a broken `nystrom_eigen`
   or a mis-scaled quadrature would perturb the individual `λ_k` and move them.

2. **Printed — the fitted slope** (`−0.8893`, threshold `< −0.50` → PASS). The
   headline number: the least-squares slope of `log g` vs `log T`, clearly negative
   and comfortably past the `−0.5` margin. This is the quantitative statement "the
   gap shrinks as a power of `T`". If the symbol comparison were wrong, this slope
   would collapse toward 0 (the code comment's "≈0") and the gate would FAIL — the
   single number that would catch the central normalization bug this commit exists
   to get right.

3. **`cross_check.png` — the log–log ladder with fitted-slope line.** The headline
   evidence for the gate: `g(T)` plotted against `T` on log–log axes (markers) with
   the fitted-slope dashed line overlaid, titled with the slope. A reader sees a
   downward-sloping straight line — the geometric decay of the gap. A plateau (flat
   line) would be the visual signature of the wrong-symbol bug; a jagged, non-
   monotone scatter would flag an unstable eigendecomposition. (Note monotonicity is
   *not* required by the gate — only the fitted slope — but the committed run is in
   fact cleanly monotone.)

4. **`welch_overlay.png` — the spectral reconciliation (not gated).** The Welch PSD
   estimate of one circulant-embedded OU path (solid, one-sided) overlaid on the
   analytic `2·S(ω)` (dashed), on `ω ∈ [0, 8]`. The estimate should track the
   analytic curve — a noisy but clearly-following overlay. This is the human-legible
   demonstration that Unit 1's spectral route and Unit 2's Nyström route describe
   the same spectrum. A systematic vertical offset between solid and dashed would
   reveal a normalization mismatch in `welch_psd` or the one-sided `2×` factor; it
   catches such a regression *visually* even though it is not a gate.

5. **Printed — aggregate `ALL GATES: PASS`.** The AND of all three Unit-3 gates
   (`route_ok && dist_ok && xcheck_ok`). One line that flips to FAIL if route
   equivalence, any distributional identity, or the cross-check regresses — the
   single top-level verdict for the whole unit.

6. **Printed — the byte-identical route-equivalence and distributional-identity
   numbers.** Not Phase-5 statistics, but load-bearing: the Commit-3 block still
   prints `Chol–KL = 2.2411`, `Chol–Circ = 2.5811`, `KL–Circ = 1.8591`, and all
   five Commit-4 sub-checks still PASS. Their invariance is the concrete evidence
   that Phase 5's append (and its separate `SEED_XCHECK` stream) perturbed no
   earlier RNG draw.

---

## Empirical / runtime verification

Running `julia --project=experiments experiments/03_process_zoo/run.jl` against the
committed code produced (the Phase-5 portion; the earlier phases printed exactly
their Commit-3/Commit-4 values):

```
cross-check g(T)=max_{k∈bulk}|λ_k − Rhat(kπ/T)| (analytic Rhat=2π·S):
  T=  4  g=0.28917
  T=  8  g=0.17892
  T= 16  g=0.09686
  T= 32  g=0.04971
  T= 64  g=0.02516
  fitted slope log g vs log T = -0.8893 (reported, not claimed vs theory); threshold < -0.50 -> PASS

ALL GATES: PASS

recorded: T_OU=5.0 N_GRID=64 D=1.0 alpha=1.0 N_ROUTE=4000 N_SPLIT=200 N_BRIDGE=200 c=4.0 K_KL=8 T_ladder=[4, 8, 16, 32, 64] jitter=1e-10 seed=271828 seed_dist=20250101 seed_xcheck=141421
```

Facts worth calling out:

- **The gap shrinks cleanly and monotonically** across the ladder (`0.289 → 0.179 →
  0.097 → 0.050 → 0.025`, roughly halving per doubling of `T`), and the fitted
  slope `−0.8893` clears the `−0.5` margin with room — a decisive convergence
  signal, exactly what comparing against the *correct* un-normalized symbol `R̂`
  produces (and precisely what comparing against `S` would *not*).

- **The Commit-3 route-equivalence block is byte-identical** to its committed
  output (`Chol–KL = 2.2411`, `Chol–Circ = 2.5811`, `KL–Circ = 1.8591`), and **all
  five Commit-4 distributional-identity sub-checks still PASS** — confirming Phase 5
  is purely *additive* to the script's earlier RNG streams. Its only stochastic
  draw is on the fresh `StableRNG(SEED_XCHECK)` stream for the Welch path, which
  touches neither the `rng` (route) nor `rng4` (distributional) streams.

- **`ALL GATES: PASS`** printed — the first run at which all three Unit-3 gates
  jointly pass.

- **The expanded `recorded:` line** captures every reproducibility constant,
  including the two new fields `T_ladder=[4, 8, 16, 32, 64]` and
  `seed_xcheck=141421`, while retaining all prior fields.

Both new figures (`cross_check.png`, `welch_overlay.png`) were generated headless
and **visually inspected**: the cross-check plot shows a clean downward log–log line
tracking the fitted slope, and the Welch overlay shows the estimated PSD following
the analytic `2·S(ω)` curve across `ω ∈ [0, 8]`. The figures folder now holds **9
committed PNGs**.

The full deterministic test suite (`julia --project -e 'using Pkg; Pkg.test()'`)
was unaffected — this commit touches **no** `src/` or `test/` file, so CI coverage
is unchanged. Per `CLAUDE.md`, Monte-Carlo experiments are deliberately **not** run
in CI; this script is run locally and its figures committed.

---

## Trade-offs and known limitations

- **Un-normalized symbol `R̂ = 2π·S` vs. the density `S` — the rejected
  alternative.** Comparing `λ_k` against `S(ω_k)` directly (the natural-looking but
  *wrong* choice) yields no convergence signal at all: the two differ by a constant
  `2π ≈ 6.28`, so the gap plateaus and the fitted slope is ≈ 0 rather than strongly
  negative. The correct comparison is against the un-normalized symbol `R̂`, which
  is what the Nyström eigenvalues actually converge to (Unit 2's anchor
  `λ_k → R̂(k)`). This was a deliberate, verified correction, not a footnote.

- **Fixed-margin deterministic slope gate vs. the SE-multiple gate.** This repo's
  default for a *stochastic* slope is a standard-error-multiple gate; it does **not**
  apply here because `g(T)` carries no randomness (analytic `R̂` + deterministic
  eigendecomposition). A fixed margin below zero (`slope < −0.5`) is the correct
  gate shape for a deterministic asymptotic claim, and the slope is *reported* (the
  theorem gives no specific rate to test against), not claimed vs theory.

- **Bulk-restricted max vs. `max_k` over everything.** The statistic maxes over
  `k = 2 … khi`, excluding the `k = 1` spectral edge (where Grenander–Szegő
  convergence is weakest) and the sub-`RES_FLOOR` noise tail. An unrestricted
  `max_k` would be a strictly worse statistic — edge-contaminated by the slow
  `k = 1` mode and noise-contaminated by the round-off tail — flattening the slope
  and understating the real bulk convergence.

- **The Welch overlay is pedagogical-only, deliberately not gated.** Folding it into
  the gate would require its own standard-error-based tolerance (it has its own
  stochastic seed `SEED_XCHECK`), mixing a Monte-Carlo tolerance regime into the
  same gate as the deterministic cross-check. Keeping it separate keeps each check's
  tolerance regime clean, per `CLAUDE.md`'s rule against conflating regimes. Its job
  is visual reconciliation, not a pass/fail assertion.

- **`khi` is stable only at the current constants (a verified non-issue, not a
  fix).** With `D = 1`, `α = 1`, `khi` evaluates to `30` for every rung of
  `T_LADDER`, so `2:khi` is `2:30` throughout and never empty. This was checked at
  these parameters; if the constants were changed the `findlast`/`min` chain could
  in principle produce a small or empty range. That fragility was raised in
  self-review and **declined as out of scope** (see "Code review") — it is a
  hypothetical about future edits, not a defect in the code as written.

---

## Code review

A `/code-review` self-review pass was run against this diff before commit. **One
finding was accepted and fixed; one was declined as out of scope** (per an explicit
pre-resolved decision recorded in this commit's plan). Reported factually below.

### Fixed

1. **Hardcoding-drift risk between `R̂` and `S`.** The plan's initial transcription
   had `R̂` and `S_density` as **two independently hand-written formulas** both
   encoding the same OU spectral shape — `S(ω) = D/(π(α²+ω²))` and, separately,
   `R̂(ω) = 2D/(α²+ω²)`. Both are correct, but the `2π` relationship between them
   lived only in a comment: a future edit to one formula (say, changing `D` handling
   or adding a parameter) could silently break the `R̂ = 2π·S` identity that the
   whole cross-check depends on. **Fix:** define `R̂` as **derived** from `S` —
   `Rhat(ω) = 2 * pi * S_density(ω)` — so the `2π` relationship is enforced *in
   code*, not merely asserted in prose. This is numerically identical to the two
   separate formulas (`2π · D/(π(α²+ω²)) = 2D/(α²+ω²)`) with zero effect on any
   printed value or gate outcome — it removes a drift hazard, not a bug.

### Declined (with reason)

- **`khi`'s `findlast`/`min` fragility.** A finding proposed hardening the
  `khi = min(K_BULK, findlast(...))` chain against a hypothetical future in which
  changed constants make `findlast` return `nothing` or the `2:khi` range empty.
  **Declined**, per an **explicit pre-resolved decision already recorded in this
  commit's plan** as a "known non-issue... do not fix". The reason: at the actual
  parameters (`D = 1`, `α = 1`) the predicate `λx[k] > RES_FLOOR·λx[1]` always holds
  at `k = 1`, so `findlast` never returns `nothing` and `khi ≥ 1`; in fact `khi`
  is `30` for every rung, so `2:khi` is `2:30` and never empty. No concrete failure
  path exists in the code as written; hardening against a speculative future
  constant change belongs to that future edit's scope, not this commit's.

The self-review also confirmed the load-bearing `CLAUDE.md` conventions hold for the
new block: the only stochastic draw uses an explicit `StableRNG(SEED_XCHECK)` stream
(no bare `randn()` / global RNG), recorded in the final line; the cross-check
statistic is deterministic and gated by a fixed margin *because* it carries no
randomness (the SE-multiple convention is correctly *not* applied); the Phase-5
append does not touch the earlier `rng`/`rng4` streams (verified byte-identical);
and `ENV["GKSwstype"] = "100"` (set earlier in the file) governs the two new
figures too.

---

## Deviations from plan

**None.** The algorithm and code were transcribed **verbatim** from the dispatched
plan (per its "Transcribe verbatim" instruction), with exactly the **one mechanical
refactor** described under "Code review" applied afterward during the mandatory
self-review pass: defining `Rhat` as `2π · S_density(ω)` (derived) rather than as a
second hand-typed closed-form. That refactor is numerically identical and changes no
printed value, RNG draw, or gate outcome. **No other deviation.** All five pass
conditions were verified exactly as the plan specified.

---

## Pass conditions verified

1. **Script runs clean end to end.**
   `julia --project=experiments experiments/03_process_zoo/run.jl` completes without
   error and produces the transcript above.

2. **Route-equivalence and distributional-identity numbers stayed byte-identical.**
   The Commit-3 block still prints `Chol–KL = 2.2411`, `Chol–Circ = 2.5811`,
   `KL–Circ = 1.8591`, and all five Commit-4 sub-checks still PASS — proving the
   Phase-5 append (on its own `SEED_XCHECK` stream) perturbed no earlier draw.

3. **Cross-check ladder and slope match exactly.** `g(T)` = `0.28917, 0.17892,
   0.09686, 0.04971, 0.02516` across `T = 4, 8, 16, 32, 64`, fitted slope
   `−0.8893 < −0.5` → the cross-check gate reads PASS.

4. **`ALL GATES: PASS`** — the AND of all three Unit-3 gates
   (`route_ok && dist_ok && xcheck_ok`) prints PASS.

5. **Two new figures written** (`cross_check.png`, `welch_overlay.png`) under
   `experiments/03_process_zoo/figures/`, bringing the folder to **9 figures**,
   generated headless and visually inspected.

6. **No CI/library regression.** The commit touches no `src/` or `test/` file and
   adds no dependency; the deterministic suite is unaffected, consistent with the
   two-tier split (this experiment is intentionally outside CI). The expanded
   `recorded:` line carries the two new fields `T_ladder=[4, 8, 16, 32, 64]` and
   `seed_xcheck=141421` alongside all prior constants.

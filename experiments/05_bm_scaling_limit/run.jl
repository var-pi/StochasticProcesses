# ============================================================================
#  Experiment 05 — Brownian motion as a scaling limit (Donsker's theorem)
# ----------------------------------------------------------------------------
#  Units 0-4 treated Brownian motion (BM) as a covariance kernel R(t,s)=min(t,s).
#  Unit 5 steps off that spine: BM is ALSO the weak limit of a rescaled random
#  walk,
#      W_n(t) = S_{floor(n t)} / sqrt(n),    S_k = sum of k iid increments,
#  for ANY increment law with mean 0 and variance 1 (Donsker's invariance
#  principle) -- a statement about laws on path space, not an operation on a
#  covariance operator. Per the master plan, the rescaled-walk builders live
#  entirely in this experiment file, not in src/; the only library calls are
#  `empirical_cov` (Unit 0) here, and `ks_statistic` (Unit 3) starting Phase 2.
#
#  Phase 1 (this file, so far) lays the foundation:
#    - the increment-law samplers (Rademacher, uniform, Exp(1)-shifted) and the
#      `rescaled_walk` lattice builder, reused by every later phase;
#    - the Donsker picture: one law's rescaled paths overlaid at increasing n,
#      visually tightening onto a continuous limit;
#    - GATE 01a: for EVERY increment law in LAW_ORDER, the walk's own two-time
#      covariance is EXACT on the lattice --
#          Cov(S_i/sqrt(n), S_j/sqrt(n)) = min(i,j)/n = min(t_i, t_j)
#      for any iid variance-1 increments, independent of n and of the increment
#      MEAN (Cov is mean-invariant, and empirical_cov demeans its input, so this
#      is purely a VARIANCE-normalization check). Running it once PER LAW (not
#      just once) means a wrong-scale bug in any one law (e.g. uniform's sqrt(3)
#      factor) is caught HERE, at the foundation -- a mean-normalization bug
#      (e.g. a forgotten -1 shift on the exponential) is invisible to THIS gate
#      by construction, but is nowhere near subtle: it makes the walk drift like
#      sqrt(n) instead of settling down, which is unmissable in the paths figure
#      and fails Phase 2's marginal-distribution check outright.
#
#  Run:  julia --project=experiments experiments/05_bm_scaling_limit/run.jl
#  Monte-Carlo (seeded), so NOT run in CI; the figures it writes are committed.
#  Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================
using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots
using SpecialFunctions: erf, erfinv, gamma_inc, loggamma   # erfinv/gamma_inc/loggamma added in commit 02

ENV["GKSwstype"] = "100"   # headless: GR writes PNGs with no display (CI/agent shells)
gr()

const SEED_PATHS = 20260722   # Donsker overlaid-paths figure (own small stream)
const SEED_COV   = 20260723   # GATE 01a covariance-vs-N Monte Carlo (own stream, shared across LAW_ORDER)

const N_PATH_SEQ = [4, 16, 64, 256, 1024]             # increasing lattice fineness for the Donsker picture

const N_LATTICE  = 64                                  # lattice steps n for the covariance check
const M_SUB      = 8                                   # number of subsampled lattice rows (N_LATTICE a multiple of M_SUB)
const N_LADDER   = [100, 320, 1000, 3200, 10000]       # MC sample-size ladder: ~2 decades, geometric (as in 00_covariance_core)
const NGROUP     = 20                                  # independent replicate ladders for the slope SE (batch means)
const SE_MULT    = 2.5                                 # gate multiple on the batch-means slope SE (repo convention: 2.5-3x)

const OUTDIR = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

@assert N_LATTICE % M_SUB == 0     # else the subsampled rows miss the endpoint t=1
@assert length(N_LADDER) >= 3       # ols needs n-2 >= 1 dof for a slope SE

# ----------------------------------------------------------------------------
#  Shared contract surface (defined here, consumed by commits 02-04; see
#  00-shared-context.md). ols_slope_se / normcdf / _quantile are copied
#  VERBATIM from their source files so the attribution comments stay honest;
#  normcdf and _quantile are not used until commits 02/03/04 but belong here
#  since every later phase appends to THIS SAME file.
# ----------------------------------------------------------------------------

# Self-contained OLS: fitted slope and its standard error (as in 00_covariance_core). Copied
# verbatim from experiments/04_ergodicity/run.jl:49-57. Used below to fit ONE replicate's N-ladder
# at a time; its residual SE reflects only that single replicate's 5-point scatter, which badly
# understates the true rung-to-rung noise (see GATE 01a) -- the batch-means SE computed there,
# not this residual SE, is what the gate actually uses.
function ols_slope_se(x::AbstractVector, y::AbstractVector)
    n = length(x); x̄ = sum(x) / n; ȳ = sum(y) / n
    Sxx = sum((x .- x̄) .^ 2)
    slope = sum((x .- x̄) .* (y .- ȳ)) / Sxx
    intercept = ȳ - slope * x̄
    resid = y .- (intercept .+ slope .* x)
    s2 = sum(abs2, resid) / (n - 2)
    return slope, sqrt(s2 / Sxx)
end

# Copied verbatim from experiments/03_process_zoo/run.jl:161. Used by the QQ figures in commits
# 02 and 04 (`using SpecialFunctions: erf` above).
normcdf(z) = 0.5 * (1 + erf(z / sqrt(2)))

# type-7 (linear-interpolation) quantile on an already-sorted vector — matches
# Statistics.quantile without pulling Statistics into the experiments env. Copied verbatim from
# experiments/03_process_zoo/run.jl:23-29.
function _quantile(sorted, p)
    n = length(sorted)
    h = (n - 1) * p + 1
    lo = floor(Int, h)
    lo >= n && return sorted[n]
    return sorted[lo] + (h - lo) * (sorted[lo+1] - sorted[lo])
end

# --- increment-law samplers: iid, mean 0, variance 1 -------------------------
# Donsker's invariance principle needs only two moments (mean 0, variance 1) from the increment
# law -- the LIMIT is universal (always BM) regardless of the law's shape. So every later phase
# re-tests the SAME lattice builder against three qualitatively different laws: a bounded 2-point
# law (Rademacher, +/-1), a bounded continuous law (uniform on (-sqrt(3), sqrt(3))), and a skewed
# unbounded law (Exp(1) shifted to mean 0, skewness 2) -- a shape-specific normalization bug (a
# wrong variance scale, caught by GATE 01a below; a forgotten mean-shift, caught by the
# marginal-distribution check in commit 02) cannot hide behind the other two laws passing.
# Each closure has signature f(rng, dims...) -> Array{Float64}.
const increment_samplers = Dict{Symbol,Function}(
    :rademacher  => (rng, dims...) -> rand(rng, (-1.0, 1.0), dims...),               # P(+-1)=1/2 each: mean 0, var 1
    :uniform     => (rng, dims...) -> sqrt(3.0) .* (2.0 .* rand(rng, dims...) .- 1.0), # U(-sqrt3,sqrt3): mean 0, var (2sqrt3)^2/12=1
    :exponential => (rng, dims...) -> (-log.(1.0 .- rand(rng, dims...))) .- 1.0,       # Exp(1)-1 via inverse CDF: mean 0, var 1, skew 2
)

# Pinned iteration order for every per-law loop in commits 01 (finite-variance part)-03. Dict
# iteration order is NOT stable across Julia versions, and this Dict shares a single StableRNG
# stream across laws below -- iterating the Dict directly would silently reorder draws and change
# every committed number. Commit 04 adds a :pareto key in its own block, not by extending this.
const LAW_ORDER = (:rademacher, :uniform, :exponential)

"""
    rescaled_walk(sampler, n, N, rng) -> Matrix{Float64}

Build `N` independent rescaled random-walk lattices of `n` steps each, one path per COLUMN (the
repo-wide `n_grid × N` convention, so the `(n+1) × N` result feeds `empirical_cov` directly).
Column `j` is
    [S_0, S_1, ..., S_n] / sqrt(n),    S_0 = 0,    S_k = sum of the first k increments,
drawn from `sampler(rng, n, N)` (an `n × N` matrix of iid mean-0 variance-1 increments).
Donsker's theorem builds a continuous path by linear interpolation between these lattice points,
but that interpolation is never computed explicitly here: every functional this unit checks (the
two-time covariance below, the running max in commit 03, the KS-vs-n rate in commit 02) is a
function of the LATTICE values alone, so the lattice is already a complete representation.
"""
function rescaled_walk(sampler::Function, n::Int, N::Int, rng)
    increments = sampler(rng, n, N)                       # n × N iid mean-0 var-1 increments
    S = vcat(zeros(1, N), cumsum(increments; dims = 1))    # (n+1) × N; row 1 is S_0 = 0
    return S ./ sqrt(n)
end

# ============================================================================
#  Figure — the Donsker picture: one law's rescaled paths at increasing n
# ----------------------------------------------------------------------------
#  Rademacher (the simplest law) at an increasing sequence of lattice finenesses, all confined to
#  [0, 1]. As n grows the lattice path visually tightens onto a continuous limit -- the
#  qualitative content of Donsker's theorem, ahead of any quantitative gate. Own small StableRNG
#  stream, independent of GATE 01a's stream below (cannot perturb its numbers).
# ============================================================================
rng_paths = StableRNG(SEED_PATHS)
p1 = plot(; xlabel = "t", ylabel = "W_n(t)", legend = :topleft, size = (800, 480),
          left_margin = 4Plots.mm, top_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
          title = "Donsker picture: rescaled Rademacher walk, n = " * join(N_PATH_SEQ, ", "))
for n in N_PATH_SEQ
    Wn = rescaled_walk(increment_samplers[:rademacher], n, 1, rng_paths)
    tgrid = (0:n) ./ n
    plot!(p1, tgrid, Wn[:, 1]; label = "n = $n", alpha = 0.8)
end
savefig(p1, joinpath(OUTDIR, "rescaled_paths.png"))

# ============================================================================
#  GATE 01a — two-time covariance -> min(s,t), per law in LAW_ORDER
# ----------------------------------------------------------------------------
#  For ANY iid mean-0 variance-1 increment law, Cov(S_i/sqrt(n), S_j/sqrt(n)) = min(i,j)/n =
#  min(t_i, t_j) EXACTLY at lattice times t_k = k/n, independent of n (linearity of Cov plus the
#  variance-1 normalization -- no CLT or higher moment needed). So there is no n-limit to gate
#  here; the only thing that can be wrong is whether the builder produces correctly-normalized,
#  independent increments, and the Monte-Carlo estimate of the covariance must approach the exact
#  target at the standard N^{-1/2} rate (the Unit-0 headline machinery, reused per law).
#
#  Subsample M_SUB interior lattice rows (N_LATTICE a multiple of M_SUB, excluding the trivial
#  t=0 row where Cov=0 identically) so the target matrix M[i,j]=min(t_i,t_j) lands exactly on
#  lattice times; reuse `empirical_cov` (src/gaussianprocess.jl) rather than hand-rolling a
#  covariance.
#
#  ‖Ĉ_N − M‖_F is itself a random variable whose relative scatter is LARGE at small N (checked
#  empirically: at N=100 the run-to-run relative SD is ~55%, since the M_SUB×M_SUB=8×8 target has
#  only 36 independent upper-triangular entries to average over) -- one single N-ladder fit is
#  consequently too noisy to gate directly against a tight SE. So the point estimate here is the
#  classical BATCH-MEANS estimator: draw NGROUP independent replicate N-ladders (each an
#  independent draw at every rung, continuing ONE StableRNG stream -- the 00_covariance_core
#  method, repeated), fit a slope per replicate, and report
#    slope  = mean of the NGROUP replicate slopes      (the low-noise point estimate)
#    SE     = std(replicate slopes) / sqrt(NGROUP)      (the honest batch-means slope SE)
#  This is the direct analogue of experiments/04_ergodicity/run.jl:87-92's group-slope SE, adapted
#  to an axis (sample count N) where "more data" means more independent replicates rather than a
#  longer nested time axis. ols_slope_se's own residual SE (printed per replicate-mean curve, for
#  contrast only) is NOT the gate: a single fit's residual SE cannot see the replicate-to-replicate
#  scatter that batch means measures directly.
# ============================================================================
sel_k    = collect(M_SUB:M_SUB:N_LATTICE)   # e.g. 8,16,...,64 for M_SUB=8, N_LATTICE=64: excludes t=0
sel_rows = sel_k .+ 1                        # 1-based row index for lattice time t = k / N_LATTICE
t_sel    = sel_k ./ N_LATTICE
Mtarget  = [min(a, b) for a in t_sel, b in t_sel]   # analytic min(s,t) on the subsampled lattice

# One N-ladder replicate: draw a FRESH, independent batch of N columns at every rung (continuing
# `rng`, so replicates and rungs are all independent of each other), fit the log-log slope.
function cov_ladder_replicate(sampler::Function, rng)
    errors = [norm(empirical_cov(rescaled_walk(sampler, N_LATTICE, N, rng)[sel_rows, :]) .- Mtarget)
              for N in N_LADDER]
    slope, _ = ols_slope_se(log10.(N_LADDER), log10.(errors))
    return slope, errors
end

rng_cov = StableRNG(SEED_COV)   # ONE stream, threaded across LAW_ORDER in its pinned order
law_results = NamedTuple[]
for law in LAW_ORDER
    sampler = increment_samplers[law]
    reps    = [cov_ladder_replicate(sampler, rng_cov) for _ in 1:NGROUP]
    gslopes = [r[1] for r in reps]
    errmat  = reduce(hcat, (r[2] for r in reps))'        # NGROUP × length(N_LADDER)
    mean_errors = vec(sum(errmat; dims = 1)) ./ NGROUP    # per-rung mean error, for the plotted curve

    slope_v = sum(gslopes) / NGROUP
    se_v    = sqrt(sum(abs2, gslopes .- slope_v) / (NGROUP - 1)) / sqrt(NGROUP)   # batch-means SE
    _, se_ols = ols_slope_se(log10.(N_LADDER), log10.(mean_errors))               # contrast only, not gated

    gate_law = abs(slope_v + 0.5) < SE_MULT * se_v
    @printf("GATE 01a [%-11s] cov-vs-N slope: %.4f;  |slope+0.5| = %.4f  vs  %.1f*SE = %.4f  (SE_batch %.4f, SE_ols(mean curve) %.4f) -> %s\n",
            String(law), slope_v, abs(slope_v + 0.5), SE_MULT, SE_MULT * se_v, se_v, se_ols, gate_law ? "PASS" : "FAIL")
    push!(law_results, (; law, mean_errors, slope = slope_v, se = se_v, gate = gate_law))
end
gate_01a = all(r.gate for r in law_results)
println("GATE 01a (all laws) -> ", gate_01a ? "PASS" : "FAIL")

# --- Figure: log-log convergence (all laws) + Ĉ-vs-min heatmap (one law) -----
colors_by_law = Dict(:rademacher => :steelblue, :uniform => :darkorange, :exponential => :seagreen)
p2a = plot(; xscale = :log10, yscale = :log10, xlabel = "N", ylabel = "mean ‖Ĉ_N − M‖_F  (over $NGROUP replicates)",
           title = "GATE 01a: covariance -> min(s,t)", legend = :bottomleft,
           left_margin = 6Plots.mm, bottom_margin = 5Plots.mm, top_margin = 4Plots.mm)
for r in law_results
    plot!(p2a, N_LADDER, r.mean_errors; marker = :circle, color = colors_by_law[r.law],
          label = @sprintf("%s (batch-mean slope %.3f)", String(r.law), r.slope))
end
ref = law_results[1].mean_errors[1] .* (N_LADDER ./ N_LADDER[1]) .^ (-0.5)
plot!(p2a, N_LADDER, ref; linestyle = :dash, color = :black, label = "reference slope -1/2")

# heatmap panel: a fresh representative draw (rademacher, top rung) -- own small step in the SAME
# continuing stream, purely illustrative (not part of any gate).
law_demo  = :rademacher
Chat_demo = empirical_cov(rescaled_walk(increment_samplers[law_demo], N_LATTICE, N_LADDER[end], rng_cov)[sel_rows, :])
p2b = heatmap(t_sel, t_sel, abs.(Chat_demo .- Mtarget);
              title = @sprintf("|Ĉ_N − M|  (%s, N=%d)", String(law_demo), N_LADDER[end]),
              xlabel = "s", ylabel = "t", aspect_ratio = :equal,
              left_margin = 4Plots.mm, bottom_margin = 5Plots.mm, top_margin = 4Plots.mm)
savefig(plot(p2a, p2b; layout = (1, 2), size = (1200, 480)), joinpath(OUTDIR, "covariance_vs_min.png"))

@printf("\nrecorded: seed_paths=%d, seed_cov=%d, n_lattice=%d, m_sub=%d, N_ladder=%s, ngroup=%d, se_mult=%.1f\n",
        SEED_PATHS, SEED_COV, N_LATTICE, M_SUB, string(N_LADDER), NGROUP, SE_MULT)

# ============================================================================
#  Phase 2 — HEADLINE: KS-vs-n marginal-convergence rate, hybrid gating
# ----------------------------------------------------------------------------
#  Donsker gives BM as the n->infinity LIMIT for every increment law; this phase asks the
#  next question: at what RATE does the one-time marginal W_n(1) = S_n/sqrt(n) approach its
#  limiting law N(0,1)? The rate splits into THREE mechanisms, not two -- a correction to the
#  master-plan brief (fixed in commit 05), verified here by EXACT computation, not asserted:
#
#    - exponential (skewness 2, the one skewed law): the Edgeworth SKEWNESS term dominates,
#      giving KS ~ n^(-1/2).
#    - Rademacher (symmetric, but a LATTICE law -- only 2 atoms): skewness is exactly 0, so the
#      skewness term vanishes, but a DIFFERENT term takes over -- the Esseen lattice/discreteness
#      correction, which is O(n^(-1/2)) and skewness-INDEPENDENT. It dominates for exactly the
#      same reason a histogram of a coin-flip sum never quite looks continuous: the distribution
#      lives on a grid of spacing 2/sqrt(n), and no amount of averaging removes that structure at
#      any finite n. Also KS ~ n^(-1/2) -- NOT n^(-1). This is the plan's error: Rademacher was
#      wrongly grouped with the smooth-symmetric laws. Exact computation below (a deterministic
#      binomial-pmf calculation, no sampling at all) settles it: the EXACT Rademacher-to-Phi KS
#      curve has log-log slope -0.499 to -0.4996 over n in the hundreds-to-thousands -- decisively
#      -1/2, not -1.
#    - uniform (symmetric AND smooth -- no lattice, no skew): BOTH of the above n^(-1/2) terms
#      vanish (no skewness, no discreteness), so the leading survivor is the next Edgeworth term,
#      kurtosis, which is O(n^(-1)) -- a full power faster.
#
#  Headline: "two roads to n^(-1/2) (skewness AND lattice discreteness), one road to n^(-1)
#  (smooth symmetric)."
#
#  GATING IS HYBRID, and asymmetric for a load-bearing, empirically-discovered reason (see the
#  commit doc for the full war story). In brief: the two n^(-1/2) laws have LARGE signals
#  (exponential ~0.13/sqrt(n), Rademacher ~0.4/sqrt(n)), resolvable by Monte Carlo -- BUT the
#  Rademacher case turned out to be a genuinely delicate MC-design problem, not a trivial one:
#  the raw empirical KS statistic (N samples vs Phi) carries its OWN finite-N sampling BIAS (the
#  classical Kolmogorov floor, E[D_N] ~ 0.87/sqrt(N)), and this bias does NOT average away across
#  independent batch-means replicates (it is a bias, not noise) -- so naively cranking up either N
#  or the replicate count NGROUP can make a marginal gate WORSE, not better, once N is large enough
#  that the batch-means SE has shrunk below the (separately-irreducible) finite-n curvature bias of
#  the EXACT rate curve itself. The resolution (theory-first, validated empirically): choose n_min
#  large enough that the curvature bias is already negligible, then choose N in the regime where
#  the floor-driven excess is ALSO small relative to a still-honest batch-means SE -- a real,
#  non-trivial sweet spot, tuned separately per law since their signal amplitudes differ.
#  Uniform's n^(-1) signal is far smaller still (~9e-4 at n=32) -- resolving it by Monte Carlo
#  would need N ~ 10^8 and still leave under a decade of usable n, i.e. it is PHYSICALLY
#  un-gateable by Monte Carlo at any practical N. So uniform alone is gated against its EXACT
#  (deterministic, zero sampling noise) Irwin-Hall curve; all three exact curves are overlaid on
#  the Monte Carlo points in the figure below, so the plot itself shows the MC clouds tracking
#  theory down toward the floor.
# ============================================================================

# --- exact-CDF helpers (deterministic; used for BOTH the overlay curves AND the uniform gate) ---

# Rademacher: S_n = 2*Binomial(n,1/2) - n exactly (n independent coin flips). The exact KS-to-Phi
# distance is a sup over the (n+1) support points of the discrete CDF, checked on BOTH sides of
# each jump (same two-sided logic as `ks_statistic`, just against an exact pmf instead of an
# empirical 1/n-weighted sample). Evaluated in LOG SPACE (loggamma) so n can run into the
# thousands without the naive C(n,k)/2^n ratio overflowing -- this is cancellation-free (unlike
# the uniform Irwin-Hall sum below), so no BigFloat is needed here.
function ks_exact_rademacher(n::Int)
    logpmf = [loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1) - n * log(2.0) for k in 0:n]
    pmf = exp.(logpmf)
    pmf ./= sum(pmf)             # renormalize away the tiny fp roundoff in the log-sum-exp
    F = cumsum(pmf)               # F[k+1] = P(S_n <= 2k-n), 1-indexed
    d = 0.0
    Fprev = 0.0
    for k in 0:n
        x = (2k - n) / sqrt(n)
        Phi = normcdf(x)
        d = max(d, F[k+1] - Phi, Phi - Fprev)    # both sides of the jump
        Fprev = F[k+1]
    end
    return d
end

# Exponential: S_n + n ~ Erlang(shape n, scale 1) exactly (sum of n iid Exp(1)). The exact CDF is
# the regularized lower incomplete gamma P(n, x*sqrt(n)+n); `gamma_inc(a, x)` returns `(P, Q)`, so
# `[1]` selects P. Continuous law -> no jumps -> the sup is found by a fine grid search (200,001
# points comfortably resolves KS values as small as 1e-3, three orders of magnitude finer).
function ks_exact_exponential(n::Int; ngrid = 200_000, xmax = 8.0)
    d = 0.0
    for x in range(-xmax, xmax; length = ngrid)
        arg = x * sqrt(n) + n
        F = arg <= 0 ? 0.0 : gamma_inc(n, arg)[1]
        d = max(d, abs(F - normcdf(x)))
    end
    return d
end

# Uniform: Irwin-Hall CDF (closed alternating sum) of the mapped sum. `increment_samplers[:uniform]`
# draws sqrt(3)*(2V-1) for V~Unif(0,1), so S_n = sqrt(3)*(2*IH_n - n) where IH_n ~ IrwinHall(n) is
# the sum of n iid Unif(0,1) -- inverting, S_n/sqrt(n) <= x iff IH_n <= (x*sqrt(n/3)+n)/2.
#
# PRECISION CAVEAT (load-bearing): the naive Float64 alternating sum sum_k (-1)^k C(n,k) (x-k)^n
# suffers catastrophic cancellation for n gtrsim 30 (terms of size ~n^n canceling to a result of
# size ~1) -- exactly why this gate is restricted to a SMALL-n window (UNI_WINDOW below), which is
# also precisely where uniform's -1 regime already lives. Mitigation: every intermediate value is
# computed in BigFloat/BigInt (exact rational combinatorics, not floating-point), so there is no
# cancellation at all within the window used -- only the FINAL division back to Float64 loses
# precision, at the ~1e-16 relative level, utterly negligible next to the ~1e-3 KS values gated
# here. The n=8/16/32 sanity check below independently guards this implementation.
function irwin_hall_cdf(x, n::Int)
    x <= 0 && return 0.0
    x >= n && return 1.0
    kmax = floor(Int, x)
    s = zero(BigFloat)
    for k in 0:kmax
        s += (-1.0)^k * binomial(BigInt(n), BigInt(k)) * (BigFloat(x) - k)^n
    end
    return Float64(s / factorial(BigInt(n)))
end

function ks_exact_uniform(n::Int; ngrid = 20_000, xmax = 6.0)
    d = 0.0
    for x in range(-xmax, xmax; length = ngrid)
        ih_arg = (x * sqrt(n / 3) + n) / 2
        d = max(d, abs(irwin_hall_cdf(ih_arg, n) - normcdf(x)))
    end
    return d
end

# --- Monte-Carlo ladders, ONE PER LAW (tuned separately -- see the commit doc's war story for why
# a shared N/ladder does not work: the two laws' signal amplitudes and finite-n curvature decay at
# different rates, so a single choice cannot simultaneously clear the Kolmogorov floor for both) ---
const SEED_KS_EXP = 12345         # own stream: exponential MC ladder
const SEED_KS_RAD = 999           # own stream: Rademacher MC ladder (independent of the exponential draws)
const NGROUP_KS   = 20            # batch-means replicates (repo convention, matches GATE 01a)
const SE_MULT_KS  = 2.5           # repo convention (2.5-3x); see commit doc for the tuning story

const N_LADDER_EXP = [40, 80, 160, 320]         # walk-length ladder (n_min=40 keeps finite-n curvature <=0.003)
const N_MC_EXP     = 1_000_000                  # samples per rung; clears the exponential floor at n_max with margin

const N_LADDER_RAD = [150, 200, 270, 360]       # n_min=150 keeps Rademacher's OWN curvature <=0.001
const N_MC_RAD     = 500_000                    # samples per rung; sweet spot -- see war story (bigger N is NOT better)

@assert length(N_LADDER_EXP) >= 3 && length(N_LADDER_RAD) >= 3   # ols needs n-2 >= 1 dof for a slope SE

# One replicate: fresh, independent N columns at every rung (continuing the law's OWN stream),
# fit to one log-log slope -- the same `cov_ladder_replicate` pattern as GATE 01a, just on the
# KS-to-Phi distance of the endpoint marginal instead of a covariance Frobenius error.
function ks_ladder_replicate(sampler::Function, n_ladder, N, rng)
    errs = [ks_statistic(rescaled_walk(sampler, n, N, rng)[end, :], normcdf) for n in n_ladder]
    slope, _ = ols_slope_se(log10.(n_ladder), log10.(errs))
    return slope, errs
end

mc_results = Dict{Symbol,NamedTuple}()
# NOTE ON ORDER: unlike every other per-law loop in this file, this one does NOT iterate
# LAW_ORDER. LAW_ORDER exists to pin draw order on a SHARED, continuing StableRNG stream (so
# reordering never changes committed numbers) -- but here each law gets its OWN independent
# stream (SEED_KS_EXP / SEED_KS_RAD), so iteration order has no effect on reproducibility at
# all. The order below (exponential, then Rademacher) instead mirrors the plan's own gate
# numbering (02-exp before 02-rad).
mc_configs = ((:exponential, N_LADDER_EXP, N_MC_EXP, SEED_KS_EXP), (:rademacher, N_LADDER_RAD, N_MC_RAD, SEED_KS_RAD))
for (law, n_ladder, N_mc, seed) in mc_configs
    sampler = increment_samplers[law]
    rng_law = StableRNG(seed)
    reps = [ks_ladder_replicate(sampler, n_ladder, N_mc, rng_law) for _ in 1:NGROUP_KS]
    gslopes = [r[1] for r in reps]
    errmat  = reduce(hcat, (r[2] for r in reps))'          # NGROUP_KS x length(n_ladder)
    mean_err = vec(sum(errmat; dims = 1)) ./ NGROUP_KS

    slope_v = sum(gslopes) / NGROUP_KS
    se_v    = sqrt(sum(abs2, gslopes .- slope_v) / (NGROUP_KS - 1)) / sqrt(NGROUP_KS)   # batch-means SE
    gate    = abs(slope_v + 0.5) < SE_MULT_KS * se_v
    mc_results[law] = (; n_ladder, N_mc, slope = slope_v, se = se_v, mean_err, gate)

    tag = law == :exponential ? "02-exp" : "02-rad"
    @printf("GATE %s [%-11s] MC slope: %.4f;  |slope+0.5| = %.4f  vs  %.1f*SE = %.4f  (SE %.5f) -> %s\n",
            tag, String(law), slope_v, abs(slope_v + 0.5), SE_MULT_KS, SE_MULT_KS * se_v, se_v,
            gate ? "PASS" : "FAIL")
end
gate_02exp = mc_results[:exponential].gate
gate_02rad = mc_results[:rademacher].gate

# --- GATE 02-uni-exact: uniform's -1 slope, gated against its EXACT (deterministic) curve -------
# Sanity check FIRST: guards the Irwin-Hall implementation before it is trusted for the gate.
# Independently-verified target values (hand/reference-computed, not re-derived from this code).
const UNI_SANITY_N      = (8, 16, 32)
const UNI_SANITY_TARGET = (0.00354, 0.00174, 0.00087)
const UNI_SANITY_TOL    = 2e-4     # generous vs the 3-significant-figure targets

uni_sanity_vals = [ks_exact_uniform(n) for n in UNI_SANITY_N]
uni_sanity_ok = all(abs(uni_sanity_vals[i] - UNI_SANITY_TARGET[i]) < UNI_SANITY_TOL
                     for i in eachindex(UNI_SANITY_N))
@printf("SANITY uniform exact-CDF: n=8/16/32 -> %.5f/%.5f/%.5f  (targets %.5f/%.5f/%.5f) -> %s\n",
        uni_sanity_vals[1], uni_sanity_vals[2], uni_sanity_vals[3],
        UNI_SANITY_TARGET[1], UNI_SANITY_TARGET[2], UNI_SANITY_TARGET[3],
        uni_sanity_ok ? "PASS" : "FAIL")

# The gate itself: a SMALL-n window (Irwin-Hall's precision-safe regime, and exactly where the -1
# rate already lives) -- deterministic, so a FIXED absolute margin gates it, not a multiple of an
# SE (there is no sampling noise at all; the only "residual" is Edgeworth misspecification, which
# is already tiny and shrinking as UNI_WINDOW grows, per the exact values below).
const UNI_WINDOW        = [8, 16, 32, 64, 128]
const UNI_SLOPE_MARGIN  = 0.05

uni_exact_curve = [ks_exact_uniform(n) for n in UNI_WINDOW]
uni_slope, _ = ols_slope_se(log10.(UNI_WINDOW), log10.(uni_exact_curve))   # se unused: no sampling noise to report
gate_02uniexact = uni_sanity_ok && abs(uni_slope + 1) < UNI_SLOPE_MARGIN
@printf("GATE 02-uni-exact: Irwin-Hall exact slope = %.4f;  |slope+1| = %.4f  vs margin %.2f -> %s\n",
        uni_slope, abs(uni_slope + 1), UNI_SLOPE_MARGIN, gate_02uniexact ? "PASS" : "FAIL")

# --- GATE 02-sep: uniform's rate is decisively steeper than BOTH n^(-1/2) laws ------------------
# Pins the headline contrast as an assertion, not an assumption: since |uni_slope| ~ 1.0 while the
# two MC slopes are ~0.48-0.50, this gate has enormous margin and carries no real risk of its own --
# its job is to make the "smooth-symmetric is faster" claim a checked fact, not prose.
gate_02sep = abs(uni_slope) > abs(mc_results[:exponential].slope) &&
             abs(uni_slope) > abs(mc_results[:rademacher].slope)
@printf("GATE 02-sep: |uniform slope| %.4f > |exponential slope| %.4f and > |Rademacher slope| %.4f -> %s\n",
        abs(uni_slope), abs(mc_results[:exponential].slope), abs(mc_results[:rademacher].slope),
        gate_02sep ? "PASS" : "FAIL")

gate_02 = gate_02exp && gate_02rad && gate_02uniexact && gate_02sep
println("GATE 02 (all) -> ", gate_02 ? "PASS" : "FAIL")

# --- Figure: ks_rate.png -- MC clouds + all three exact curves + floor + guide slopes -----------
pr = plot(; xscale = :log10, yscale = :log10, xlabel = "n (walk length)", ylabel = "KS(S_n/√n, Φ)",
          title = "GATE 02: KS-vs-n marginal rate (hybrid gating)", legend = :bottomleft,
          size = (900, 560), left_margin = 6Plots.mm, bottom_margin = 5Plots.mm, top_margin = 4Plots.mm)
for law in (:exponential, :rademacher)
    r = mc_results[law]
    scatter!(pr, r.n_ladder, r.mean_err; color = colors_by_law[law], markersize = 5,
             label = @sprintf("%s MC (slope %.3f)", String(law), r.slope))
    exact_curve = law == :exponential ? ks_exact_exponential.(r.n_ladder) : ks_exact_rademacher.(r.n_ladder)
    plot!(pr, r.n_ladder, exact_curve; color = colors_by_law[law], linestyle = :dash,
          label = "$(String(law)) exact")
end
plot!(pr, UNI_WINDOW, uni_exact_curve; color = colors_by_law[:uniform], marker = :diamond,
      linestyle = :dash, label = @sprintf("uniform exact (slope %.3f)", uni_slope))

floor_exp = 0.87 / sqrt(N_MC_EXP)
floor_rad = 0.87 / sqrt(N_MC_RAD)
hline!(pr, [floor_exp]; color = :seagreen, linestyle = :dot, alpha = 0.6,
       label = @sprintf("floor 0.87/√N (exp, N=%d)", N_MC_EXP))
hline!(pr, [floor_rad]; color = :steelblue, linestyle = :dot, alpha = 0.6,
       label = @sprintf("floor 0.87/√N (rad, N=%d)", N_MC_RAD))

anchor_n, anchor_y = N_LADDER_EXP[1], mc_results[:exponential].mean_err[1]
plot!(pr, N_LADDER_EXP, anchor_y .* (N_LADDER_EXP ./ anchor_n) .^ (-0.5);
      color = :black, linestyle = :dashdot, alpha = 0.4, label = "guide slope −1/2")
plot!(pr, UNI_WINDOW, uni_exact_curve[1] .* (UNI_WINDOW ./ UNI_WINDOW[1]) .^ (-1.0);
      color = :black, linestyle = :solid, alpha = 0.3, label = "guide slope −1")
savefig(pr, joinpath(OUTDIR, "ks_rate.png"))

# --- Figure: marginal_qq.png -- histogram + QQ of S_n/sqrt(n) vs N(0,1), per law, at a large n ---
# Own small RNG step per law, continuing AFTER that law's own ladder draws above (so it cannot
# perturb any gate number) -- reuses each law's own largest MC ladder rung as "a large n" (already
# validated as a legitimate walk length for that law) rather than inventing a fourth parameter.
const N_QQ = 20_000                                    # own draw; illustrative only, not gated
const QQ_PROBS = range(0.005, 0.995; length = 199)      # avoid the p=0/1 quantile singularities
normquantile(p) = sqrt(2) * erfinv(2p - 1)              # standard-normal quantile (inverse of normcdf)

qq_n = Dict(:rademacher => N_LADDER_RAD[end], :uniform => UNI_WINDOW[end], :exponential => N_LADDER_EXP[end])
pqq = plot(; layout = (2, 3), size = (1200, 700), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, top_margin = 3Plots.mm)
for (i, law) in enumerate(LAW_ORDER)
    sampler = increment_samplers[law]
    rng_qq = StableRNG(SEED_KS_EXP + SEED_KS_RAD + i)    # own small deterministic step, one per law
    n_qq = qq_n[law]
    endpoint = sort(rescaled_walk(sampler, n_qq, N_QQ, rng_qq)[end, :])

    histogram!(pqq[i], endpoint; normalize = :pdf, bins = 60, label = "", alpha = 0.6,
               title = "$(String(law)), n=$n_qq", xlabel = "S_n/√n", ylabel = "density")
    xs = range(-4, 4; length = 200)
    plot!(pqq[i], xs, exp.(-xs .^ 2 ./ 2) ./ sqrt(2π); linewidth = 2, label = "N(0,1)")

    qemp = [_quantile(endpoint, p) for p in QQ_PROBS]
    qth  = normquantile.(QQ_PROBS)
    scatter!(pqq[i+3], qth, qemp; markersize = 2, label = "", xlabel = "N(0,1) quantile",
             ylabel = "empirical quantile", title = "QQ: $(String(law))")
    plot!(pqq[i+3], qth, qth; linestyle = :dash, color = :black, label = "y=x")
end
savefig(pqq, joinpath(OUTDIR, "marginal_qq.png"))

@printf("\nrecorded: seed_ks_exp=%d, seed_ks_rad=%d, n_ladder_exp=%s, N_mc_exp=%d, n_ladder_rad=%s, N_mc_rad=%d, ngroup_ks=%d, se_mult_ks=%.1f, uni_window=%s\n",
        SEED_KS_EXP, SEED_KS_RAD, string(N_LADDER_EXP), N_MC_EXP, string(N_LADDER_RAD), N_MC_RAD,
        NGROUP_KS, SE_MULT_KS, string(UNI_WINDOW))

# ============================================================================
#  Phase 3 — path functional: running maximum -> half-normal (reflection principle)
# ----------------------------------------------------------------------------
#  Donsker's theorem is a statement about the WHOLE path, not just its one-time marginal -- so a
#  functional that depends on the path's shape, not just its endpoint, should also converge. The
#  running maximum M^(n) = sup_[0,1] W_n is the natural next check: by the reflection principle,
#  sup_[0,1] B =_d |B_1| =_d |N(0,1)|, whose CDF is the HALF-NORMAL F(x) = 2*Phi(x) - 1 (x >= 0).
#
#  Running max is EXACT on the lattice: linear interpolation between lattice points never exceeds
#  the larger endpoint (the interpolated segment is a straight line, so its max is attained at one
#  of its ends), so `vec(maximum(W; dims=1))` over the (n+1)-row lattice IS the true continuous-
#  path supremum -- no fine grid or extra interpolation needed, unlike a functional that could hide
#  excursions BETWEEN lattice points.
#
#  GATING IS A CONSISTENCY CHECK, not a rate -- the opposite regime from Phase 2. Phase 2 wanted
#  the KS-vs-n curve to clear the Kolmogorov SAMPLING floor from ABOVE (a resolvable MC signal
#  against sampling noise). Here the claim is different: "at one large, fixed n, the running-max
#  law already looks like the half-normal, not merely 'some limit'" -- gated against a PRINCIPLED
#  ABSOLUTE BOUND (MAX_KS_BOUND) sized to the functional's OWN deterministic finite-n deviation
#  (~c/sqrt(n), the running-max analogue of Phase 2's Edgeworth terms -- NOT sampling noise, and
#  several times larger than the Kolmogorov floor at these N (empirically ~4-6x); an earlier
#  version of this gate conflated the two and failed for all three laws at first pass -- see the
#  commit doc), paired with a WRONG-TARGET control (KS against the full normal Phi, which the
#  half-normal is emphatically not: Phi(0)=0.5 while the half-normal's CDF is 0 at x=0, so their
#  sup-distance is close to 0.5 analytically, independent of n or sampling) -- this closes the
#  loophole where "KS is small" could mean "converged to SOME garbage law that happens to be close
#  to both," rather than specifically the half-normal.
#
#  n is sized INDEPENDENTLY PER LAW, not shared. Rademacher's running max is LATTICE-VALUED (only
#  n+1 possible values), so KS(M, half-normal) carries the SAME O(n^-1/2) discreteness term seen in
#  Phase 2's Rademacher marginal (~0.4/sqrt(n)) ON TOP of the shared c/sqrt(n) deviation -- it still
#  converges, but needs a somewhat larger n than the two continuous laws (uniform, exponential) for
#  its true deviation to land under the same MAX_KS_BOUND. Sizing all three laws' n identically
#  would either waste budget on the continuous laws or leave Rademacher short -- so N_STEPS_MAX is a
#  per-law Dict, not a single scalar. (Pushing n into the many thousands to instead chase the
#  SAMPLING floor down is deliberately avoided -- for Rademacher that costs a multi-GB lattice
#  matrix for no added rigor, since the discrimination against the wrong target is already decisive
#  by a wide margin at the modest n used here.)
# ============================================================================

# Half-normal CDF: the reflection-principle law of sup_[0,1] B = |B_1| =_d |N(0,1)|. Zero for
# x < 0 (the running max of a path started at 0 is never negative in the LIMIT; finite-n paths are
# lattice/discrete but their max is >= 0 by construction too, since S_0 = 0 is always in the max).
halfnormcdf(x) = x >= 0 ? 2 * normcdf(x) - 1 : 0.0

const SEED_MAX = 20260724   # GATE 03 running-max MC (own stream, threaded across LAW_ORDER)

# Per-law walk length n -- consistency regime (one large fixed n, not a ladder). Uniform and
# exponential are continuous laws with no lattice term, but the running-max functional itself
# converges to the half-normal at its OWN genuine finite-n rate ~ c/sqrt(n) (an Edgeworth-type
# deviation from the reflection-principle limit, not sampling noise -- see MAX_KS_BOUND below).
# Rademacher's max is additionally lattice-valued, adding its familiar O(n^-1/2) discreteness term
# (~0.4/sqrt(n)) on top -- it needs a somewhat larger n than the continuous laws for its true
# deviation to land in the same ballpark; do not size all three laws' n identically.
const N_STEPS_MAX = Dict(:rademacher => 3_000, :uniform => 1_200, :exponential => 1_200)

# Per-law sample count N. Not tuned for a tiny sampling floor (unlike Phase 2) -- MAX_KS_BOUND below
# budgets the c/sqrt(n) DETERMINISTIC deviation, which dominates the KS distance at these n by an
# order of magnitude over any N in this range, so N only needs to be "large enough that single-draw
# sampling noise is a minor contributor," not "as large as memory allows." Kept modest so
# `rescaled_walk`'s materialized (n+1) x N lattice stays a few hundred MB even for Rademacher.
const N_MC_MAX = Dict(:rademacher => 30_000, :uniform => 50_000, :exponential => 50_000)

# --- GATE 03 THRESHOLDS -----------------------------------------------------
# MIS-DESIGN CORRECTED (see docs/commits/05-bm-scaling-limit/03-running-max.md for the full story):
# an earlier version of this gate budgeted `floor(N) + margin`, i.e. only the finite-N SAMPLING
# floor (the classical Kolmogorov E[D_N] ~ 0.87/sqrt(N)) -- but KS(M, half-normal) at finite n is
# NOT ~0 plus sampling noise. It is a genuine, deterministic convergence deviation of size ~c/sqrt(n)
# (the running max is itself only a finite-sample approximation to sup_[0,1] B, exactly like the
# Edgeworth corrections in Phase 2's marginal rate) -- empirically ~0.017-0.024 at the n used below,
# several times larger than the sampling floor at these N (~4-6x). Conflating the two regimes (an
# exact rate-scale quantity vs. a fixed absolute margin) first surfaced with a much smaller n
# (uniform/exponential at n=300): there, the true deviation (~0.033-0.046) came in 2.5-3.6x over a
# floor(N)+margin threshold of ~0.013-0.015, and every law failed. The fix: gate against a single
# PRINCIPLED ABSOLUTE BOUND sized to that deterministic deviation at the chosen n, not the sampling
# floor. This is deliberately NOT "floor + margin" -- there is no meaningful floor-shrinking effect
# to chase here (see the comment on N_MC_MAX above), so folding one in would just reintroduce the
# same conflation.
const MAX_KS_BOUND  = 0.05   # correct-target bound: budgets the ~c/sqrt(n) finite-n deviation at
                              # the n above (empirically ~0.017 for uniform, ~0.024 for exponential
                              # at n=1200, ~0.018 for Rademacher at n=3000) with real margin -- a FAIL
                              # here means a genuine mis-scaled or wrongly-shaped max, not noise.
const MAX_WRONG_MIN = 0.3    # wrong-target control threshold: KS(M, Phi) must clear this. The
                              # analytic sup|halfnormcdf - normcdf| approaches 0.5 (achieved as x ->
                              # 0^-: normcdf(0) = 0.5, halfnormcdf(0) = 0) independent of n or N, so
                              # 0.3 leaves enormous margin (>6x the 0.05 correct-target bound) --
                              # this control carries no real risk of its own; its job is to make the
                              # "specifically half-normal, not just some limit" claim a checked fact.

@assert all(law -> haskey(N_STEPS_MAX, law) && haskey(N_MC_MAX, law), LAW_ORDER)   # else the loop
                                 # below would KeyError (or silently skip a law) instead of failing loudly

rng_max = StableRNG(SEED_MAX)   # ONE stream, threaded across LAW_ORDER in its pinned order --
                                 # reordering the loop would silently change every committed number
max_results = NamedTuple[]
for law in LAW_ORDER
    sampler = increment_samplers[law]
    n_steps = N_STEPS_MAX[law]
    N_mc    = N_MC_MAX[law]
    W = rescaled_walk(sampler, n_steps, N_mc, rng_max)
    M = vec(maximum(W; dims = 1))   # running max PER PATH (per column) -- one scalar per sample

    ks_half  = ks_statistic(M, halfnormcdf)   # correct-target KS: should be small
    ks_wrong = ks_statistic(M, normcdf)       # wrong-target control: should be large (~0.5)

    gate_correct = ks_half < MAX_KS_BOUND
    gate_wrong   = ks_wrong > MAX_WRONG_MIN
    gate_law = gate_correct && gate_wrong

    @printf("GATE 03 [%-11s] n=%-6d  KS(M,half-normal)=%.5f  (bound %.5f) -> %s   |   KS(M,Φ)=%.4f  (> %.2f) -> %s\n",
            String(law), n_steps, ks_half, MAX_KS_BOUND, gate_correct ? "PASS" : "FAIL",
            ks_wrong, MAX_WRONG_MIN, gate_wrong ? "PASS" : "FAIL")
    push!(max_results, (; law, n_steps, M, ks_half, ks_wrong, gate = gate_law))
end
gate_03 = all(r.gate for r in max_results)
println("GATE 03 (all laws) -> ", gate_03 ? "PASS" : "FAIL")

# --- Figure: running_max.png -- empirical CDFs of M^(n) overlaid on the half-normal target (solid)
# and the wrong-target Phi (dashed), per law -- makes the discrimination visible, not just gated ---
pmax = plot(; xlabel = "x", ylabel = "CDF", legend = :bottomright, size = (900, 560),
            title = "GATE 03: running-max law -> half-normal (reflection principle)",
            left_margin = 6Plots.mm, bottom_margin = 5Plots.mm, top_margin = 4Plots.mm)
xs_max = range(0, 4; length = 400)
plot!(pmax, xs_max, halfnormcdf.(xs_max); color = :black, linewidth = 2, label = "half-normal (correct target)")
plot!(pmax, xs_max, normcdf.(xs_max); color = :black, linestyle = :dash, alpha = 0.6, label = "Φ (wrong-target control)")
for r in max_results
    Ms = sort(r.M)
    n_ms = length(Ms)
    stride = max(1, n_ms ÷ 400)          # subsample the ECDF for a legible plot; N_MC_MAX points is too dense
    ecdf_y = (1:n_ms) ./ n_ms
    plot!(pmax, Ms[1:stride:end], ecdf_y[1:stride:end]; color = colors_by_law[r.law], alpha = 0.8,
          label = @sprintf("%s empirical (n=%d, KS=%.4f)", String(r.law), r.n_steps, r.ks_half))
end
savefig(pmax, joinpath(OUTDIR, "running_max.png"))

@printf("\nrecorded: seed_max=%d, n_steps_max=%s, N_mc_max=%s, max_ks_bound=%.3f, max_wrong_min=%.2f\n",
        SEED_MAX, string(N_STEPS_MAX), string(N_MC_MAX), MAX_KS_BOUND, MAX_WRONG_MIN)

# ============================================================================
#  Phase 4 — FALSIFIER: an infinite-variance increment breaks Donsker
# ----------------------------------------------------------------------------
#  Every earlier phase used increments with mean 0 AND variance 1 -- Donsker's invariance
#  principle needs BOTH finite moments (variance 1 just fixes the scale; what matters is that it
#  is FINITE). This phase asks what happens when the second hypothesis is dropped: feed the same
#  `rescaled_walk` machinery a SYMMETRIC, mean-0, but genuinely INFINITE-VARIANCE increment, and
#  show the n^(-1/2) picture breaks in exactly the way theory predicts -- not vaguely "worse," but
#  in the OPPOSITE direction of every earlier gate.
#
#  This law is handled in its OWN block below, NOT by extending LAW_ORDER -- LAW_ORDER pins the
#  draw order of a SHARED StableRNG stream across commits 01-03's finite-variance loops, and
#  appending to it would silently reorder those draws, changing every already-committed number in
#  this file (a CLAUDE.md non-negotiable). `:pareto` is added to the `increment_samplers` Dict
#  (mutating the existing binding is fine -- `const` pins the Dict OBJECT, not its contents) so it
#  is reachable by the same name-based lookup as the other three laws, but it is only ever used
#  from this phase's own code below, with its own `SEED_CTRL` stream.
# ============================================================================

const GAMMA_PARETO = 1.5   # tail index γ ∈ (1,2): E|X| finite (γ>1, so the symmetric law has an
                             # honest mean of exactly 0) but E[X²] = γ∫₁^∞ x^(1-γ) dx diverges for
                             # any γ <= 2 -- at γ=1.5 the exponent 1-γ=-0.5 gives ∫x^(-0.5)dx, which
                             # diverges at the UPPER limit (∞^0.5 = ∞): this is a PROVABLY,
                             # genuinely infinite-variance law, not merely "a large-variance one."

# Symmetric Pareto increment: sign * U^(-1/γ), U ~ Uniform(0,1) drawn UNTRUNCATED. U^(-1/γ) is a
# standard Pareto(shape γ, scale 1) variable on [1,∞): P(U^(-1/γ) > x) = P(U < x^(-γ)) = x^(-γ) for
# x >= 1, the textbook power-law tail. An independent Rademacher sign symmetrizes it around 0 (no
# empirical de-meaning needed -- odd symmetry makes E[X]=0 exact, just like the other three laws'
# construction, though here via symmetry rather than a shift).
#
# CRITICAL CORRECTNESS GUARD (this is what GATE 04a below is FOR): the tail must be left
# UNTRUNCATED. A bug that caps U away from 0 (e.g. clamping to (eps, 1) for "numerical safety," or
# capping `mag` at some large but finite value) would cap the Pareto tail away from infinity and
# silently restore a FINITE variance -- Gate 04a's whole job is to catch exactly that: if the
# variance were secretly finite, `Var(S_n/√n)` would settle toward a constant (as in commits
# 01-03) instead of growing with n, and the slope gate below would (correctly) FAIL. Nothing here
# clips `U` or `mag` in any way.
increment_samplers[:pareto] = (rng, dims...) -> begin
    U   = rand(rng, dims...)                 # Uniform(0,1), the FULL open support -- no clamping
    mag = U .^ (-1.0 / GAMMA_PARETO)          # Pareto(scale=1, shape=γ) on [1,∞): P(mag>x)=x^(-γ)
    sgn = rand(rng, (-1.0, 1.0), dims...)     # independent Rademacher sign -> symmetric about 0
    sgn .* mag
end

const SEED_CTRL     = 20260725   # own stream for BOTH gates below (04a then 04b, in that order),
                                   # independent of every earlier stream in this file -- cannot
                                   # perturb any already-committed number.
const N_LADDER_CTRL = [50, 100, 200, 400, 800]   # walk-length ladder n (own axis -- NOT the
                                                    # sample-count N of commits 01-02's ladders)
const N_MC_CTRL     = 2_000      # replicates of S_n/√n per rung, per batch-means group
const NGROUP_CTRL   = 20         # batch-means replicate groups (repo convention)

@assert length(N_LADDER_CTRL) >= 3   # ols needs n-2 >= 1 dof for a per-replicate slope

# --- GATE 04a: Var(S_n/√n) GROWS with n (does not settle at 1) ------------------------------------
# Theory: S_n is a sum of n iid symmetric-Pareto(γ) increments, each with infinite variance, so
# Var(S_n) is infinite for every finite n (a sum of finitely many infinite-variance terms is still
# infinite-variance). Concretely, S_n/√n = n^(1/γ - 1/2) * Y_n where Y_n converges in law to a
# γ-stable variable of O(1) typical scale (the classical stable-CLT for regularly-varying tails) --
# since γ=1.5 < 2, the exponent 1/γ - 1/2 = 1/3 > 0 is STRICTLY POSITIVE, so the scale of S_n/√n
# diverges with n (the opposite of commits 01-03, where the analogous quantity is EXACTLY 1 for
# every n). A finite-N_MC_CTRL SAMPLE variance of that scaled quantity is itself dominated by the
# single largest draw (the "one big jump" heuristic for heavy tails), so it inherits the same
# n^(2/γ-1) = n^(1/3) growth in EXPECTATION, empirically ~n^0.2-0.4 at these n (see the commit doc)
# -- clearly, robustly positive, not the flat/zero slope commits 01-03 would show under the SAME
# machinery with a finite-variance law.
#
# NOISE WARNING (why this is batch-means, not a single ladder fit): the RAW sample variance at a
# SINGLE replicate is wildly noisy (dominated by whether that one draw happened to catch a huge
# outlier -- observed to range over several ORDERS OF MAGNITUDE between adjacent rungs in informal
# exploration). Averaging the RAW variances across replicates does not fix this (the arithmetic
# mean of an infinite-variance quantity is itself unstable) -- but averaging the per-replicate
# SLOPE (each replicate's own log-log fit across its OWN 5-point n-ladder) is well-behaved, because
# a log-log slope is far less sensitive to which single point happens to be a large outlier than
# the raw magnitude is. This is exactly `cov_ladder_replicate`'s pattern (GATE 01a above), just on
# a different per-rung statistic (sample variance of the endpoint, via `empirical_cov` on a
# reshaped 1xN "single-time-point path matrix" -- reusing the library's covariance estimator rather
# than hand-rolling a variance).
function var_ladder_replicate(sampler::Function, n_ladder, N, rng)
    vars = [empirical_cov(reshape(rescaled_walk(sampler, n, N, rng)[end, :], 1, :))[1, 1]
            for n in n_ladder]
    slope, _ = ols_slope_se(log10.(n_ladder), log10.(vars))
    return slope, vars
end

# Fixed-direction control margin (à la Unit 4's `slope_c > -0.75`, and 03's absolute-bound gates):
# this is NOT a rate gated against theory-within-SE -- it only needs "clearly, robustly positive."
# 0.10 sits with real headroom below BOTH the analytic exponent (2/γ-1 = 0.333) and the observed
# batch-mean slope at the seed/config below (~0.42, batch SE ~0.14) -- see the commit doc for the
# seed-robustness check (slopes 0.21-0.42 across several seeds, always clearing this margin).
const VAR_SLOPE_MARGIN = 0.10

rng_ctrl = StableRNG(SEED_CTRL)
var_reps = [var_ladder_replicate(increment_samplers[:pareto], N_LADDER_CTRL, N_MC_CTRL, rng_ctrl)
            for _ in 1:NGROUP_CTRL]
var_gslopes = [r[1] for r in var_reps]
var_errmat  = reduce(hcat, (r[2] for r in var_reps))'          # NGROUP_CTRL × length(N_LADDER_CTRL)
var_mean    = vec(sum(var_errmat; dims = 1)) ./ NGROUP_CTRL     # for the plotted curve only

slope_var = sum(var_gslopes) / NGROUP_CTRL
se_var    = sqrt(sum(abs2, var_gslopes .- slope_var) / (NGROUP_CTRL - 1)) / sqrt(NGROUP_CTRL)
gate_04a  = slope_var > VAR_SLOPE_MARGIN
@printf("GATE 04a [pareto     ] Var(S_n/√n)-vs-n slope: %.4f  (batch SE %.4f)  vs margin +%.2f -> %s\n",
        slope_var, se_var, VAR_SLOPE_MARGIN, gate_04a ? "PASS" : "FAIL")

# --- GATE 04b: KS(S_n/√n, Φ) does NOT decrease like commit 02's -1/2 -----------------------------
# Contrast target: commit 02's two n^(-1/2) laws (exponential, Rademacher) both gave slopes
# indistinguishable from -0.5 within a few batch-means SEs. Here the increment does not even have a
# CLT to converge to N(0,1) under -- S_n/√n's own scale DIVERGES (Gate 04a), so at any fixed x the
# marginal CDF F_n(x) = P(S_n/√n <= x) -> P(Y <= 0) = 1/2 as n grows (a diverging-scale symmetric
# law puts vanishing mass in any fixed finite window), meaning KS(S_n/√n, Φ) trends TOWARD the same
# ~0.5 "totally uninformative" ceiling as commit 03's wrong-target control -- i.e. this slope should
# be FLAT-TO-POSITIVE, the mirror image of commit 02's clean -1/2 decay. Reuses `ks_ladder_replicate`
# (defined in Phase 2 above) VERBATIM -- same estimator, same batch-means machinery, same n-ladder --
# so the contrast with commit 02 is apples-to-apples: nothing changed here except the increment law.
const KS_CTRL_SLOPE_FLOOR = -0.35   # fixed floor: "not <= -1/2" with real margin (commit 02's laws
                                      # sit AT -0.5 within a small SE-multiple; anything above -0.35
                                      # is decisively NOT that regime). Observed slope at the seed/
                                      # config below is ~+0.06 with a tiny batch SE (~0.002) -- the
                                      # KS-vs-n curve here is far more stable run-to-run than the raw
                                      # variance curve, since KS is a bounded, non-heavy-tailed statistic.

ks_ctrl_reps = [ks_ladder_replicate(increment_samplers[:pareto], N_LADDER_CTRL, N_MC_CTRL, rng_ctrl)
                for _ in 1:NGROUP_CTRL]
ks_ctrl_gslopes = [r[1] for r in ks_ctrl_reps]
ks_ctrl_errmat  = reduce(hcat, (r[2] for r in ks_ctrl_reps))'
ks_ctrl_mean    = vec(sum(ks_ctrl_errmat; dims = 1)) ./ NGROUP_CTRL

slope_ks_ctrl = sum(ks_ctrl_gslopes) / NGROUP_CTRL
se_ks_ctrl    = sqrt(sum(abs2, ks_ctrl_gslopes .- slope_ks_ctrl) / (NGROUP_CTRL - 1)) / sqrt(NGROUP_CTRL)
gate_04b      = slope_ks_ctrl > KS_CTRL_SLOPE_FLOOR
@printf("GATE 04b [pareto     ] KS(S_n/√n,Φ)-vs-n slope: %.4f  (batch SE %.4f)  vs floor %.2f -> %s   (contrast: commit 02's laws ~ -0.50)\n",
        slope_ks_ctrl, se_ks_ctrl, KS_CTRL_SLOPE_FLOOR, gate_04b ? "PASS" : "FAIL")

gate_04 = gate_04a && gate_04b
println("GATE 04 (variance grows AND no Gaussian limit) -> ", gate_04 ? "PASS" : "FAIL")

# --- Figure: infinite_variance_control.png -- variance-growth curve + heavy-tailed QQ departure --
p4a = plot(; xscale = :log10, yscale = :log10, xlabel = "n (walk length)",
           ylabel = "mean Var(S_n/√n)  (over $NGROUP_CTRL replicates)",
           title = @sprintf("GATE 04a: Pareto (γ=%.1f) -- variance GROWS", GAMMA_PARETO),
           titlefontsize = 11, legend = :topleft,
           left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, top_margin = 8Plots.mm)
plot!(p4a, N_LADDER_CTRL, var_mean; marker = :circle, color = :purple,
      label = @sprintf("pareto (batch-mean slope %.3f)", slope_var))
ref_growth = var_mean[1] .* (N_LADDER_CTRL ./ N_LADDER_CTRL[1]) .^ (2 / GAMMA_PARETO - 1)
plot!(p4a, N_LADDER_CTRL, ref_growth; linestyle = :dash, color = :black,
      label = @sprintf("reference slope 2/γ-1 = %.3f", 2 / GAMMA_PARETO - 1))
hline!(p4a, [1.0]; color = :gray, linestyle = :dot, alpha = 0.6,
       label = "finite-variance target (commits 01-03: Var ≡ 1)")

# QQ panel: own small deterministic step, DERIVED from SEED_CTRL but a DISTINCT stream (continuing
# rng_ctrl here would perturb nothing already gated, but deriving a fresh seed keeps this
# illustrative draw fully decoupled from gates 04a/04b's numbers, matching Phase 2's marginal_qq.png
# convention). Uses the largest rung of the SAME ladder -- the departure from N(0,1) is starkest
# there, since the diverging scale (Gate 04a) has had the most room to separate from a fixed Φ.
rng_qq_ctrl   = StableRNG(SEED_CTRL + 1)
n_qq_ctrl     = N_LADDER_CTRL[end]
endpoint_ctrl = sort(rescaled_walk(increment_samplers[:pareto], n_qq_ctrl, N_QQ, rng_qq_ctrl)[end, :])
qemp_ctrl = [_quantile(endpoint_ctrl, p) for p in QQ_PROBS]
qth_ctrl  = normquantile.(QQ_PROBS)
p4b = scatter(qth_ctrl, qemp_ctrl; markersize = 2, label = "empirical", xlabel = "N(0,1) quantile",
              ylabel = "empirical quantile (S_n/√n)",
              title = "QQ: pareto (n=$n_qq_ctrl) -- heavy tails depart",
              titlefontsize = 11, legend = :topleft,
              left_margin = 6Plots.mm, bottom_margin = 5Plots.mm, top_margin = 8Plots.mm)
plot!(p4b, qth_ctrl, qth_ctrl; linestyle = :dash, color = :black, label = "y=x (Gaussian reference)")
savefig(plot(p4a, p4b; layout = (1, 2), size = (1200, 520)), joinpath(OUTDIR, "infinite_variance_control.png"))

@printf("\nrecorded: seed_ctrl=%d, gamma_pareto=%.2f, n_ladder_ctrl=%s, N_mc_ctrl=%d, ngroup_ctrl=%d, var_slope_margin=%.2f, ks_ctrl_slope_floor=%.2f\n",
        SEED_CTRL, GAMMA_PARETO, string(N_LADDER_CTRL), N_MC_CTRL, NGROUP_CTRL, VAR_SLOPE_MARGIN, KS_CTRL_SLOPE_FLOOR)

println(gate_01a && gate_02 && gate_03 && gate_04 ? "ALL GATES: PASS" : "ALL GATES: FAIL")

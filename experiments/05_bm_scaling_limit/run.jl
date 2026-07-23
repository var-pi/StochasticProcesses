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
using SpecialFunctions: erf

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
println(gate_01a ? "ALL GATES: PASS" : "ALL GATES: FAIL")

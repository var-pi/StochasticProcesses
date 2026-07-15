# ============================================================================
#  Experiment 00 — the covariance core
# ----------------------------------------------------------------------------
#  The claim we demonstrate: if we assemble the true covariance matrix Σ of a
#  Gaussian process, draw many sample paths from it, and estimate Σ back from
#  those paths (empirical_cov), then the estimate converges to Σ as we use more
#  paths. Specifically the Frobenius error ‖Σ̂_N - Σ‖ shrinks at the Monte-Carlo
#  rate ∝ N^(-1/2) — a slope of -1/2 on a log-log plot of error against N.
#
#  We use Brownian motion (Σ_ij = min(t_i, t_j)) because its grid-through-zero Σ
#  is exactly singular — which makes it both the main check and, at the end, a
#  negative control for the Cholesky nugget.
#
#  Run:  julia --project=experiments experiments/00_covariance_core/run.jl
#  This is a Monte-Carlo study, so it is NOT run in CI; the two figures it writes
#  are committed. Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================

using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: let GR write PNGs with no display (CI/agent shells)
gr()

const SEED     = 20240501
const JITTER   = 1e-10
const N_GRID   = 64
const N_LADDER = [100, 316, 1000, 3162, 10000]      # N spanning 1e2 .. 1e4, ~evenly in log
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

# --- Self-contained ordinary-least-squares: fitted slope and its standard error ---
# We fit log(error) against log(N); the slope is the convergence rate and its standard
# error tells us how tight the estimate is (used to gate the slope against -1/2 below).
function ols_slope_se(x::AbstractVector, y::AbstractVector)
    n = length(x)
    x̄ = sum(x) / n
    ȳ = sum(y) / n
    Sxx = sum((x .- x̄) .^ 2)
    slope = sum((x .- x̄) .* (y .- ȳ)) / Sxx
    intercept = ȳ - slope * x̄
    resid = y .- (intercept .+ slope .* x)
    s2 = sum(abs2, resid) / (n - 2)        # residual variance, n-2 degrees of freedom
    se = sqrt(s2 / Sxx)                     # standard error of the fitted slope
    return slope, se
end

# --- Setup ---------------------------------------------------------------------
rng    = StableRNG(SEED)
t_grid = range(0, 1; length = N_GRID)
gp     = GaussianProcess(brownian_motion_kernel)
Sigma  = assemble_cov(gp, t_grid)
Sigma_dense = Matrix(Sigma)

function draw_paths(N)
    P = Matrix{Float64}(undef, N_GRID, N)        # n_grid × N: one path per column
    for j in 1:N
        P[:, j] = sample_cholesky(Sigma, rng; jitter = JITTER)
    end
    return P
end

# --- (i) Sample-paths figure ---------------------------------------------------
# Draw order matters: the 5 demo paths here and the N-ladder below both pull from the
# same StableRNG(SEED) stream, so drawing the demo first is part of what fixes every
# committed number and figure. Do NOT reorder these two blocks.
paths_demo = draw_paths(5)
p1 = plot(t_grid, paths_demo;
          xlabel = "t", ylabel = "X(t)", legend = false,
          title = "Brownian motion sample paths (Cholesky sampler)")
savefig(p1, joinpath(OUTDIR, "sample_paths.png"))

# --- (ii) Convergence study ----------------------------------------------------
# For each N, draw N paths, estimate Σ, and record the Frobenius error against the true Σ.
errors = Float64[]
println("  N        ||Sigma_hat - Sigma||_F")
for N in N_LADDER
    paths = draw_paths(N)
    @assert size(paths, 1) == length(t_grid)     # orientation guard: rows are grid points
    err = norm(empirical_cov(paths) .- Sigma_dense)   # Frobenius norm of the error matrix
    push!(errors, err)
    @printf("  %-7d  %.6e\n", N, err)
end

logN   = log10.(N_LADDER)
logErr = log10.(errors)
slope, se = ols_slope_se(logN, logErr)
@printf("\nfitted slope = %.4f +/- %.4f (SE);  target = -0.5\n", slope, se)

# Pass condition: the fitted slope agrees with the theoretical -1/2 to within a small
# multiple of its own standard error (a stochastic slope, so we compare to its SE, not to
# a fixed absolute tolerance).
gate_pass = abs(slope + 0.5) < 2.5 * se
println(gate_pass ? "GATE: PASS  (|slope + 1/2| < 2.5*SE)" :
                    "GATE: FAIL  (|slope + 1/2| >= 2.5*SE)")

# error-vs-N log–log plot with the -1/2 reference line
ref = errors[1] .* (N_LADDER ./ N_LADDER[1]) .^ (-0.5)
p2 = plot(N_LADDER, errors;
          xscale = :log10, yscale = :log10, marker = :circle,
          xlabel = "N", ylabel = "Frobenius error  ||Sigma_hat - Sigma||_F",
          label = @sprintf("empirical (slope %.3f)", slope),
          title = "Monte-Carlo convergence of the empirical covariance")
plot!(p2, N_LADDER, ref; linestyle = :dash, label = "reference slope -1/2")
savefig(p2, joinpath(OUTDIR, "error_vs_N.png"))

# Sanity check: the smallest error is far larger than the jitter, so the convergence is
# genuine and not artificially floored by the nugget.
@printf("min error %.3e  >>  jitter %.1e  (error is not epsilon-floored)\n",
        minimum(errors), JITTER)

# --- (iii) Negative control: jitter = 0 on the Brownian-motion Σ ---------------
# Reuse the very Σ from the main check: its all-zero t = 0 row (R(0,s) = 0) makes it exactly
# singular, so Cholesky must throw at jitter = 0. This shows the nugget is load-bearing for
# the real matrix, not for some separately-constructed pathological one.
try
    sample_cholesky(Sigma, StableRNG(SEED); jitter = 0.0)
    println("NEGATIVE CONTROL: FAIL - expected PosDefException at jitter = 0")
catch e
    e isa PosDefException ?
        println("NEGATIVE CONTROL: PASS - caught PosDefException at jitter = 0") :
        rethrow(e)
end

@printf("\nrecorded: seed = %d, jitter = %.1e, n_grid = %d\n", SEED, JITTER, N_GRID)

# Unit 0 — covariance core: Monte-Carlo convergence of the empirical covariance.
# Run:  julia --project experiments/00_covariance_core/run.jl
# NOT run in CI (Monte-Carlo); the two figures it writes are committed.

using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: let GR write PNGs with no display (CI/agent shells)
gr()

const SEED     = 20240501
const JITTER   = 1e-10
const N_GRID   = 64
const N_LADDER = [100, 316, 1000, 3162, 10000]      # N in [1e2, 1e4]
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

# --- Self-contained OLS: fitted slope and its standard error -------------------
function ols_slope_se(x::AbstractVector, y::AbstractVector)
    n = length(x)
    x̄ = sum(x) / n
    ȳ = sum(y) / n
    Sxx = sum((x .- x̄) .^ 2)
    slope = sum((x .- x̄) .* (y .- ȳ)) / Sxx
    intercept = ȳ - slope * x̄
    resid = y .- (intercept .+ slope .* x)
    s2 = sum(abs2, resid) / (n - 2)        # residual variance, n-2 dof
    se = sqrt(s2 / Sxx)                     # SE of the fitted slope
    return slope, se
end

# --- Setup ---------------------------------------------------------------------
rng    = StableRNG(SEED)
t_grid = range(0, 1; length = N_GRID)
gp     = GaussianProcess(brownian_motion_kernel)
Sigma  = assemble_cov(gp, t_grid)
Sigma_dense = Matrix(Sigma)

function draw_paths(N)
    P = Matrix{Float64}(undef, N_GRID, N)        # n_grid × N: one path per COLUMN
    for j in 1:N
        P[:, j] = sample_cholesky(Sigma, rng; jitter = JITTER)
    end
    return P
end

# --- (i) Sample-paths figure ---------------------------------------------------
# Draw order (this demo of 5, THEN the ladder below) is fixed for reproducibility:
# both pull from the same StableRNG(SEED) stream, so reordering silently changes
# every committed number and figure. Do NOT reorder.
paths_demo = draw_paths(5)
p1 = plot(t_grid, paths_demo;
          xlabel = "t", ylabel = "X(t)", legend = false,
          title = "Brownian motion sample paths (Cholesky sampler)")
savefig(p1, joinpath(OUTDIR, "sample_paths.png"))

# --- (ii) Convergence study ----------------------------------------------------
errors = Float64[]
println("  N        ||Sigma_hat - Sigma||_F")
for N in N_LADDER
    paths = draw_paths(N)
    @assert size(paths, 1) == length(t_grid)     # orientation guard (Phase 3 pins this)
    err = norm(empirical_cov(paths) .- Sigma_dense)   # Frobenius norm
    push!(errors, err)
    @printf("  %-7d  %.6e\n", N, err)
end

logN   = log10.(N_LADDER)
logErr = log10.(errors)
slope, se = ols_slope_se(logN, logErr)
@printf("\nfitted slope = %.4f +/- %.4f (SE);  target = -0.5\n", slope, se)

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

# confirm the error is not floored by the jitter epsilon
@printf("min error %.3e  >>  jitter %.1e  (error is not epsilon-floored)\n",
        minimum(errors), JITTER)

# --- (iii) Negative control: jitter = 0 on the BM Σ ----------------------------
# Reuse the main-check Sigma: its all-zero t=0 row (R(0,s)=0) makes it exactly
# singular, so cholesky must throw at jitter = 0 -- the nugget is load-bearing for
# the very Σ used above, not for a separate pathological matrix.
try
    sample_cholesky(Sigma, StableRNG(SEED); jitter = 0.0)
    println("NEGATIVE CONTROL: FAIL - expected PosDefException at jitter = 0")
catch e
    e isa PosDefException ?
        println("NEGATIVE CONTROL: PASS - caught PosDefException at jitter = 0") :
        rethrow(e)
end

@printf("\nrecorded: seed = %d, jitter = %.1e, n_grid = %d\n", SEED, JITTER, N_GRID)

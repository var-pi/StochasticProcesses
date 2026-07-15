# ============================================================================
#  Experiment 02 — Karhunen-Loève via a quadrature eigenproblem
# ----------------------------------------------------------------------------
#  The second diagonalization of the covariance operator. Where Unit 1 put the
#  spectrum on the frequency axis (Bochner), this unit finds the operator's own
#  eigenbasis (Mercer/KL) by a symmetrized Nystrom quadrature eigenproblem, and
#  checks it against Brownian motion's closed form (Pavliotis Example 1.29):
#      lambda_n = (2/((2n-1)pi))^2,   e_n(t) = sqrt(2) sin((n-1/2)pi t).
#
#  Four things are demonstrated (all deterministic -- no RNG in the gates):
#    (i)   eigenpairs match the closed form, and lambda_k ~ (2k-1)^-2 (slope -2);
#    (ii)  the trace identity sum(lambda) = int_0^T R(t,t) dt pins assembly
#          (= T^2/2 = 1/2 for BM, NOT T*R(0)=0);
#    (iii) the torus coincidence lambda_k = Rhat(k) holds EXACTLY on the circle
#          (periodic_kernel) and FAILS on a bounded interval (the control);
#    (iv)  the KL truncation cost/accuracy trade-off, and KL vs Cholesky cost.
#
#  Run:  julia --project=experiments experiments/02_kl_quadrature/run.jl
#  Heavy (a 4096-point dense eigensolve), so NOT run in CI; figures are committed.
#  Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================

using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: GR writes PNGs with no display (CI/agent shells)
gr()

const T        = 1.0        # domain [0,T]; Pavliotis fixes T=1 in section 1.5
const N_BM     = 400        # Nystrom grid for the Brownian-motion eigenproblem
const N_TORUS  = 4096       # torus grid; the coincidence residual is aliasing O(1/n^2)
const ALPHA    = 1.0        # periodic_kernel decay rate
const KFIT     = 40         # slope-fit window k=1..KFIT, well above the discretization floor
const N_COST   = 256        # grid for the KL-vs-Cholesky cost comparison
const N_PATHS  = 2000       # draws timed in the cost study
const K_TRUNC  = 20         # truncation level for the cost demo
const JITTER   = 1e-10      # Cholesky nugget for the cost-study factorization (reported below)
const SEED     = 20240715   # only the cost-study draws are stochastic
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

# Brownian-motion closed form (Example 1.29). efun_bm is evaluated on a given grid.
lam_bm(k)          = (2 / ((2k - 1) * pi))^2
efun_bm(k, grid)   = sqrt(2) .* sin.((k - 0.5) * pi .* grid)
# periodic_kernel Fourier coefficients = its torus KL eigenvalues (see the kernel docstring).
Rhat(k) = 2 * ALPHA * (1 - (-1)^k * exp(-ALPHA / 2)) / (ALPHA^2 + (2pi * k)^2)

# --- (0) Solve the Brownian-motion eigenproblem -------------------------------------
nodes, w = quad_nodes_weights(T; n = N_BM)                 # trapezoid on [0,T]
lambdas, eigfuncs = nystrom_eigen(brownian_motion_kernel, nodes, w)

# --- (i) Eigenpairs vs the closed form; the (2k-1)^-2 decay slope --------------------
relerr   = [abs(lambdas[k] - lam_bm(k)) / lam_bm(k) for k in 1:min(200, N_BM)]
resolved = maximum(findall(<(0.10), relerr))              # last k with <10% relative error
# discrete W-weighted L2 error of the first few eigenfunctions, sign-aligned to the analytic sine
efun_L2 = Float64[]
for k in 1:3
    e = copy(eigfuncs[:, k]); ea = efun_bm(k, nodes)
    dot(w .* e, ea) < 0 && (e .*= -1)
    push!(efun_L2, sqrt(sum(w .* (e .- ea).^2)))
end
# Fit log(lambda) vs log(2k-1): the odd Sturm-Liouville wavenumber, for which lambda ∝ (2k-1)^-2 is
# EXACT (slope -2). Fitting vs log(k) instead reads ~-2.21 from small-k (2k-1 vs 2k) curvature -- a
# real teaching point, not the gate abscissa.
ks     = 1:KFIT
Xdes   = hcat(ones(KFIT), log.(2 .* collect(ks) .- 1))
slope  = (Xdes \ log.(lambdas[ks]))[2]
slope_naive = (hcat(ones(KFIT), log.(collect(ks))) \ log.(lambdas[ks]))[2]
gate_slope  = abs(slope + 2) < 0.15
@printf("(i)  eigenpairs: top-5 rel err = %s\n",
        join([@sprintf("%.1e", relerr[k]) for k in 1:5], ", "))
@printf("     resolved range (rel<10%%): k=1..%d ; eigenfunction L2 err (modes 1-3) = %s\n",
        resolved, join([@sprintf("%.1e", e) for e in efun_L2], ", "))
@printf("     slope(log λ vs log(2k-1)) over k=1..%d = %.4f (target -2, gate ±0.15) -> %s\n",
        KFIT, slope, gate_slope ? "PASS" : "FAIL")
@printf("     (contrast) naive slope vs log(k) = %.4f (curved; -> -2 only asymptotically)\n", slope_naive)

# --- (ii) Trace identity sum(lambda) = int_0^T R(t,t) dt ----------------------------
td         = trace_diag(brownian_motion_kernel, nodes, w)  # = T^2/2 = 0.5 for BM, exact on the linear diag
trace_err  = abs(sum(lambdas) - td) / abs(td)
gate_trace = trace_err < 1e-3
@printf("(ii) trace: sum(λ)=%.6f  trace_diag=%.6f  (BM T^2/2=%.3f)  rel err=%.2e -> %s\n",
        sum(lambdas), td, T^2 / 2, trace_err, gate_trace ? "PASS" : "FAIL")
pnodes, pw = quad_nodes_weights(T; n = N_BM, rule = :periodic)
lam_stat, ef_stat = nystrom_eigen((t, s) -> periodic_kernel(t, s; alpha = ALPHA), pnodes, pw)
@printf("     stationary specialization: sum(λ_torus)=%.6f  (T·R(0)=%.3f)\n", sum(lam_stat), T * 1.0)

# --- (iii) Torus coincidence lambda_k = Rhat(k), and its off-torus failure ----------
tnodes, tw = quad_nodes_weights(T; n = N_TORUS, rule = :periodic)
lam_torus, _ = nystrom_eigen((t, s) -> periodic_kernel(t, s; alpha = ALPHA), tnodes, tw)
# Analytic torus spectrum, descending, with the CORRECT circulant multiplicities: DC (k=0) and
# Nyquist (k=N_TORUS/2) once, every interior k twice (±k degenerate). Length is exactly N_TORUS.
ana_torus  = sort(vcat(Rhat(0), repeat([Rhat(k) for k in 1:div(N_TORUS, 2) - 1], inner = 2),
                       Rhat(div(N_TORUS, 2))); rev = true)
torus_err  = maximum(abs.(lam_torus .- ana_torus))
gate_torus = torus_err < 1e-7
@printf("(iii) torus coincidence (n=%d): max_k|λ_k - Rhat(k)| = %.3e  (gate <1e-7) -> %s\n",
        N_TORUS, torus_err, gate_torus ? "PASS" : "FAIL")
# NEGATIVE CONTROL: the SAME kernel on the sub-period interval [0,1/2] (where it is just the OU
# kernel) no longer matches Rhat -- the coincidence is a torus fact, broken by a boundary.
inodes, iw = quad_nodes_weights(T / 2; n = N_TORUS)        # Dirichlet interval, half a period
lam_int, _ = nystrom_eigen((t, s) -> periodic_kernel(t, s; alpha = ALPHA), inodes, iw)
ctrl_gap   = maximum(abs.(lam_int[1:20] .- ana_torus[1:20]))
@printf("      control: interval [0,%.2f] max_k|λ_k - Rhat(k)| (top 20) = %.3e (>> torus: fails on purpose)\n",
        T / 2, ctrl_gap)

# --- (iv) Truncation tail energy; KL vs Cholesky cost -------------------------------
println("(iv) truncation tail energy (fraction of variance discarded):")
Ks_tail = (1, 2, 5, 10, 20, 40)
tails   = [kl_tail_energy(lambdas, K) for K in Ks_tail]
for (K, tl) in zip(Ks_tail, tails)
    @printf("     K=%2d  tail=%.4f\n", K, tl)
end
# Cost: setup (factorization) + per-draw, KL (full & truncated) vs Cholesky, on an N_COST grid.
cn, cw = quad_nodes_weights(T; n = N_COST)
Sig    = Matrix(assemble_cov(GaussianProcess(brownian_motion_kernel), cn))
_ks    = @timed nystrom_eigen(brownian_motion_kernel, cn, cw); lc, efc = _ks.value; t_kl_setup = _ks.time
_cs    = @timed cholesky(Symmetric(Sig) + JITTER * I(N_COST)).L; Lc = _cs.value; t_chol_setup = _cs.time
rng    = StableRNG(SEED)
sample_kl(lc, efc, rng); Lc * randn(rng, N_COST)          # warm up (exclude JIT from the timings)
t_kl_full  = @elapsed for _ in 1:N_PATHS; sample_kl(lc, efc, rng); end
t_kl_trunc = @elapsed for _ in 1:N_PATHS; sample_kl(lc[1:K_TRUNC], efc[:, 1:K_TRUNC], rng); end
t_chol     = @elapsed for _ in 1:N_PATHS; Lc * randn(rng, N_COST); end
@printf("     cost (n=%d, %d draws): setup KL=%.3fs Chol=%.3fs | draws KL(full)=%.3fs KL(K=%d)=%.3fs Chol=%.3fs\n",
        N_COST, N_PATHS, t_kl_setup, t_chol_setup, t_kl_full, K_TRUNC, t_kl_trunc, t_chol)

# --- Figures -----------------------------------------------------------------------
# (A) eigenfunctions.png: numeric e_k (solid) vs analytic sqrt(2) sin((k-1/2)pi t) (dashed).
pA = plot(; xlabel = "t", ylabel = "e_k(t)", title = "Brownian-motion KL eigenfunctions")
for k in 1:3
    e = copy(eigfuncs[:, k]); ea = efun_bm(k, nodes)
    dot(w .* e, ea) < 0 && (e .*= -1)
    plot!(pA, nodes, e; label = "e_$k (numeric)")
    plot!(pA, nodes, ea; linestyle = :dash, label = "e_$k (analytic)")
end
savefig(pA, joinpath(OUTDIR, "eigenfunctions.png"))

# (B) eigenvalue_decay.png: lambda_k vs the odd wavenumber (2k-1), log-log, with the fitted -2 slope,
#     the resolved fit window shaded, and the discretization floor visible where numeric flattens.
kk   = 1:min(200, N_BM)
xw   = 2 .* collect(kk) .- 1
pB = plot(xw, lambdas[kk]; seriestype = :scatter, markersize = 2, xscale = :log10, yscale = :log10,
          label = "λ_k (numeric)", xlabel = "wavenumber 2k-1", ylabel = "λ_k",
          title = @sprintf("KL eigenvalue decay: slope %.3f (target -2)", slope))
plot!(pB, xw, lam_bm.(kk); linestyle = :dash, label = "λ_k = (2/((2k-1)π))² (analytic)")
vspan!(pB, [1, 2 * KFIT - 1]; alpha = 0.12, label = "fit window k≤$KFIT")
# Annotate the discretization noise floor: past k≈resolved the numeric λ_k plateaus and the decay
# flattens away from the analytic line (the spec's second negative control, made visible).
vline!(pB, [2 * resolved - 1]; linestyle = :dot, label = @sprintf("floor: resolved to k=%d", resolved))
savefig(pB, joinpath(OUTDIR, "eigenvalue_decay.png"))

# (C) torus_vs_interval.png: TWO panels. Left: the eigenvalue gap |λ_k - Rhat(k)| for the torus
#     (tiny) vs the interval control (O(1)). Right: the MODE SHAPES behind that gap — torus
#     eigenfunctions are Fourier characters (full-period sinusoids), while the interval eigenfunctions
#     are Sturm–Liouville sines √2 sin((k-½)πt). Same operator machinery, different boundary → a
#     different basis, which is *why* λ_k = Rhat(k) holds on the circle and fails on the interval.
kc = 1:20
pC1 = plot(kc, abs.(lam_torus[kc] .- ana_torus[kc]); yscale = :log10, marker = :circle,
           label = "torus (periodic) — coincides", xlabel = "k", ylabel = "|λ_k - Rhat(k)|",
           title = "Coincidence vs failure")
plot!(pC1, kc, abs.(lam_int[kc] .- ana_torus[kc]); marker = :square, label = "interval [0,½] — fails")
# The interval representatives here are the BM Example-1.29 eigenfunctions on [0,1] (the canonical
# Sturm-Liouville sines) -- NOT the [0,½] OU control plotted at left; both are stationary-kernel
# interval eigenfunctions, but the cleanest torus-vs-Dirichlet BC contrast puts the S–L sines on the
# same [0,1] axis as the torus character.
pC2 = plot(; xlabel = "t", ylabel = "e(t)", title = "Torus characters vs interval (BM [0,1]) S–L sines")
plot!(pC2, pnodes, ef_stat[:, 2]; label = "torus e₂ (Fourier character)")
plot!(pC2, nodes, eigfuncs[:, 1]; linestyle = :dash, label = "BM e₁ (S–L sine, [0,1])")
plot!(pC2, nodes, eigfuncs[:, 2]; linestyle = :dash, label = "BM e₂ (S–L sine, [0,1])")
pC = plot(pC1, pC2; layout = (1, 2), size = (900, 350))
savefig(pC, joinpath(OUTDIR, "torus_vs_interval.png"))

# (D) truncation_cost.png: tail energy vs K (left), and the per-draw cost bars (right).
pD1 = plot(collect(Ks_tail), tails; yscale = :log10, marker = :circle, legend = false,
           xlabel = "modes kept K", ylabel = "tail energy", title = "KL truncation error")
pD2 = bar(["KL full", "KL K=$K_TRUNC", "Cholesky"], [t_kl_full, t_kl_trunc, t_chol];
          legend = false, ylabel = "time for $N_PATHS draws (s)", title = "Per-draw cost (n=$N_COST)")
pD = plot(pD1, pD2; layout = (1, 2), size = (900, 350))
savefig(pD, joinpath(OUTDIR, "truncation_cost.png"))

# --- Summary -----------------------------------------------------------------------
@printf("\nrecorded: T=%.1f, N_BM=%d, N_TORUS=%d, alpha=%.1f, KFIT=%d, N_COST=%d, chol_jitter=%.0e, seed=%d\n",
        T, N_BM, N_TORUS, ALPHA, KFIT, N_COST, JITTER, SEED)
println(all((gate_slope, gate_trace, gate_torus)) ? "ALL GATES: PASS" : "ALL GATES: FAIL")

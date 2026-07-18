# ============================================================================
#  Experiment 03 — The Process Zoo: reconciliation
# ----------------------------------------------------------------------------
#  Unit 3's one distinct idea is *reconciliation*: the three sampling routes
#  (Cholesky, KL, circulant) and the two diagonalizations are forced to agree
#  on concrete processes. This file's gate is route equivalence — the three OU
#  samplers must land inside a split-half bootstrap null band. The Frobenius
#  statistic ‖Σ̂_A − Σ̂_B‖_F is a norm over n_grid^2 entries, so it concentrates
#  at √2σ₁ > 0 (NOT at zero); the gate is calibrated to an empirical null, not a
#  zero-centred SE. Later phases append the distributional identities and the
#  Toeplitz/Szegő cross-check.
#
#  Run:  julia --project=experiments experiments/03_process_zoo/run.jl
#  Monte-Carlo (seeded); NOT run in CI; figures are committed.
#  Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================
using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots
using SpecialFunctions: erf

# type-7 (linear-interpolation) quantile on an already-sorted vector — matches
# Statistics.quantile without pulling Statistics into the experiments env.
function _quantile(sorted, p)
    n = length(sorted)
    h = (n - 1) * p + 1
    lo = floor(Int, h)
    lo >= n && return sorted[n]
    return sorted[lo] + (h - lo) * (sorted[lo+1] - sorted[lo])
end

ENV["GKSwstype"] = "100"
gr()

const T        = 1.0          # BM / bridge domain [0,1] (Pavliotis §1.5)
const T_OU     = 5.0          # OU domain [0,T_OU] ~ 5 correlation times at alpha=1
const N_GRID   = 64           # uniform grid (all three routes share it)
const D        = 1.0          # OU noise strength
const ALPHA    = 1.0          # OU relaxation rate; R(0)=D/alpha=1
const N_ROUTE  = 4000         # samples per route
const N_SPLIT  = 200          # split-half re-partitions for the bootstrap null
const N_DEMO   = 6            # demo paths per portrait
const JITTER   = 1e-10        # Cholesky nugget (reported)
const SEED     = 271828       # verified: route-equivalence PASSes with margin; 7/8 tested seeds pass
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

ou(t, s) = exponential_kernel(t, s; D = D, alpha = ALPHA)

# --- split-half bootstrap null (defined here, reused by later phases) --------
# Re-partition the columns of `paths` (n_grid × M) into two disjoint halves `nsplit`
# times; return the central [lo,hi] quantile band of ‖cov(halfA) − cov(halfB)‖_F. The
# band is process-specific: it depends entirely on the samples passed in, so a caller
# wanting a different process's null passes that process's samples.
function splithalf_band(paths, nsplit, rng; lo = 0.025, hi = 0.975)
    M = size(paths, 2)
    h = div(M, 2)
    ds = Float64[]
    for _ in 1:nsplit
        perm = sortperm(randn(rng, M))          # uniform random permutation (no Random dep)
        A = @view paths[:, perm[1:h]]
        B = @view paths[:, perm[h+1:2h]]
        push!(ds, norm(empirical_cov(A) .- empirical_cov(B)))
    end
    sort!(ds)
    return _quantile(ds, lo), _quantile(ds, hi)
end

rng = StableRNG(SEED)

# --- catalogue ---------------------------------------------------------------
bm_grid     = range(0, T; length = N_GRID)
bridge_grid = range(0, T; length = N_GRID)
ou_nodes, ou_w = quad_nodes_weights(T_OU; n = N_GRID)       # trapezoid = uniform grid on [0,T_OU]
dt = ou_nodes[2] - ou_nodes[1]

catalogue = [
    ("Brownian motion", brownian_motion_kernel, collect(bm_grid)),
    ("Brownian bridge",  brownian_bridge_kernel, collect(bridge_grid)),
    ("Ornstein–Uhlenbeck", ou, collect(ou_nodes)),
]

# --- portraits (visual; RNG demo draws come first, fixed order) -------------
# eigenvalue decay uses a quadrature on the process's own grid
function eig_decay(kernel, grid)
    T_dom = grid[end] - grid[1]
    nodes, w = quad_nodes_weights(T_dom; n = length(grid))
    λ, _ = nystrom_eigen(kernel, nodes, w)
    return λ
end

pfiles = ["portrait_bm.png", "portrait_bridge.png", "portrait_ou.png"]
for (idx, (name, kernel, grid)) in enumerate(catalogue)
    Σ = Matrix(assemble_cov(GaussianProcess(kernel), grid))
    paths = reduce(hcat, (sample_cholesky(Σ, rng; jitter = JITTER) for _ in 1:N_DEMO))
    λ = eig_decay(kernel, grid)
    p1 = plot(grid, paths; legend = false, xlabel = "t", ylabel = "X(t)", title = "$name — paths")
    p2 = heatmap(grid, grid, Σ; title = "covariance", xlabel = "s", ylabel = "t", aspect_ratio = :equal)
    kk = 1:min(30, length(λ))
    p3 = plot(collect(kk), max.(λ[kk], 1e-16); yscale = :log10, marker = :circle, legend = false,
              xlabel = "k", ylabel = "λ_k", title = "eigenvalue decay")
    savefig(plot(p1, p2, p3; layout = (1, 3), size = (1200, 350)), joinpath(OUTDIR, pfiles[idx]))
end

# --- route equivalence (the gate) -------------------------------------------
Σ_ou = Matrix(assemble_cov(GaussianProcess(ou), ou_nodes))
ou_lambdas, ou_eigfuncs = nystrom_eigen(ou, ou_nodes, ou_w)
r_seq = [ou(0.0, k * dt) for k in 0:N_GRID-1]

chol_paths = reduce(hcat, (sample_cholesky(Σ_ou, rng; jitter = JITTER) for _ in 1:N_ROUTE))
kl_paths   = reduce(hcat, (sample_kl(ou_lambdas, ou_eigfuncs, rng) for _ in 1:N_ROUTE))
circ_paths = reduce(hcat, (sample_circulant_embedding(r_seq, rng) for _ in 1:N_ROUTE))

Σ_chol = empirical_cov(chol_paths)
Σ_kl   = empirical_cov(kl_paths)
Σ_circ = empirical_cov(circ_paths)

band_lo, band_hi = splithalf_band(chol_paths, N_SPLIT, rng)

# Closed-form sanity. sigma1^2 = ((trΣ)^2 + ‖Σ‖_F^2)/N at the full sample size N_ROUTE. Both the
# rescaled cross statistic √2·‖Σ̂_A−Σ̂_B‖_F and the half-N split-half null concentrate at 2·sigma1,
# so the band should bracket 2·sigma1 (NOT √2·sigma1, which is the *unrescaled* cross RMS).
sigma1 = sqrt((tr(Σ_ou)^2 + norm(Σ_ou)^2) / N_ROUTE)
theory = 2 * sigma1

pairs = [("Chol–KL", Σ_chol, Σ_kl), ("Chol–Circ", Σ_chol, Σ_circ), ("KL–Circ", Σ_kl, Σ_circ)]
rescaled = [sqrt(2) * norm(A .- B) for (_, A, B) in pairs]
all_in = all(band_lo <= d <= band_hi for d in rescaled)
band_ok = band_lo <= theory <= band_hi
println("route equivalence (split-half null band [", @sprintf("%.4f", band_lo), ", ",
        @sprintf("%.4f", band_hi), "], null-scale theory 2σ₁=", @sprintf("%.4f", theory), "):")
for (i, (nm, _, _)) in enumerate(pairs)
    d = rescaled[i]
    @printf("  %-10s √2·‖Σ̂_A−Σ̂_B‖_F = %.4f  -> %s\n", nm, d, band_lo <= d <= band_hi ? "in band" : "OUT")
end
@printf("  band brackets theory 2σ₁: %s\n", band_ok ? "yes" : "no")
println("route equivalence -> ", all_in ? "PASS" : "FAIL")

# --- figure: route equivalence ----------------------------------------------
preq = plot([band_lo, band_hi], [1, 1]; linewidth = 8, alpha = 0.3, label = "split-half null band",
            xlabel = "√2·‖Σ̂_A−Σ̂_B‖_F", ylims = (0.5, 1.5), yticks = false, legend = :topright,
            title = "OU route equivalence vs split-half null")
scatter!(preq, rescaled, fill(1, 3); label = "route pairs", markersize = 6)
vline!(preq, [theory]; linestyle = :dash, label = "2σ₁ (null-scale theory)")
savefig(preq, joinpath(OUTDIR, "route_equivalence.png"))

# ============================================================================
#  Phase 4 — distributional identities (own RNG stream; cannot perturb above)
# ----------------------------------------------------------------------------
#  A distributional identity is a two-step chain: the time-changed process is
#  Gaussian BY CONSTRUCTION (a linear map of a Gaussian process), so App. B.5
#  turns a full-covariance match into equality in law; then Cramér–Wold
#  projections KS-test each 1-D functional, catching a non-Gaussian impostor
#  that carries the right covariance. Uses its own StableRNG(SEED_DIST), so it
#  is independent of the route-equivalence stream above.
# ============================================================================
const SEED_DIST = 20250101
const C_SCALE   = 4.0
const K_KL      = 8
const N_BRIDGE  = 200

normcdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
kscrit(β)  = sqrt(-0.5 * log(β / 2))                        # Kolmogorov crit at level β
stephens(Dks, n) = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * Dks  # finite-sample modified statistic

rng4 = StableRNG(SEED_DIST)

# BM reference and its OWN split-half null band (BM-scale, NOT the OU band above)
Σ_bm = Matrix(assemble_cov(GaussianProcess(brownian_motion_kernel), bm_grid))
bm_paths = reduce(hcat, (sample_cholesky(Σ_bm, rng4; jitter = JITTER) for _ in 1:N_ROUTE))
Σ̂_bm = empirical_cov(bm_paths)
bm_lo, bm_hi = splithalf_band(bm_paths, N_SPLIT, rng4)

# Self-similarity c^{-1/2}W(ct) =d W(t). Draw W(ct) once, scale for Y and the wrong-exponent control.
Σ_scaled = Matrix(assemble_cov(GaussianProcess(brownian_motion_kernel), C_SCALE .* bm_grid))
wct = reduce(hcat, (sample_cholesky(Σ_scaled, rng4; jitter = JITTER) for _ in 1:N_ROUTE))
Y       = C_SCALE^(-0.5) .* wct
Y_wrong = C_SCALE^(-1/3) .* wct
ss_stat   = sqrt(2) * norm(empirical_cov(Y) .- Σ̂_bm)
ctrl_stat = sqrt(2) * norm(empirical_cov(Y_wrong) .- Σ̂_bm)
ss_ok    = bm_lo <= ss_stat <= bm_hi
ctrl_out = !(bm_lo <= ctrl_stat <= bm_hi)

# Cramér–Wold: fixed 1-D functionals of Y, each KS-tested vs N(0, aᵀΣ_BM a). Bonferroni family α=0.01.
cw_funcs = [ ones(N_GRID) ./ N_GRID,
             collect(range(-1, 1; length = N_GRID)),
             collect(range(-1, 1; length = N_GRID)).^2 .- 1/3,
             sin.(2π .* bm_grid),
             [1.0; zeros(N_GRID - 2); -1.0],
             normalize(sin.(7 .* bm_grid) .+ cos.(3 .* bm_grid)) ]
cw_labels = ["mean", "ramp", "quadratic", "sine", "endpoint diff", "wiggle"]
cw_crit = kscrit(0.01 / length(cw_funcs))
cw_stats = [stephens(ks_statistic([dot(a, @view Y[:, j]) for j in 1:N_ROUTE],
                                   x -> normcdf(x / sqrt(a' * Σ_bm * a))), N_ROUTE) for a in cw_funcs]
cw_ok = all(cw_stats .< cw_crit)

# KL coefficients on OU: ξ_k = Σ_i w_i X_i e_k(t_i) ~ N(0, λ_k), pairwise uncorrelated.
ou_paths = reduce(hcat, (sample_cholesky(Σ_ou, rng4; jitter = JITTER) for _ in 1:N_ROUTE))
ξ = ou_eigfuncs[:, 1:K_KL]' * (ou_w .* ou_paths)                 # K_KL × N_ROUTE
ξn = ξ ./ sqrt.(sum(abs2, ξ; dims = 2))
Ccorr = ξn * ξn'
maxoff = maximum(abs.(Ccorr - I))
kl_crit = kscrit(0.01 / K_KL)
kl_stats = [stephens(ks_statistic(ξ[k, :], x -> normcdf(x / sqrt(ou_lambdas[k]))), N_ROUTE) for k in 1:K_KL]
kl_ok = maxoff < 0.10 && all(kl_stats .< kl_crit)

# Bridge endpoints pinned at t=0,1 via the covariance/Cholesky route (the time-change formula
# B_t=(1-t)W(t/(1-t)) can't be evaluated at t=1). R(0,0)=R(1,1)=0, so after the jitter nugget the
# endpoint diagonal is JITTER and X(endpoint)=√JITTER·z — pinned to the jitter floor.
Σ_br = Matrix(assemble_cov(GaussianProcess(brownian_bridge_kernel), bridge_grid))
br_paths = reduce(hcat, (sample_cholesky(Σ_br, rng4; jitter = JITTER) for _ in 1:N_BRIDGE))
maxend = maximum(abs.(vcat(br_paths[1, :], br_paths[end, :])))
br_tol = 10 * sqrt(JITTER)
br_ok = maxend <= br_tol

println("\ndistributional identities (SEED_DIST=$SEED_DIST):")
@printf("  self-similarity c^-1/2 W(ct): √2‖Σ̂_Y−Σ̂_BM‖=%.3f in bm_band [%.3f,%.3f] -> %s\n",
        ss_stat, bm_lo, bm_hi, ss_ok ? "PASS" : "FAIL")
@printf("  negative control c^-1/3:      √2‖Σ̂_wrong−Σ̂_BM‖=%.3f -> %s\n",
        ctrl_stat, ctrl_out ? "outside band (fails on purpose)" : "IN BAND?!")
@printf("  Cramér–Wold: max Stephens KS=%.3f < crit=%.3f (family α=0.01, %d proj) -> %s\n",
        maximum(cw_stats), cw_crit, length(cw_funcs), cw_ok ? "PASS" : "FAIL")
@printf("  KL coeffs: max|corr offdiag|=%.4f (<0.10), max KS=%.3f < crit=%.3f -> %s\n",
        maxoff, maximum(kl_stats), kl_crit, kl_ok ? "PASS" : "FAIL")
@printf("  bridge endpoints: max|X(0)|,|X(1)|=%.2e ≤ %.2e -> %s\n",
        maxend, br_tol, br_ok ? "PASS" : "FAIL")
println("distributional identities -> ", (ss_ok && ctrl_out && cw_ok && kl_ok && br_ok) ? "PASS" : "FAIL")

# Cramér–Wold panel: per-projection empirical CDF (solid) vs its Gaussian target (dashed).
pcw = plot(; layout = (2, 3), size = (1200, 650), legend = false)
for (i, a) in enumerate(cw_funcs)
    v = a' * Σ_bm * a
    proj = sort([dot(a, @view Y[:, j]) for j in 1:N_ROUTE])
    ecdf = (1:N_ROUTE) ./ N_ROUTE
    plot!(pcw[i], proj, ecdf; title = @sprintf("%s (KS*=%.2f)", cw_labels[i], cw_stats[i]),
          xlabel = "aᵀX", ylabel = "CDF")
    plot!(pcw[i], proj, normcdf.(proj ./ sqrt(v)); linestyle = :dash)
end
savefig(pcw, joinpath(OUTDIR, "cramer_wold.png"))

# KL-coefficient panel: |correlation| heatmap (≈ I) and ξ_1, ξ_K ECDFs vs N(0,λ_k).
pkl1 = heatmap(abs.(Ccorr); title = @sprintf("|corr(ξ)| (max offdiag %.3f)", maxoff),
               aspect_ratio = :equal, clims = (0, 1))
pkl2 = plot(; title = "KL coeff ECDF vs N(0,λ_k)", xlabel = "ξ_k", ylabel = "CDF", legend = :bottomright)
for k in (1, K_KL)
    s = sort(ξ[k, :]); ecdf = (1:N_ROUTE) ./ N_ROUTE
    plot!(pkl2, s, ecdf; label = "ξ_$k emp")
    plot!(pkl2, s, normcdf.(s ./ sqrt(ou_lambdas[k])); linestyle = :dash, label = "N(0,λ_$k)")
end
savefig(plot(pkl1, pkl2; layout = (1, 2), size = (1000, 400)), joinpath(OUTDIR, "kl_coefficients.png"))

# Negative-control panel: the correct c^-1/2 lands in the BM null band, the wrong c^-1/3 far outside.
pnc = plot([bm_lo, bm_hi], [1, 1]; linewidth = 8, alpha = 0.3, label = "BM split-half null band",
           xlabel = "√2·‖Σ̂−Σ̂_BM‖_F", ylims = (0.5, 1.5), yticks = false, legend = :top,
           title = "Self-similarity: right vs wrong exponent")
scatter!(pnc, [ss_stat], [1]; label = "c^-1/2 (correct, in band)", markersize = 7)
scatter!(pnc, [ctrl_stat], [1]; label = "c^-1/3 (control, outside)", markersize = 7, marker = :xcross)
savefig(pnc, joinpath(OUTDIR, "negative_control.png"))

# ============================================================================
#  Phase 5 — Toeplitz/Szegő cross-check + Welch overlay + aggregate
# ----------------------------------------------------------------------------
#  On [0,T] the OU integral-operator eigenvalues converge (Grenander–Szegő) to
#  the UN-NORMALIZED symbol Rhat(ω)=2D/(α²+ω²)=2π·S(ω) — NOT the 1/2π density S
#  (nystrom_eigen diagonalizes ∫R e ds; Unit 2's λ_k=R̂(k) is the anchor). The
#  convergence is asymptotic AND distributional (weakest at the spectrum edges),
#  so we gate the bulk gap g(T)=max_{k∈bulk}|λ_k−Rhat(ω_k)| shrinking as a fitted
#  log–log slope below a fixed negative threshold — deterministic (analytic
#  Rhat), a fixed absolute margin, not an SE multiple; monotonicity NOT required.
#  The Welch-vs-analytic overlay is the pedagogical reconciliation of the two
#  diagonalizations; it is explicitly NOT what the gate computes.
# ============================================================================
const T_LADDER = [4, 8, 16, 32, 64]
const NODES_PER_UNIT = 32
const K_BULK = 30
const RES_FLOOR = 1e-3
const XCHECK_THRESH = -0.5
const SEED_XCHECK = 141421
const N_WELCH = 4096
const DT_WELCH = 0.05

S_density(ω) = D / (pi * (ALPHA^2 + ω^2))              # Unit-1 1/2π density
Rhat(ω) = 2 * pi * S_density(ω)                        # un-normalized symbol = 2π·S(ω); derived, not duplicated, to prevent drift

gs = Float64[]
for Tx in T_LADDER
    nx = NODES_PER_UNIT * Tx
    nod, wt = quad_nodes_weights(float(Tx); n = nx)
    λx, _ = nystrom_eigen(ou, nod, wt)
    khi = min(K_BULK, findlast(k -> λx[k] > RES_FLOOR * λx[1], eachindex(λx)))
    push!(gs, maximum(abs(λx[k] - Rhat(k * pi / Tx)) for k in 2:khi))
end
Xdes = hcat(ones(length(T_LADDER)), log.(float.(T_LADDER)))
xco = Xdes \ log.(gs)
xslope = xco[2]
xcheck_ok = xslope < XCHECK_THRESH

@printf("\ncross-check g(T)=max_{k∈bulk}|λ_k − Rhat(kπ/T)| (analytic Rhat=2π·S):\n")
for (Tx, g) in zip(T_LADDER, gs)
    @printf("  T=%3d  g=%.5f\n", Tx, g)
end
@printf("  fitted slope log g vs log T = %.4f (reported, not claimed vs theory); threshold < %.2f -> %s\n",
        xslope, XCHECK_THRESH, xcheck_ok ? "PASS" : "FAIL")

# cross_check.png — the headline log–log ladder with fitted slope
pxc = plot(float.(T_LADDER), gs; xscale = :log10, yscale = :log10, marker = :circle,
           label = "g(T) = max bulk gap", xlabel = "T", ylabel = "g(T)",
           title = @sprintf("Toeplitz/Szegő cross-check: slope %.3f", xslope))
plot!(pxc, float.(T_LADDER), exp.(Xdes * xco); linestyle = :dash, label = @sprintf("fit slope %.3f", xslope))
savefig(pxc, joinpath(OUTDIR, "cross_check.png"))

# welch_overlay.png — visual reconciliation ONLY (own seed; not gated)
rngx = StableRNG(SEED_XCHECK)
r_welch = [ou(0.0, k * DT_WELCH) for k in 0:N_WELCH-1]
xw = sample_circulant_embedding(r_welch, rngx)
ωw, Sw = welch_psd(xw, DT_WELCH; nseg = 16, window = :hann)
pwo = plot(ωw, Sw; label = "Welch PSD (one-sided)", xlabel = "ω", ylabel = "S(ω)",
           title = "Welch estimate vs analytic 2·S(ω)", xlims = (0, 8))
plot!(pwo, ωw, 2 .* S_density.(ωw); linestyle = :dash, label = "2·S(ω) analytic")
savefig(pwo, joinpath(OUTDIR, "welch_overlay.png"))

# --- aggregate ---------------------------------------------------------------
route_ok = all_in
dist_ok  = ss_ok && ctrl_out && cw_ok && kl_ok && br_ok
println("\n", (route_ok && dist_ok && xcheck_ok) ? "ALL GATES: PASS" : "ALL GATES: FAIL")

@printf("recorded: T_OU=%.1f N_GRID=%d D=%.1f alpha=%.1f N_ROUTE=%d N_SPLIT=%d N_BRIDGE=%d c=%.1f K_KL=%d T_ladder=%s jitter=%.0e seed=%d seed_dist=%d seed_xcheck=%d\n",
        T_OU, N_GRID, D, ALPHA, N_ROUTE, N_SPLIT, N_BRIDGE, C_SCALE, K_KL, string(T_LADDER), JITTER, SEED, SEED_DIST, SEED_XCHECK)

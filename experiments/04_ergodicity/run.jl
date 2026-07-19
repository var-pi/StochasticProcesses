# ============================================================================
#  Experiment 04 — ergodicity as loop-closer
# ----------------------------------------------------------------------------
#  The methodological keystone: for a stationary process with an integrable
#  correlation (C in L^1), the time-average of ONE path converges in L^2 to the
#  mean (Pavliotis Prop. 1.16). The finite-T variance obeys Lemma 1.17,
#      Var((1/T) int_0^T X_s ds) = (2/T^2) int_0^T (T-u) C(u) du  ->  2 D* / T,
#  and the integrated process grows as E(int_0^t X)^2 -> 2 D* t, with the
#  Green-Kubo coefficient D* = int_0^inf C = D/alpha^2 (Example 1.18). This
#  licenses the one-path-vs-analytic methodology used across Units 0-3.
#
#  Two stochastic gates on ONE seeded OU ensemble (circulant embedding):
#    (a) the variance-vs-T log-log slope lands on -1 (within a multiple of its SE);
#    (b) the fitted constant lands on 2 D* -- the constant is what pins alpha^2
#        (D/alpha^2, NOT R(0)=D/alpha; they differ here since alpha != 1).
#  Plus the MSD-vs-t diagnostic (slope +1, the integrated restatement of (a)),
#  the exact Lemma-1.17 curve overlaid, and the running-average +-sqrt(2D*/T) band.
#
#  Run:  julia --project=experiments experiments/04_ergodicity/run.jl
#  Monte-Carlo, so NOT run in CI; the figures it writes are committed.
#  Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================

using StochasticProcesses
using StableRNGs, LinearAlgebra, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: GR writes PNGs with no display (CI/agent shells)
gr()

const SEED_OU  = 20260718
const D        = 1.0
const ALPHA    = 2.0                 # alpha != 1 so 2 D* = 0.5 differs from 2 R(0) = 1
const DT       = 0.05                 # Nyquist pi/DT ~ 63 >> ALPHA; resolves the OU correlation
const N_GRID   = 2^14                 # one ensemble of length-16384 records -> T_max ~ 819
const N_ENS    = 4000                 # ensemble size; per-point rel. scatter of Var ~ sqrt(2/N) ~ 2%
const TMIN_FIT = 10.0                 # fit the asymptote well past the correlation time 1/alpha = 0.5
const NLADDER  = 14                   # geometric T-ladder points (subsampled: prefixes are nested)
const TPLATEAU = 20.0                 # constant gate reads T*Var on the plateau T >= TPLATEAU
const NGROUP   = 20                   # independent sub-ensembles for the MC slope SE (see gate (a))
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

@assert N_ENS % NGROUP == 0    # else the last columns silently drop out of the group SE

# Self-contained OLS: fitted slope and its standard error (as in 00_covariance_core). The T-ladder is
# subsampled from nested prefixes of one path matrix, so successive points are autocorrelated and this
# SE is a conservative lower bound -- we fit deep in the asymptotic regime (T >= TMIN_FIT) so the
# systematic finite-T curvature is far below the noise the gate budgets.
function ols_slope_se(x::AbstractVector, y::AbstractVector)
    n = length(x); x̄ = sum(x) / n; ȳ = sum(y) / n
    Sxx = sum((x .- x̄) .^ 2)
    slope = sum((x .- x̄) .* (y .- ȳ)) / Sxx
    intercept = ȳ - slope * x̄
    resid = y .- (intercept .+ slope .* x)
    s2 = sum(abs2, resid) / (n - 2)
    return slope, sqrt(s2 / Sxx)
end

# --- (0) One seeded OU ensemble via circulant embedding -----------------------
r_seq = [exponential_kernel(0.0, k * DT; D = D, alpha = ALPHA) for k in 0:N_GRID-1]
Dstar = green_kubo(r_seq, DT)                       # ~ D/alpha^2 = 0.25 (library, not hard-coded)
@printf("Green-Kubo D* = green_kubo(r,dt) = %.6f  (analytic D/alpha^2 = %.6f)\n", Dstar, D / ALPHA^2)

rng   = StableRNG(SEED_OU)
paths = Matrix{Float64}(undef, N_GRID, N_ENS)       # n_grid x N, one path per COLUMN
for j in 1:N_ENS
    paths[:, j] = sample_circulant_embedding(r_seq, rng)
end

tav       = time_average_variance(paths, DT)        # ensemble Var(A_T) at every T
msd       = mean_square_displacement(paths, DT)     # E(int_0^t X)^2 at every t
tav_exact = time_average_variance_exact(r_seq, DT)  # exact Lemma-1.17 curve
Tgrid     = [(k - 1) * DT for k in 1:N_GRID]

# subsampled geometric asymptotic ladder
Tlad = exp.(range(log(TMIN_FIT), log(Tgrid[end] * 0.98); length = NLADDER))
idx  = unique([argmin(abs.(Tgrid .- T)) for T in Tlad])
Tk   = Tgrid[idx]
@assert length(idx) >= 3       # ols needs n-2 >= 1 dof for a slope SE

# --- (a) GATE: variance-vs-T slope -> -1 --------------------------------------
# Point estimate from the full ensemble; MC uncertainty from NGROUP DISJOINT sub-ensembles. The
# ladder points are nested prefixes of one path matrix, so ols_slope_se's residual SE is a lower
# bound (autocorrelated residuals); the spread of independent sub-ensemble slopes is the honest
# slope SE. (The ols SE is printed too, for contrast.)
slope_v, se_ols = ols_slope_se(log10.(Tk), log10.(tav[idx]))
g = div(N_ENS, NGROUP)
gslopes = [ols_slope_se(log10.(Tk),
             log10.(time_average_variance(view(paths, :, (m-1)*g+1:m*g), DT)[idx]))[1]
           for m in 1:NGROUP]
ḡ  = sum(gslopes) / NGROUP
se_v = sqrt(sum(abs2, gslopes .- ḡ) / (NGROUP - 1)) / sqrt(NGROUP)   # SE of the ensemble slope
gate_a = abs(slope_v + 1) < 2.5 * se_v
@printf("GATE (a) variance slope: %.4f;  |slope+1| = %.4f  vs  2.5*SE = %.4f  (SE_grp %.4f, SE_ols %.4f) -> %s\n",
        slope_v, abs(slope_v + 1), 2.5 * se_v, se_v, se_ols, gate_a ? "PASS" : "FAIL")

# --- (b) GATE: fitted constant -> 2 D* (pins alpha^2) -------------------------
# Slope-free: on the plateau T >= TPLATEAU, T*Var -> 2 D*. Median is robust to the tail scatter.
plateau = Tgrid .>= TPLATEAU
cvals   = sort(Tgrid[plateau] .* tav[plateau])
cmedian = cvals[div(length(cvals) + 1, 2)]
relc    = abs(cmedian - 2 * Dstar) / (2 * Dstar)
gate_b  = relc < 0.05
@printf("GATE (b) constant: median(T*Var) = %.5f  vs  2 D* = %.5f  (rel %.4f, 2R(0)=%.3f would be wrong) -> %s\n",
        cmedian, 2 * Dstar, relc, 2 * D / ALPHA, gate_b ? "PASS" : "FAIL")

# --- MSD diagnostic (NOT a gate: msd == t^2 .* tav exactly, so slope = slope_a + 2) ------
slope_m, se_m = ols_slope_se(log10.(Tk), log10.(msd[idx]))
@printf("DIAGNOSTIC MSD slope: %.4f +/- %.4f (integrated restatement of gate (a); target +1)\n", slope_m, se_m)

# --- Figures ------------------------------------------------------------------
mask = (Tgrid .>= 1.0)     # plot from ~2 correlation times onward
# (A) variance_vs_T.png: MC estimate + exact Lemma-1.17 curve + 2D*/T asymptote.
pA = plot(Tgrid[mask], tav[mask]; xscale = :log10, yscale = :log10, label = "MC Var(A_T)",
          xlabel = "T", ylabel = "Var((1/T) int_0^T X)", alpha = 0.6,
          title = @sprintf("Ergodic variance decay: slope %.3f (target -1)", slope_v))
plot!(pA, Tgrid[mask], tav_exact[mask]; label = "exact Lemma 1.17", linewidth = 2)
plot!(pA, Tgrid[mask], 2 * Dstar ./ Tgrid[mask]; linestyle = :dash, label = "asymptote 2D*/T")
scatter!(pA, Tk, tav[idx]; label = "fit ladder", markersize = 3)
savefig(pA, joinpath(OUTDIR, "variance_vs_T.png"))

# (B) msd_vs_t.png: MC MSD + 2 D* t reference.
pB = plot(Tgrid[mask], msd[mask]; xscale = :log10, yscale = :log10, label = "MC E(int X)^2",
          xlabel = "t", ylabel = "MSD  E(int_0^t X)^2", alpha = 0.6,
          title = @sprintf("Integrated MSD: slope %.3f (target +1)", slope_m))
plot!(pB, Tgrid[mask], 2 * Dstar .* Tgrid[mask]; linestyle = :dash, label = "reference 2D* t")
savefig(pB, joinpath(OUTDIR, "msd_vs_t.png"))

# (C) running_average_band.png: a few running averages inside the shrinking +-sqrt(2D*/T) band.
bmask = (Tgrid .>= 0.5) .& (Tgrid .<= 150.0)
band  = sqrt.(2 * Dstar ./ Tgrid[bmask])
pC = plot(Tgrid[bmask], band; xscale = :log10, label = "+/- sqrt(2D*/T) band",
          linestyle = :dash, color = :black, xlabel = "T", ylabel = "running time-average",
          title = "Running time-average settling into the 1-sigma band")
plot!(pC, Tgrid[bmask], -band; linestyle = :dash, color = :black, label = "")
for j in 1:6
    Aj = running_time_average(view(paths, :, j), DT)
    plot!(pC, Tgrid[bmask], Aj[bmask]; label = "", alpha = 0.7)
end
hline!(pC, [0.0]; color = :gray, linestyle = :dot, label = "")
savefig(pC, joinpath(OUTDIR, "running_average_band.png"))

# ============================================================================
#  NEGATIVE CONTROL — a non-integrable correlation breaks the ergodic 1/T rate
# ----------------------------------------------------------------------------
#  Feed the SAME time_average_variance estimator a synthetic stationary process
#  with C(u) = (1+u)^{-1/2}: positive-definite by Polya's criterion (even, convex,
#  decreasing, C(0)=1) so it is a valid covariance and Cholesky-samplable, but
#  NON-integrable (int_0^inf C = inf), so Prop. 1.16's L^1 hypothesis FAILS and the
#  time-average variance no longer decays like 1/T. Own seed; Cholesky factored ONCE
#  (the covariance is fixed) so the ensemble is L*Z on a single StableRNG draw.
#  Appended after the main block -- it cannot perturb the gate (a)/(b) numbers.
# ============================================================================
const SEED_CTRL = 13579
const N_C       = 4096                # modest grid: Cholesky is O(n^3)
const DT_C      = 0.1                 # T_max_c = N_C*DT_C ~ 410
const N_ENS_C   = 2000
const JITTER    = 1e-10               # Cholesky nugget (reported)
@assert N_ENS_C % NGROUP == 0

Cnon(u) = 1 / sqrt(1 + u)             # non-integrable, positive-definite (Polya)
r_c   = [Cnon(k * DT_C) for k in 0:N_C-1]
Σc    = [r_c[abs(i - j) + 1] for i in 1:N_C, j in 1:N_C]
Lc    = cholesky(Symmetric(Σc) + JITTER * I).L          # factor ONCE, then draw the ensemble as L*Z
paths_c = Lc * randn(StableRNG(SEED_CTRL), N_C, N_ENS_C)  # N_C x N_ENS_C, one path per COLUMN

tav_c   = time_average_variance(paths_c, DT_C)
exact_c = time_average_variance_exact(r_c, DT_C)          # exact Lemma-1.17 curve (finite even for C not in L^1)
Tg_c    = [(k - 1) * DT_C for k in 1:N_C]
Tlad_c  = exp.(range(log(20.0), log(Tg_c[end] * 0.95); length = NLADDER))
idx_c   = unique([argmin(abs.(Tg_c .- T)) for T in Tlad_c])
Tk_c    = Tg_c[idx_c]
@assert length(idx_c) >= 3

slope_c,   _ = ols_slope_se(log10.(Tk_c), log10.(tav_c[idx_c]))
slope_cex, _ = ols_slope_se(log10.(Tk_c), log10.(exact_c[idx_c]))   # exact-curve slope (deterministic)
gc = div(N_ENS_C, NGROUP)
gsl_c = [ols_slope_se(log10.(Tk_c),
           log10.(time_average_variance(view(paths_c, :, (m-1)*gc+1:m*gc), DT_C)[idx_c]))[1]
         for m in 1:NGROUP]
ḡc   = sum(gsl_c) / NGROUP
se_c = sqrt(sum(abs2, gsl_c .- ḡc) / (NGROUP - 1)) / sqrt(NGROUP)

# GATE (c): the falsifier. (c1) the MC slope tracks the EXACT finite-T Lemma-1.17 slope for this
# non-integrable C; (c2) that slope is clearly shallower than -1 (the rate an integrable C gives) --
# so Prop. 1.16's L^1 hypothesis is load-bearing. The finite-T slope sits ABOVE -1/2 (the -4/T
# correction makes the local slope ~ -1/2 + 0.75/sqrt(T)); reaching -1/2 would need T_max in the
# thousands, which the O(n^3) Cholesky control cannot afford -- so we gate the tracking, not -1/2.
gate_c1 = abs(slope_c - slope_cex) < 2.5 * se_c
gate_c2 = slope_c > -0.75
gate_c  = gate_c1 && gate_c2
@printf("CONTROL slope: MC %.4f  exact %.4f;  |MC-exact| = %.4f  vs  2.5*SE = %.4f  (c1 %s);  slope > -0.75 (c2 %s; -1 would be FALSE) -> %s\n",
        slope_c, slope_cex, abs(slope_c - slope_cex), 2.5 * se_c,
        gate_c1 ? "PASS" : "FAIL", gate_c2 ? "PASS" : "FAIL", gate_c ? "PASS" : "FAIL")

# (D) nonintegrable_control.png: MC + exact curve, with -1/2 and (falsified) -1 reference slopes.
maskc = Tg_c .>= 1.0
i0 = findfirst(maskc); T0 = Tg_c[i0]; a0 = exact_c[i0]
pD = plot(Tg_c[maskc], tav_c[maskc]; xscale = :log10, yscale = :log10, label = "MC Var(A_T)",
          xlabel = "T", ylabel = "Var((1/T) int_0^T X)", alpha = 0.6,
          title = @sprintf("Non-integrable control: slope %.3f (not -1)", slope_c))
plot!(pD, Tg_c[maskc], exact_c[maskc]; label = "exact Lemma 1.17", linewidth = 2)
plot!(pD, Tg_c[maskc], a0 .* (Tg_c[maskc] ./ T0) .^ (-0.5); linestyle = :dash, label = "slope -1/2 (asymptotic)")
plot!(pD, Tg_c[maskc], a0 .* (Tg_c[maskc] ./ T0) .^ (-1.0); linestyle = :dot,  label = "slope -1 (integrable, falsified)")
savefig(pD, joinpath(OUTDIR, "nonintegrable_control.png"))

@printf("\nrecorded: seed=%d, D=%.1f, alpha=%.1f, dt=%.3f, n_grid=%d, n_ens=%d, Dstar=%.5f | control: seed=%d, n_c=%d, dt_c=%.2f, n_ens_c=%d, jitter=%.1e\n",
        SEED_OU, D, ALPHA, DT, N_GRID, N_ENS, Dstar, SEED_CTRL, N_C, DT_C, N_ENS_C, JITTER)
println(all((gate_a, gate_b, gate_c)) ? "ALL GATES: PASS" : "ALL GATES: FAIL")

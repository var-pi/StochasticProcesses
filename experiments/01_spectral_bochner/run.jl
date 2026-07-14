# Unit 1 — spectral/Bochner: Welch & Bochner estimates of the OU Lorentzian,
# circulant-embedding synthesis, and the normalization controls.
# Run:  julia --project experiments/01_spectral_bochner/run.jl
# NOT run in CI (Monte-Carlo); the figures it writes are committed.

using StochasticProcesses
using StableRNGs, LinearAlgebra, FFTW, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: GR writes PNGs with no display (CI/agent shells)
gr()

const SEED     = 20240611
const D        = 1.0
const ALPHA    = 1.0
const DT       = 0.05                 # Nyquist omega = pi/DT ~ 62.8, well above ALPHA
const N_GRID   = 2^17                 # one exact stationary record of length 131072;
                                      # SIZED so gate (a)'s MC scatter (sd propto 1/sqrt(N_GRID)) is
                                      # ~1.7% -> the 5% gate carries ~3 sigma (M1), not ~0.7 sigma at 2^14
const NSEG     = 64                   # Welch segment count; per-bin rel SE ~ 1/sqrt(NSEG) (conservative under overlap)
const L_SEG    = div(N_GRID, NSEG)    # segment length = 2048
const NOVERLAP = L_SEG ÷ 2            # 50% Hann overlap (textbook Welch); sharpens gate (b), not gate (a)'s margin (M1/P1)
const WINDOW   = :hann
const OMEGA_MAX = 20.0                # resolved band |omega| <= OMEGA_MAX (Lorentzian decayed)
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

R0 = D / ALPHA                                   # total variance = R(0) = D/alpha
S_lorentzian(w) = D / (pi * (w^2 + ALPHA^2))     # TWO-SIDED OU spectral density (Example 1.15)
# ONE-SIDED density: the estimators are reported one-sided (folded, interior bins doubled), so at
# interior omega they approximate 2*S(omega). SHAPE comparisons use this; INTEGRAL gates use R(0).
S_lorentzian_onesided(w) = 2 * S_lorentzian(w)   # = 2D/(pi(w^2+alpha^2)), for omega > 0

# gate (a) below uses the LIBRARY spectral_power (rectangular Parseval sum dOmega*sum(Shat), Phase 1,
# CI-tested -- M2), NOT the trapezoidal spectral_variance. spectral_power counts the un-doubled DC bin
# in FULL, so it is UNBIASED at any grid (the trapezoid's DC-halving lesson lives in the Phase-1 paired
# test, not here). Gate (a)'s only residual is Monte-Carlo scatter (~1.7% at N_GRID=2^17 with 50%
# overlap), so its 5% budget carries ~3 sigma (M1) -- an unbiased integrator on a record SIZED for its
# scatter, NOT "pure headroom / exact."

# --- Biased (PSD-tapering) autocovariance estimate, for the Bochner-of-ACF route ----
function biased_acf(x, maxlag)
    N = length(x); xc = x .- sum(x) / N
    r = zeros(maxlag + 1)
    for k in 0:maxlag
        r[k+1] = sum(xc[t] * xc[t+k] for t in 1:N-k) / N   # /N (biased) => PSD, tapers to 0
    end
    return r
end

# --- (0) Synthesize ONE exact stationary OU record via circulant embedding ----------
# Single seeded draw (there is only one stochastic object in this experiment).
r_seq  = [exponential_kernel(0.0, k * DT; D = D, alpha = ALPHA) for k in 0:N_GRID-1]
record = sample_circulant_embedding(r_seq, StableRNG(SEED))

# --- (1) Two spectral estimates + analytic Lorentzian -------------------------------
omega_w, Shat_w = welch_psd(record, DT; nseg = NSEG, noverlap = NOVERLAP, window = WINDOW)
maxlag = round(Int, 12 / (ALPHA * DT))                    # capture ~12 correlation lengths
omega_b, Shat_b = bochner_forward(biased_acf(record, maxlag), DT)

# --- (2) GATE (a): total-variance normalization  int Shat dOmega / R(0) ~ 1 ----------
# Library spectral_power (rectangular Parseval, DC-robust, CI-tested), NOT spectral_variance -- see M2/M1 above.
var_ratio = spectral_power(omega_w, Shat_w) / R0
gate_a = abs(var_ratio - 1) < 0.05
@printf("GATE (a) normalization: int Shat dOmega / R0 = %.4f  -> %s\n",
        var_ratio, gate_a ? "PASS" : "FAIL")

# --- (3) GATE (b): Lorentzian shape match over the resolved band vs Welch SE ---------
mask = (omega_w .> 0) .& (omega_w .<= OMEGA_MAX)      # exclude DC (no mean removal in Welch)
# Shat_w is ONE-SIDED (folded/doubled), so compare to the ONE-SIDED density 2*S(omega), NOT S (F1).
Sa   = S_lorentzian_onesided.(omega_w[mask])
relL2 = sqrt(sum(abs2, Shat_w[mask] .- Sa)) / sqrt(sum(abs2, Sa))
se_scale = 1 / sqrt(NSEG)                                  # heuristic per-bin relative SE (see README, F9)
gate_b = relL2 < 3 * se_scale                             # "a few x the estimator's own SE"
@printf("GATE (b) shape L2: rel = %.4f  vs  3/sqrt(nseg) = %.4f  -> %s\n",
        relL2, 3 * se_scale, gate_b ? "PASS" : "FAIL")

# --- (4) GATE (c): circulant faithfulness (DETERMINISTIC, 1e-10) ---------------------
# Reconstruct the covariance from the circulant eigenvalues; compare to analytic Toeplitz.
# Two independent conditions, BOTH must hold: (1) the first column round-trips (r_recon == r,
# an FFT identity that pins the even-extension convention), and (2) every eigenvalue is
# non-negative -- the PSD precondition that makes the embedding an EXACT stationary covariance
# (a spurious negative eigenvalue is precisely what an invalid embedding produces; without this
# the gate reduces to a can't-fail ifft(fft(.)) round-trip). Same scale-relative tolerance the
# Phase-3 sampler's own guard uses.
c = vcat(r_seq, r_seq[end-1:-1:2]); lambda = real(fft(c))
r_recon = real(ifft(lambda))[1:N_GRID]
cov_err = norm(r_recon .- r_seq) / norm(r_seq)
min_lambda = minimum(lambda)
psd_ok = min_lambda >= -1e-10 * max(1, maximum(abs, lambda))   # non-negative eigenvalues
gate_c = cov_err < 1e-10 && psd_ok
@printf("GATE (c) circulant faithfulness: ||r_recon - r|| / ||r|| = %.3e, min(lambda) = %.3e  -> %s\n",
        cov_err, min_lambda, gate_c ? "PASS" : "FAIL")

# --- (5) NEGATIVE CONTROL (ii): dropped-1/2pi -> lands on 2pi, not 1 -----------------
dropped_ratio = spectral_power(omega_w, 2pi .* Shat_w) / R0   # same rectangular integrator as gate (a)
@printf("CONTROL dropped-1/2pi: int (2pi*Shat) dOmega / R0 = %.4f  (target 2pi = %.4f)\n",
        dropped_ratio, 2pi)

# --- (6) NEGATIVE CONTROL (i): raw periodogram is inconsistent (jagged, no shrink) --
omega_r, Shat_r = raw_periodogram(record, DT)
mask_r = (omega_r .> 0) .& (omega_r .<= OMEGA_MAX)
# Diagnostic: PER-BIN roughness (mean-squared successive log-difference), so raw >> Welch reflects
# genuine jaggedness, NOT raw's ~nseg-times-finer grid (F8): dividing by the bin count normalizes it.
rough(S) = sum(abs2, diff(log.(max.(S, eps())))) / max(length(S) - 1, 1)
@printf("CONTROL raw-vs-Welch per-bin roughness: raw=%.4f  welch=%.4f (raw >> welch)\n",
        rough(Shat_r[mask_r]), rough(Shat_w[mask]))

# --- Figures -----------------------------------------------------------------------
# (A) psd_vs_lorentzian.png: log-log Welch + Bochner-ACF + analytic, resolved band shaded.
p1 = plot(omega_w[mask], Shat_w[mask]; xscale=:log10, yscale=:log10, label="Welch (one-sided)",
          xlabel="omega", ylabel="S(omega)", title="OU spectral density: estimate vs Lorentzian")
mask_b = (omega_b .> 0) .& (omega_b .<= OMEGA_MAX)
plot!(p1, omega_b[mask_b], Shat_b[mask_b]; label="Bochner(ACF), one-sided")
# Dashed reference is the ONE-SIDED density 2*S(omega) (= Sa), so the estimators TRACK it rather
# than sitting a clean 2x above a two-sided curve (F1).
plot!(p1, omega_w[mask], Sa; linestyle=:dash, label="Lorentzian (one-sided) 2D/(pi(w^2+a^2))")
savefig(p1, joinpath(OUTDIR, "psd_vs_lorentzian.png"))

# (B) raw_vs_welch.png: the inconsistency made visible (raw jagged, Welch smooth).
p2 = plot(omega_r[mask_r], Shat_r[mask_r];
          xscale=:log10, yscale=:log10, label="raw (inconsistent)", alpha=0.5,
          xlabel="omega", ylabel="S(omega)", title="Raw periodogram vs Welch")
plot!(p2, omega_w[mask], Shat_w[mask]; label="Welch (consistent)", linewidth=2)
savefig(p2, joinpath(OUTDIR, "raw_vs_welch.png"))

# (C) dropped_2pi.png: two bars, int Shat/R0 (~1) vs int 2pi*Shat/R0 (~2pi).
p3 = bar(["1/2pi kept", "1/2pi dropped"], [var_ratio, dropped_ratio];
         legend=false, ylabel="int Shat dOmega / R0",
         title="The 2pi lesson: normalization pins the variance")
hline!(p3, [1.0, 2pi]; linestyle=:dash)
savefig(p3, joinpath(OUTDIR, "dropped_2pi.png"))

# (D) circulant_cov_error.png: reconstructed vs analytic covariance sequence (err ~ 1e-14).
p4 = plot(0:min(60,N_GRID-1), r_seq[1:min(61,N_GRID)]; label="analytic R(k*dt)",
          xlabel="lag k", ylabel="R", title=@sprintf("Circulant faithfulness (err %.1e)", cov_err))
plot!(p4, 0:min(60,N_GRID-1), r_recon[1:min(61,N_GRID)]; linestyle=:dash, label="reconstructed")
savefig(p4, joinpath(OUTDIR, "circulant_cov_error.png"))

@printf("\nrecorded: seed=%d, dt=%.3f, n_grid=%d, nseg=%d, noverlap=%d, window=%s, omega_max=%.1f\n",
        SEED, DT, N_GRID, NSEG, NOVERLAP, string(WINDOW), OMEGA_MAX)
println(all((gate_a, gate_b, gate_c)) ? "ALL GATES: PASS" : "ALL GATES: FAIL")

# ============================================================================
#  Experiment 01 — the spectral / Bochner picture
# ----------------------------------------------------------------------------
#  For a stationary process, the covariance R(τ) and the spectral density S(ω)
#  are a Fourier pair (Bochner's theorem). This experiment synthesizes one long,
#  exact record of the Ornstein–Uhlenbeck process and then recovers its spectral
#  density from the data, checking it against the known analytic answer — the OU
#  Lorentzian S(ω) = D / (π(ω² + α²)).
#
#  Four things are demonstrated:
#    (a) normalization  — the estimated density integrates back to the variance R(0),
#                         which pins the 1/2π convention of the Fourier pair.
#    (b) shape          — the Welch estimate tracks the analytic Lorentzian.
#    (c) faithfulness   — the circulant embedding reproduces the covariance exactly.
#    plus two negative controls: dropping the 1/2π factor moves the total power to 2π·R(0),
#    and the raw (single-shot) periodogram stays jagged where Welch smooths.
#
#  Estimator vocabulary (deeper "why" lives in this folder's README.md):
#    Welch    — average the periodogram over many (here 50%-overlapping, Hann-windowed)
#               segments; its variance shrinks with record length (a consistent estimator).
#    one-sided— the estimators report only ω >= 0 with interior bins doubled, so at interior
#               ω they approximate the one-sided density 2*S(ω). Shape checks use 2*S;
#               integral checks use R(0).
#
#  Run:  julia --project experiments/01_spectral_bochner/run.jl
#  Monte-Carlo, so NOT run in CI; the figures it writes are committed.
#  Reproducibility conventions: see ../../README.md#conventions.
# ============================================================================

using StochasticProcesses
using StableRNGs, LinearAlgebra, FFTW, Printf, Plots

ENV["GKSwstype"] = "100"   # headless: GR writes PNGs with no display (CI/agent shells)
gr()

const SEED     = 20240611
const D        = 1.0
const ALPHA    = 1.0
const DT       = 0.05                 # Nyquist frequency π/DT ≈ 62.8, well above ALPHA
const N_GRID   = 2^17                 # one exact stationary record of length 131072. Sized so the
                                      # Monte-Carlo scatter of gate (a) (std ∝ 1/sqrt(N_GRID)) is
                                      # ~1.7%, giving the 5% gate roughly 3σ of headroom — a shorter
                                      # 2^14 record would leave only ~0.7σ.
const NSEG     = 64                   # Welch segment count; per-bin relative SE ~ 1/sqrt(NSEG)
const L_SEG    = div(N_GRID, NSEG)    # segment length = 2048
const NOVERLAP = L_SEG ÷ 2            # 50% Hann overlap (textbook Welch); tightens gate (b)'s shape
                                      # estimate, without changing gate (a)'s scatter margin
const WINDOW   = :hann
const OMEGA_MAX = 20.0                # resolved band |ω| <= OMEGA_MAX (the Lorentzian has decayed by here)
const OUTDIR   = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

R0 = D / ALPHA                                   # total variance = R(0) = D/alpha
S_lorentzian(w) = D / (pi * (w^2 + ALPHA^2))     # two-sided OU spectral density (Pavliotis, Example 1.15)
# One-sided density: the estimators are reported one-sided (folded, interior bins doubled), so at
# interior ω they approximate 2*S(ω). Shape comparisons use this; integral gates use R(0).
S_lorentzian_onesided(w) = 2 * S_lorentzian(w)   # = 2D/(pi(w^2+alpha^2)), for omega > 0

# Gate (a) below integrates with the library's spectral_power (the rectangular Parseval sum
# dOmega*sum(Shat)), NOT the trapezoidal spectral_variance. spectral_power counts the un-doubled DC
# bin at full weight, so it is unbiased at any grid resolution. Its only residual is Monte-Carlo
# scatter (~1.7% at this record length), so the 5% budget is genuine ~3σ headroom on an unbiased
# integrator — not an artifact of an exact/analytic input.

# --- Biased (PSD-tapering) autocovariance estimate, for the Bochner-of-ACF route ----
# Dividing by N rather than N-k tapers the estimate toward zero at large lag, which keeps the
# resulting spectrum non-negative — the property we want before Fourier-transforming it.
function biased_acf(x, maxlag)
    N = length(x); xc = x .- sum(x) / N
    r = zeros(maxlag + 1)
    for k in 0:maxlag
        r[k+1] = sum(xc[t] * xc[t+k] for t in 1:N-k) / N   # /N (biased) so it tapers to 0
    end
    return r
end

# --- (0) Synthesize ONE exact stationary OU record via circulant embedding ----------
# A single seeded draw — there is only one stochastic object in this whole experiment.
r_seq  = [exponential_kernel(0.0, k * DT; D = D, alpha = ALPHA) for k in 0:N_GRID-1]
record = sample_circulant_embedding(r_seq, StableRNG(SEED))

# --- (1) Two spectral estimates + analytic Lorentzian -------------------------------
omega_w, Shat_w = welch_psd(record, DT; nseg = NSEG, noverlap = NOVERLAP, window = WINDOW)
maxlag = round(Int, 12 / (ALPHA * DT))                    # capture ~12 correlation lengths
omega_b, Shat_b = bochner_forward(biased_acf(record, maxlag), DT)

# --- (2) GATE (a): total-variance normalization  ∫ Ŝ dω / R(0) ≈ 1 -------------------
# Integrated with spectral_power (rectangular Parseval, DC-robust) — see the note above.
var_ratio = spectral_power(omega_w, Shat_w) / R0
gate_a = abs(var_ratio - 1) < 0.05
@printf("GATE (a) normalization: int Shat dOmega / R0 = %.4f  -> %s\n",
        var_ratio, gate_a ? "PASS" : "FAIL")

# --- (3) GATE (b): Lorentzian shape match over the resolved band vs Welch's own SE ---
mask = (omega_w .> 0) .& (omega_w .<= OMEGA_MAX)      # exclude DC (Welch does no mean removal)
# Shat_w is one-sided (folded/doubled), so compare it to the one-sided density 2*S(ω), not S.
Sa   = S_lorentzian_onesided.(omega_w[mask])
relL2 = sqrt(sum(abs2, Shat_w[mask] .- Sa)) / sqrt(sum(abs2, Sa))
se_scale = 1 / sqrt(NSEG)                                  # rough per-bin relative SE of Welch (see README)
gate_b = relL2 < 3 * se_scale                             # pass if within a few times that SE
@printf("GATE (b) shape L2: rel = %.4f  vs  3/sqrt(nseg) = %.4f  -> %s\n",
        relL2, 3 * se_scale, gate_b ? "PASS" : "FAIL")

# --- (4) GATE (c): circulant faithfulness (DETERMINISTIC, 1e-10) ---------------------
# Two independent conditions, both required:
#   (1) the covariance round-trips exactly (r_recon == r) — an FFT identity that pins the
#       even-extension convention; and
#   (2) every circulant eigenvalue is non-negative — the PSD precondition that makes the
#       embedding an exact stationary covariance. A spurious negative eigenvalue is precisely
#       what an invalid embedding produces; without condition (2) the gate would degrade to a
#       can't-fail ifft(fft(·)) round-trip. Same scale-relative tolerance the sampler's own guard uses.
c = vcat(r_seq, r_seq[end-1:-1:2]); lambda = real(fft(c))
r_recon = real(ifft(lambda))[1:N_GRID]
cov_err = norm(r_recon .- r_seq) / norm(r_seq)
min_lambda = minimum(lambda)
psd_ok = min_lambda >= -1e-10 * max(1, maximum(abs, lambda))   # eigenvalues non-negative
gate_c = cov_err < 1e-10 && psd_ok
@printf("GATE (c) circulant faithfulness: ||r_recon - r|| / ||r|| = %.3e, min(lambda) = %.3e  -> %s\n",
        cov_err, min_lambda, gate_c ? "PASS" : "FAIL")

# --- (5) NEGATIVE CONTROL: dropping the 1/2π moves the power to 2π, not 1 -------------
dropped_ratio = spectral_power(omega_w, 2pi .* Shat_w) / R0   # same rectangular integrator as gate (a)
@printf("CONTROL dropped-1/2pi: int (2pi*Shat) dOmega / R0 = %.4f  (target 2pi = %.4f)\n",
        dropped_ratio, 2pi)

# --- (6) NEGATIVE CONTROL: the raw periodogram is inconsistent (jagged, no shrink) ---
omega_r, Shat_r = raw_periodogram(record, DT)
mask_r = (omega_r .> 0) .& (omega_r .<= OMEGA_MAX)
# Diagnostic: per-bin roughness (mean-squared successive log-difference). Dividing by the bin
# count normalizes it, so raw >> Welch reflects genuine jaggedness rather than raw's finer grid.
rough(S) = sum(abs2, diff(log.(max.(S, eps())))) / max(length(S) - 1, 1)
@printf("CONTROL raw-vs-Welch per-bin roughness: raw=%.4f  welch=%.4f (raw >> welch)\n",
        rough(Shat_r[mask_r]), rough(Shat_w[mask]))

# --- Figures -----------------------------------------------------------------------
# (A) psd_vs_lorentzian.png: log-log Welch + Bochner-of-ACF + analytic, over the resolved band.
p1 = plot(omega_w[mask], Shat_w[mask]; xscale=:log10, yscale=:log10, label="Welch (one-sided)",
          xlabel="omega", ylabel="S(omega)", title="OU spectral density: estimate vs Lorentzian")
mask_b = (omega_b .> 0) .& (omega_b .<= OMEGA_MAX)
plot!(p1, omega_b[mask_b], Shat_b[mask_b]; label="Bochner(ACF), one-sided")
# The dashed reference is the one-sided density 2*S(ω) (= Sa), so the estimators track it directly
# rather than sitting a clean factor of 2 above a two-sided curve.
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

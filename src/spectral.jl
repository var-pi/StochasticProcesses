# ============================================================================
#  Spectral — the Fourier side of a stationary process
# ----------------------------------------------------------------------------
#  Bochner's theorem says a stationary covariance R(tau) and a spectral density
#  S(omega) are a Fourier-transform pair: every mode omega carries a slice of the
#  process's variance, and R and S are two views of the same object.
#
#      S(omega) = (1/2pi) * integral e^{-i omega tau} R(tau) dtau      (Pavliotis eq. 1.7)
#      R(tau)   =           integral e^{ i omega tau} S(omega) domega   (eq. 1.8)
#
#  This module has two kinds of routine:
#    - the exact transform of a *known* covariance sequence (bochner_forward / _inverse);
#    - estimators of S from a *sampled record* (welch_psd, raw_periodogram), plus two
#      ways to integrate a spectrum back to a variance (spectral_variance / spectral_power).
#
#  Conventions and jargon used below (the deeper "why" of these lives in
#  experiments/01_spectral_bochner/README.md; here we only gloss them):
#
#    frequency convention — angular frequency omega = 2*pi*f, and the 1/2pi factor of the
#                           pair sits on S (eq. 1.7). Get this wrong and integrals land a
#                           clean factor of 2pi off (see spectral_variance's docstring).
#    DC bin               — the omega = 0 component (the mean-square / zero-frequency term).
#    Nyquist bin          — the highest frequency an even-length grid can represent, at
#                           omega = pi/dt; a fold puts it at the edge, where it is dropped.
#    one-sided fold       — real signals have S(-omega) = S(omega), so we report only
#                           omega >= 0 and double the interior bins to keep the total power.
#                           At interior omega a one-sided value approximates 2*S(omega).
#    Parseval             — total power computed by summing the spectrum equals the
#                           mean-square in the time domain; the basis of the variance checks.
#    Welch vs raw         — Welch averages the spectrum over many segments, so its variance
#                           shrinks with record length (a *consistent* estimator); the raw
#                           single-shot periodogram does not (kept only as a counter-example).
# ============================================================================
module Spectral

using FFTW
export bochner_forward, spectral_variance, spectral_power, welch_psd, raw_periodogram

# --- private helpers ---------------------------------------------------------

# Fold a two-sided (omega, S) pair onto omega >= 0, doubling every interior bin so the
# one-sided integral over omega >= 0 still equals the two-sided integral over the whole
# line. The DC bin (omega = 0) is not doubled — it has no negative-frequency twin.
#
# Caveat for even-length input: fftfreq places the Nyquist frequency at -m/2, so the
# `omega >= 0` mask DROPS the Nyquist bin rather than folding it. That is negligible for a
# band-limited signal like the OU process, and it is part of why the normalization check
# lands near 0.975*R(0) rather than exactly R(0).
function _onesided(omega, S)
    keep = omega .>= 0
    o = omega[keep]; s = S[keep]                 # boolean indexing already returns a copy
    s[2:end] .*= 2                               # double everything except the DC bin
    return o, s
end

# Sort a two-sided (omega, S) pair into ascending omega. Shared by bochner_forward and
# welch_psd so both estimators use the identical reordering convention. We sort explicitly
# (sortperm) rather than exploiting fftfreq's known layout, because bochner_forward's even
# extension has length 2*length(r)-2, which is odd when length(r) == 1 — a degenerate but
# non-crashing input where the "already ascending on the positive side" shortcut fails.
function _sorted_by_omega(omega, S)
    p = sortperm(omega)
    return omega[p], S[p]
end

# The raw, unsorted, two-sided transform, before any folding or sorting. This is the one
# place the even-extension + FFT + dt/(2pi) scaling is written down, so bochner_forward and
# the round-trip test (which reaches it by its full module path) cannot drift apart.
function _raw_transform(r, dt)
    rsym = vcat(r, r[end-1:-1:2])                 # even extension of the covariance sequence
    S = real(fft(rsym)) .* (dt / (2pi))           # discrete eq. 1.7, with the 1/2pi on S
    omega = 2pi .* fftfreq(length(rsym), 1 / dt)  # angular frequencies of the FFT bins
    return rsym, S, omega
end

# --- public transform --------------------------------------------------------

"""
    bochner_forward(r, dt; onesided = true) -> (omega, S)

Discrete Bochner forward transform (eq. 1.7): map a sampled covariance sequence
r = [R(0), R(dt), R(2dt), ...] to a spectral density S(omega).

Arguments:
  - `r`        : the covariance sequence at lags 0, dt, 2dt, ...
  - `dt`       : the lag spacing.
  - `onesided` : if true (default), fold onto omega >= 0 with interior bins doubled, so at
                 interior omega the value approximates the one-sided density 2*S(omega).
                 Pass false for the raw two-sided pair.

Returns `(omega, S)`: the frequency grid (ascending) and the density at those frequencies.
"""
function bochner_forward(r, dt; onesided = true)
    isempty(r) && throw(ArgumentError("bochner_forward needs a non-empty covariance sequence r"))
    rsym, S, omega = _raw_transform(r, dt)
    omega, S = _sorted_by_omega(omega, S)
    return onesided ? _onesided(omega, S) : (omega, S)
end

"""
    bochner_inverse(S, dOmega) -> r        (private, not exported)

Discrete Bochner inverse (eq. 1.8): spectral density back to covariance.

It expects the *natural fft-ordered, two-sided* S — NOT the sorted / one-sided output of
`bochner_forward`, which it cannot invert (that is exactly why bochner_forward's result is
reshaped and this stays private). Used by the round-trip test and future internal callers.
`dOmega` is the frequency spacing of S.
"""
function bochner_inverse(S, dOmega)
    return real(ifft(S)) .* (length(S) * dOmega)
end

# --- integrating a spectrum back to a variance -------------------------------

"""
    spectral_variance(omega, Shat) -> Float64

Total variance by the TRAPEZOIDAL rule: integral of the one-sided Shat over omega, which
should recover R(0). Use this for smooth analytic densities.

Accuracy note: the trapezoid rule weights the endpoints (including the un-doubled DC bin)
by 1/2, so on a coarse grid it *under*-integrates a spectrum whose weight sits near DC. For
the exact total power of a discrete periodogram, prefer `spectral_power` instead.

Debugging hint: a result of 2*pi*R(0) instead of R(0) is the tell-tale sign that the 1/2pi
factor was dropped somewhere in the transform.
"""
function spectral_variance(omega, Shat)
    length(omega) >= 2 || throw(ArgumentError(
        "spectral_variance needs at least 2 grid points (got $(length(omega))); " *
        "the trapezoidal integral is undefined for a single point."))
    return sum(0.5 .* (Shat[1:end-1] .+ Shat[2:end]) .* diff(omega))
end

"""
    spectral_power(omega, Shat) -> Float64

Total power by the RECTANGULAR (Parseval) rule: dOmega * sum(Shat). Assumes a uniform omega
grid (spacing omega[2] - omega[1]).

Unlike `spectral_variance`, this counts the DC bin at full weight, so it is unbiased (exact
in expectation) for a one-sided periodogram at any grid resolution. Use it for periodogram /
Parseval total-power checks; use `spectral_variance` for smooth analytic densities. The two
differ most on coarse grids or DC-heavy spectra, precisely because of that DC weighting.
"""
function spectral_power(omega, Shat)
    length(omega) >= 2 || throw(ArgumentError(
        "spectral_power needs at least 2 grid points (got $(length(omega))); " *
        "the grid spacing omega[2]-omega[1] is undefined for a single point."))
    return (omega[2] - omega[1]) * sum(Shat)
end

# --- estimating S from a sampled record --------------------------------------

"""
    welch_psd(x, dt; nseg, noverlap = 0, window = :hann, onesided = true) -> (omega, S)

Welch's estimate of the spectral density of a real record `x` sampled at spacing `dt`.

Welch splits the record into (optionally overlapping, windowed) segments and averages their
periodograms; more segments means less variance, so the estimate is *consistent* — it tightens
as the record grows.

Arguments:
  - `x`        : the sampled real signal.
  - `dt`       : the sampling interval.
  - `nseg`     : number of segments; the segment length is L = length(x) ÷ nseg.
  - `noverlap` : samples of overlap between consecutive segments (0 = no overlap).
  - `window`   : `:hann` (a smooth taper that reduces spectral leakage) or `:none`.
  - `onesided` : fold onto omega >= 0 with interior bins doubled (see the module preamble);
                 like bochner_forward, an even segment length L drops the Nyquist bin.

Returns `(omega, S)`, normalized so the discrete integral of S recovers R(0) under the
1/2pi-on-S convention (eq. 1.7–1.8). Two details of that normalization: frequencies are
angular (omega = 2*pi*f), and each periodogram is divided by the window power
U = sum(win.^2) — NOT the segment length L. (With overlap, the number of segments actually
averaged, `nused`, can exceed `nseg`.)
"""
function welch_psd(x, dt; nseg, noverlap = 0, window = :hann, onesided = true)
    nseg >= 1 || throw(ArgumentError("welch_psd needs nseg >= 1 (got $nseg)"))
    N = length(x); L = div(N, nseg)               # segment length
    L >= 2 || throw(ArgumentError(
        "welch_psd needs a segment length >= 2 (got L = $L from length(x) = $N, " *
        "nseg = $nseg); use fewer segments or a longer record."))
    0 <= noverlap < L || throw(ArgumentError(
        "welch_psd needs 0 <= noverlap < L (got noverlap = $noverlap, L = $L); " *
        "noverlap >= L makes the segment hop non-positive and never terminates."))
    win = if window === :hann
        [0.5 - 0.5 * cos(2pi * k / (L - 1)) for k in 0:L-1]
    elseif window === :none
        ones(L)
    else
        throw(ArgumentError("welch_psd: unknown window $(repr(window)); use :hann or :none"))
    end
    U = sum(abs2, win)                            # window power, the correct normalizer
    hop = L - noverlap                            # step between successive segment starts
    acc = zeros(L)                                # running sum of segment periodograms
    nused = 0                                     # number of segments actually averaged
    start = 1
    while start + L - 1 <= N
        seg = @view x[start:start+L-1]
        acc .+= abs2.(fft(win .* seg))            # |FFT of the windowed segment|^2
        nused += 1; start += hop
    end
    Sfull = (dt / (2pi * U)) .* (acc ./ nused)    # average, then the 1/2pi-on-S normalization
    omega_full = 2pi .* fftfreq(L, 1 / dt)        # angular frequencies of the segment FFT
    omega_full, Sfull = _sorted_by_omega(omega_full, Sfull)
    return onesided ? _onesided(omega_full, Sfull) : (omega_full, Sfull)
end

"""
    raw_periodogram(x, dt; onesided = true) -> (omega, S)

The single-shot periodogram: `welch_psd` with one segment and no window. It is the
*inconsistent* estimator — its variance does not shrink as the record grows — kept only as a
counter-example to Welch. It shares Welch's normalization so the two overlay honestly.
"""
raw_periodogram(x, dt; onesided = true) =
    welch_psd(x, dt; nseg = 1, window = :none, onesided = onesided)

end # module

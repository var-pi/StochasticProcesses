module Spectral

using FFTW
export bochner_forward, spectral_variance, spectral_power

# Fold a two-sided (omega, S) pair onto omega >= 0, doubling interior bins so
# that the one-sided integral still returns int S dOmega over the full line.
# NOTE: for even-length input fftfreq places Nyquist at -m/2, so keep = (omega >= 0)
# DROPS the Nyquist bin (it is not folded). Negligible for band-limited OU; it is
# part of why the normalization gate lands at ~0.975*R(0), not exactly 1.
function _onesided(omega, S)
    keep = omega .>= 0
    o = omega[keep]; s = copy(S[keep])
    s[2:end] .*= 2                               # double all but the DC bin
    return o, s
end

"Bochner forward transform of a covariance sequence r = [R(0), R(dt), ...]
 into S(omega): the discrete realization of
 S(omega) = (1/2pi) int e^{-i omega t} R(t) dt   (1.7).
 Returns a ONE-SIDED (omega>=0, interior bins doubled) spectrum by default, so at
 interior omega the value approximates the ONE-SIDED density 2*S(omega); pass
 onesided=false for the raw two-sided pair."
function bochner_forward(r, dt; onesided = true)
    rsym = vcat(r, r[end-1:-1:2])                 # even extension
    S = real(fft(rsym)) .* (dt / (2pi))
    omega = 2pi .* fftfreq(length(rsym), 1 / dt)
    p = sortperm(omega); omega, S = omega[p], S[p]
    return onesided ? _onesided(omega, S) : (omega, S)
end

"Bochner inverse: S(omega) -> covariance, the discrete form of
 R(t) = int e^{i omega t} S(omega) dOmega   (1.8).
 CONTRACT (private, not exported): expects the natural fft-ordered, TWO-SIDED S
 -- NOT the sorted/one-sided output of bochner_forward (which it CANNOT invert).
 Used only by the round-trip test and future internal callers."
function bochner_inverse(S, dOmega)
    return real(ifft(S)) .* (length(S) * dOmega)
end

"Total-variance check: TRAPEZOIDAL one-sided int Shat dOmega, which must approach R(0).
 ACCURACY DOMAIN: the un-doubled DC bin is weighted 1/2 by the trapezoid rule, so this
 UNDER-integrates DC-dominated / coarse-grid spectra (the paired CI test exhibits a 25%
 DC gap on a coarse 4-point grid: trapezoid 7.5 vs rectangular 10). For a discrete
 periodogram's EXACT total power use spectral_power (rectangular Parseval) instead (M2/N1).
 A result of 2*pi*R(0) instead of R(0) is the signature of a dropped 1/2pi."
spectral_variance(omega, Shat) =
    sum(0.5 .* (Shat[1:end-1] .+ Shat[2:end]) .* diff(omega))

"Total power of a DISCRETE spectrum: the RECTANGULAR Parseval sum dOmega*sum(Shat).
 DC-ROBUST -- counts the un-doubled DC bin in FULL, so it is UNBIASED (exact in
 expectation) for a one-sided periodogram at ANY grid resolution, unlike
 spectral_variance's trapezoid (which halves DC -> under-integrates DC-dominated /
 coarse-grid spectra; the paired CI test shows a 25% DC gap on a 4-point grid).
 Use THIS for periodogram total power / Parseval checks; use spectral_variance for
 smooth analytic densities. Assumes a UNIFORM omega grid (omega[2]-omega[1] step)."
spectral_power(omega, Shat) = (omega[2] - omega[1]) * sum(Shat)

end # module

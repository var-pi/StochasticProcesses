# ============================================================================
#  Kernels — the covariance functions that define each process
# ----------------------------------------------------------------------------
#  A Gaussian process is fixed by its mean and its covariance kernel R(t, s) =
#  Cov(X(t), X(s)). So "which process" is really just "which kernel". This file
#  collects the kernels the rest of the library builds on.
#
#  Two flavours show up:
#    - non-stationary: R depends on t and s separately (Brownian motion below).
#    - stationary:     R depends only on the lag tau = t - s, so R(t,s) = R(|tau|)
#                      (the exponential/OU kernel below).
# ============================================================================
module Kernels

export brownian_motion_kernel, exponential_kernel, periodic_kernel, brownian_bridge_kernel

"""
    brownian_motion_kernel(t, s) -> Float64

Covariance of standard Brownian motion on [0, T]:  R(t, s) = min(t, s).

`t`, `s` are two times. This kernel is non-stationary (it depends on t and s, not
just their difference) and vanishes whenever either time is 0 — the process starts
pinned at the origin, X(0) = 0.
"""
brownian_motion_kernel(t, s) = min(t, s)

"""
    exponential_kernel(t, s; D=1.0, alpha=1.0) -> Float64

Stationary exponential covariance  R(tau) = (D/alpha) * exp(-alpha * |tau|),  tau = t - s.

This is the covariance of the stationary Ornstein–Uhlenbeck process (Pavliotis,
Example 1.15) — the canonical model of a variable relaxing toward equilibrium while
buffeted by white noise.

Arguments:
  - `t`, `s`  : two times; only their difference |t - s| matters (stationarity).
  - `D`       : noise strength. It sets the variance at zero lag, R(0) = D/alpha.
  - `alpha`   : relaxation rate; correlations decay like exp(-alpha*|tau|), so 1/alpha
                is the correlation time.

A note on two constants that are easy to confuse (they coincide only when alpha = 1):
  - R(0) = D/alpha           is the variance.
  - D* = integral_0^inf R    = D/alpha^2  is the effective diffusion / transport
                                coefficient (the "Green–Kubo" coefficient — a transport
                                rate obtained by integrating the correlation over all
                                lag). Pavliotis Example 1.18 computes exactly this.
"""
exponential_kernel(t, s; D=1.0, alpha=1.0) = (D / alpha) * exp(-alpha * abs(t - s))

"""
    periodic_kernel(t, s; D=1.0, alpha=1.0) -> Float64

Stationary covariance on the unit circle (torus of circumference 1): the exponential kernel
wrapped to the *periodic distance*,

    R(t, s) = D * exp(-alpha * d(t, s)),   d(t, s) = min(delta, 1 - delta),  delta = frac(|t - s|).

d(t,s) is the shorter of the two arc distances around the circle, so R is periodic with period 1 in
each argument and depends only on that arc distance.

Unlike `exponential_kernel` (defined on the whole line), this kernel lives on a *group* — the
circle — so its covariance operator is a genuine convolution and is diagonalized *exactly* by the
Fourier characters. Its Fourier coefficients

    Rhat(k) = 2*alpha*D*(1 - (-1)^k * exp(-alpha/2)) / (alpha^2 + (2*pi*k)^2)   (> 0 for all k)

are all strictly positive — a full spectrum, unlike Pavliotis's rank-2 Exercise-27 kernel
cos 2pi(t-s) — so R is positive-definite on the torus. These Rhat(k) are exactly the
Karhunen-Loeve eigenvalues on the circle: the "torus coincidence" lambda_k = Rhat(k) that Unit 2
checks (and shows failing on a bounded interval, where the eigenfunctions become Sturm-Liouville
sines rather than characters).

Arguments:
  - `t`, `s`  : two points; only the periodic distance d(t,s) matters (stationary on the circle).
  - `D`       : sets the variance at zero lag, R(0) = D.
  - `alpha`   : decay rate of correlation with arc distance.

Note the prefactor differs from `exponential_kernel`: here R(0) = D (not D/alpha). Inside a
half-period (|t-s| <= 1/2) the periodic distance is just |t-s|, so there R = D*exp(-alpha|t-s|).
"""
function periodic_kernel(t, s; D=1.0, alpha=1.0)
    delta = abs(t - s)
    delta -= floor(delta)                 # fractional part -> [0, 1)
    d = min(delta, 1 - delta)             # shorter arc distance on the circle
    return D * exp(-alpha * d)
end

"""
    brownian_bridge_kernel(t, s) -> Float64

Covariance of the standard Brownian bridge on [0, 1]:  R(t, s) = min(t, s) - t*s.

The bridge is Brownian motion conditioned to return to 0 at time 1 (Pavliotis §1.5).
Subtracting `t*s` from the Brownian-motion kernel is exactly what pins the second
endpoint: it leaves the process non-stationary, but now zero at *both* ends of the
interval, not just at t = 0.

`t`, `s` are two times restricted to [0, 1] (the domain is fixed, unlike
`exponential_kernel`/`periodic_kernel`, which take a rate parameter — the bridge has
no free constant to tune). Key values worth having in mind:
  - R(t, t)  = t*(1 - t)          -- the variance profile, zero at both ends, maximal at t=1/2.
  - R(0, s)  = 0                  -- pinned at the start, same as Brownian motion.
  - R(1, s)  = min(1, s) - s = 0  -- pinned at the end too (min(1,s) = s for s in [0,1]).
A kernel that only implements `min(t,s)` and forgets the `-t*s` term would still pass
the t=0 pin but silently fail the t=1 one -- both endpoints must be checked to catch it.
"""
brownian_bridge_kernel(t, s) = min(t, s) - t * s

end # module

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

export brownian_motion_kernel, exponential_kernel

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

end # module

module Kernels

export brownian_motion_kernel, exponential_kernel

"Covariance R(t,s) = min(t,s): standard Brownian motion on [0,T]."
brownian_motion_kernel(t, s) = min(t, s)

"Stationary exponential kernel R(tau) = (D/alpha) * exp(-alpha*|tau|);
 corresponds to the stationary Ornstein-Uhlenbeck process (Example 1.15).
 Note: D is the noise strength (C(0)=D/alpha); the Green-Kubo transport
 coefficient is Dstar = int_0^inf C = D/alpha^2, distinct unless alpha=1.
 (Example 1.18 computes exactly this: int_0^inf C = C(0)*tau_cor = D/alpha^2.)"
exponential_kernel(t, s; D=1.0, alpha=1.0) = (D / alpha) * exp(-alpha * abs(t - s))

end # module

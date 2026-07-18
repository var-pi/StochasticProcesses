"""
    StochasticProcesses

A computational companion to Pavliotis, *Stochastic Processes and Applications*, Ch. 1.

The library is organized **by operation on the covariance operator**, not by named
process — because a stochastic process here is nothing more than a choice of covariance
kernel `R(t, s)`. The pieces build up in that spirit:

  - `Kernels`          — the kernels themselves (Brownian motion, Ornstein–Uhlenbeck).
  - `GaussianProcesses` — turn a kernel into a covariance matrix on a grid, and estimate
                          that matrix back from sample paths.
  - `Sampling`         — draw sample paths (two square-root methods: Cholesky and FFT/circulant).
  - `Spectral`         — the Fourier side: covariance <-> spectral density, and estimators of it.
  - `KL`               — the Mercer/Karhunen–Loève eigenbasis: the second diagonalization.
  - `GOF`              — the one deliberate exception: goodness-of-fit statistics (shared
                          across units, not an operation on the covariance operator).
  - `Ergodic`          — the ergodic loop: time-average estimators and the Green–Kubo coefficient.

Each `include` below pulls in one submodule and re-exports its public names, so `using
StochasticProcesses` gives a flat namespace.
"""
module StochasticProcesses

include("kernels.jl")
using .Kernels
export brownian_motion_kernel, exponential_kernel, periodic_kernel, brownian_bridge_kernel

include("gaussianprocess.jl")
using .GaussianProcesses
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

include("sampling.jl")
using .Sampling
export sample_cholesky, sample_circulant_embedding, sample_kl

include("spectral.jl")
using .Spectral
export bochner_forward, spectral_variance, spectral_power, welch_psd, raw_periodogram

include("kl.jl")
using .KL
export quad_nodes_weights, nystrom_eigen, trace_diag, kl_tail_energy

include("gof.jl")
using .GOF
export ks_statistic

include("ergodic.jl")
using .Ergodic
export running_time_average, time_average_variance, mean_square_displacement,
       green_kubo, time_average_variance_exact

end # module

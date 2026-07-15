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

Each `include` below pulls in one submodule and re-exports its public names, so `using
StochasticProcesses` gives a flat namespace.
"""
module StochasticProcesses

include("kernels.jl")
using .Kernels
export brownian_motion_kernel, exponential_kernel, periodic_kernel

include("gaussianprocess.jl")
using .GaussianProcesses
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

include("sampling.jl")
using .Sampling
export sample_cholesky, sample_circulant_embedding

include("spectral.jl")
using .Spectral
export bochner_forward, spectral_variance, spectral_power, welch_psd, raw_periodogram

include("kl.jl")
using .KL
export quad_nodes_weights, nystrom_eigen, trace_diag, kl_tail_energy

end # module

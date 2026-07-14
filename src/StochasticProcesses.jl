module StochasticProcesses

# Submodules are added one per phase (Kernels, GaussianProcesses, Sampling, Spectral).

include("kernels.jl")
using .Kernels
export brownian_motion_kernel, exponential_kernel

include("gaussianprocess.jl")
using .GaussianProcesses
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

include("sampling.jl")
using .Sampling
export sample_cholesky

include("spectral.jl")
using .Spectral
export bochner_forward, spectral_variance, spectral_power, welch_psd, raw_periodogram

end # module

module StochasticProcesses

# Submodules are added one per phase (Kernels, GaussianProcesses, Sampling).

include("kernels.jl")
using .Kernels
export brownian_motion_kernel, exponential_kernel

include("gaussianprocess.jl")
using .GaussianProcesses
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

end # module

# ============================================================================
#  GaussianProcesses — from a kernel to a covariance matrix, and back
# ----------------------------------------------------------------------------
#  A Gaussian process is completely determined by its mean function m(t) and its
#  covariance kernel R(t, s) (Pavliotis, Appendix B.5). This module does the two
#  bookkeeping steps that turn that abstract object into arrays we can compute with:
#
#    forward  (assemble_cov / assemble_mean):  evaluate R and m on a finite grid of
#             times to get the covariance matrix Sigma and mean vector mu.
#    backward (empirical_cov):                 given many sample paths, estimate Sigma
#             back from the data — the sanity check that the sampler is correct.
# ============================================================================
module GaussianProcesses

using LinearAlgebra
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

"""
    GaussianProcess(kernel; meanfn = t -> 0.0)

A second-order Gaussian process, stored as its two defining ingredients: a covariance
kernel `R(t, s)` and a mean function `m(t)` (default: the zero mean). Together they
determine the whole law of the process.
"""
struct GaussianProcess{K,M}
    kernel::K          # the covariance kernel R(t, s)
    meanfn::M          # the mean function m(t)
end
GaussianProcess(kernel; meanfn = t -> 0.0) = GaussianProcess(kernel, meanfn)

"""
    assemble_cov(gp, t_grid) -> Symmetric

Evaluate the kernel on every pair of grid times to build the covariance matrix
Sigma[i, j] = R(t_i, t_j).

`t_grid` is a length-n vector of times; the result is an n×n matrix.

It is wrapped in `Symmetric` on purpose: floating-point evaluation of R(t_i, t_j) and
R(t_j, t_i) need not produce bit-identical numbers, so a plain matrix would be only
approximately symmetric. Cholesky and eigen-solvers require an exactly symmetric
operand, and the wrapper guarantees that without a second evaluation.
"""
function assemble_cov(gp::GaussianProcess, t_grid::AbstractVector)
    R = gp.kernel
    Sigma = [R(t, s) for t in t_grid, s in t_grid]
    return Symmetric(Sigma)
end

"""
    assemble_mean(gp, t_grid) -> Vector

Evaluate the mean function on the grid: the length-n vector m(t_i).
"""
assemble_mean(gp::GaussianProcess, t_grid::AbstractVector) =
    [gp.meanfn(t) for t in t_grid]

"""
    empirical_cov(paths) -> Matrix

Estimate the covariance matrix from a collection of sample paths.

`paths` is an `n_grid × N` matrix holding **one sample path per column** (N paths, each
observed at the same n_grid times); the result is `n_grid × n_grid`. The column
convention is load-bearing — passing the transpose silently estimates the wrong matrix.
Needs N ≥ 2 (the sample-covariance formula divides by N - 1).

This is the headline consistency check for the samplers: as N grows, the estimate
converges to the assembled Sigma, and the error measured in the Frobenius norm (the
entrywise root-sum-of-squares matrix norm) shrinks at the Monte-Carlo rate ∝ N^(-1/2).
"""
function empirical_cov(paths::AbstractMatrix)
    _, N = size(paths)
    N > 1 || throw(ArgumentError("empirical_cov needs N ≥ 2 paths (got N=$N); the (N-1) " *
                                 "denominator is undefined for a single path."))
    mu = sum(paths, dims = 2) ./ N        # sample mean at each grid time (a column vector)
    Xc = paths .- mu                       # center every path by that mean
    return (Xc * Xc') ./ (N - 1)           # unbiased sample covariance
end

end # module

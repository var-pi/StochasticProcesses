module GaussianProcesses

using LinearAlgebra
export GaussianProcess, assemble_cov, assemble_mean, empirical_cov

"A second-order Gaussian process over a covariance kernel R(t,s) and an
 optional mean function. A process is nothing more than a choice of kernel;
 mean and covariance determine the law (Appendix B.5)."
struct GaussianProcess{K,M}
    kernel::K          # R(t, s)
    meanfn::M          # m(t)
end
GaussianProcess(kernel; meanfn = t -> 0.0) = GaussianProcess(kernel, meanfn)

"Assemble Sigma[i,j] = R(t_i, t_j) on a grid. Returned Symmetric so the
 downstream Cholesky/eigen see an exactly symmetric operand (floating-point
 assembly is not bitwise symmetric otherwise)."
function assemble_cov(gp::GaussianProcess, t_grid::AbstractVector)
    R = gp.kernel
    Sigma = [R(t, s) for t in t_grid, s in t_grid]
    return Symmetric(Sigma)
end

"Assemble the mean vector m(t_i) on a grid."
assemble_mean(gp::GaussianProcess, t_grid::AbstractVector) =
    [gp.meanfn(t) for t in t_grid]

"Empirical covariance across sample paths. `paths` is n_grid x N (one path
 per COLUMN -- a transpose silently estimates the wrong matrix). The
 Frobenius error vs. assemble_cov decays at the Monte Carlo rate N^{-1/2}
 (the Unit-0 headline check)."
function empirical_cov(paths::AbstractMatrix)
    n, N = size(paths)
    mu = sum(paths, dims = 2) ./ N
    Xc = paths .- mu
    return (Xc * Xc') ./ (N - 1)
end

end # module

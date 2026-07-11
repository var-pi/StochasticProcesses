module Sampling

using LinearAlgebra
export sample_cholesky

"Route 1 (general): (jittered) Cholesky factorization of the covariance matrix.
 Sigma = L * L', X = L * z, z ~ N(0, I). The nugget eps*I restores
 positive-definiteness when the assembled Sigma is singular or numerically
 indefinite (e.g. a grid through t=0 where R(0,s)=0, or near-duplicate grid
 points).
 Unit 0 note: this samples the ZERO-MEAN law X = L*z and deliberately ignores
 meanfn/assemble_mean (Unit 0 uses only zero-mean processes). A future
 non-zero-mean unit must add the mean explicitly: X = assemble_mean(gp, t_grid) .+ L*z."
function sample_cholesky(Sigma::AbstractMatrix, rng; jitter=1e-10)
    n = size(Sigma, 1)
    L = cholesky(Symmetric(Sigma) + jitter * I(n)).L
    return L * randn(rng, n)
end

end # module

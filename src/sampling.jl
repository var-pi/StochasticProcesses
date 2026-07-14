module Sampling

using LinearAlgebra, FFTW
export sample_cholesky, sample_circulant_embedding

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

# The circulant eigenvalues of the symmetric embedding of a stationary covariance
# sequence r -- the single source of truth for the even-extension + FFT, shared by
# sample_circulant_embedding and (via the qualified module path) its exactness test,
# so the two never drift out of sync. m = 2(n-1) for n >= 2; degenerates to m = 1 at
# n = 1 (single-point, non-crashing -- the formula 2(n-1) undercounts there).
function _circulant_eigenvalues(r::AbstractVector)
    isempty(r) && throw(ArgumentError("_circulant_eigenvalues needs a non-empty covariance sequence r"))
    c = vcat(r, r[end-1:-1:2])                 # even (circulant) extension
    return real(fft(c))
end

"Route 4 (stationary only): circulant (FFT) embedding -- exact O(m log m)
 sampling on a uniform grid (Wood-Chan / Dietrich-Newsam). r = [R(0), R(dt),
 ..., R((n-1)*dt)] is the first column of the Toeplitz covariance. Embed into
 a symmetric circulant of size m = 2(n-1); its eigenvalues are lambda =
 real(fft(c)). Requires lambda >= 0 up to a scale-relative tolerance (holds for
 the exponential/periodic kernels; for fBm a larger / minimal-nonnegative
 embedding may be needed -- NOT expressible through this r-only signature, so
 that case must pad r itself before calling). This is the Bochner square root
 made computational: FFT-diagonalization of the (circulant) covariance.
 NOTE: the exact scaling is the per-unit spec detail -- validate against the
 analytic covariance in Unit 1."
function sample_circulant_embedding(r::AbstractVector, rng)
    n = length(r)
    lambda = _circulant_eigenvalues(r)
    m = length(lambda)
    # Scale-relative tolerance: FFT roundoff on the embedding grows with |r|'s
    # amplitude, so a fixed absolute -1e-10 spuriously rejects large-amplitude
    # (e.g. D >> 1) but genuinely PSD kernels. At O(1) amplitude this reduces to
    # the original -1e-10.
    tol = -1e-10 * max(1.0, maximum(abs.(lambda)))
    all(lambda .>= tol) || throw(ArgumentError(
        "circulant embedding not PSD (min eigenvalue $(minimum(lambda)), tolerance $tol); " *
        "r must be padded to a larger nonnegative-definite embedding -- this r-only " *
        "signature has no size parameter to do that automatically"))
    xi = randn(rng, m) .+ im .* randn(rng, m)
    Y  = fft(sqrt.(max.(lambda, 0.0)) .* xi) ./ sqrt(m)
    return real(Y[1:n])                        # imag(Y[1:n]) is a 2nd draw
end

end # module

# ============================================================================
#  Sampling — drawing Gaussian sample paths
# ----------------------------------------------------------------------------
#  To draw X ~ N(0, Sigma) we need a "square root" of the covariance: any matrix A
#  with A * A' = Sigma. Then X = A z with z a vector of independent standard normals
#  has exactly covariance Sigma. This module offers two ways to get that square root:
#
#    sample_cholesky            — general. Factor Sigma = L L' (Cholesky). Works for any
#                                 covariance matrix, costs O(n^3).
#    sample_circulant_embedding — stationary grids only, but fast (O(m log m)): it takes
#                                 the square root with an FFT instead of a factorization.
#
#  Glossary:
#    nugget / jitter    — a tiny eps*I added to Sigma before factoring, to repair a
#                         matrix that is singular or barely non-positive-definite from
#                         rounding. Report eps: too small throws, too large biases Sigma.
#    circulant embedding— embed the (Toeplitz) stationary covariance into a larger
#                         *circulant* matrix, which the FFT diagonalizes exactly. Also
#                         called the Wood–Chan / Dietrich–Newsam method.
# ============================================================================
module Sampling

using LinearAlgebra, FFTW
export sample_cholesky, sample_circulant_embedding, sample_kl

"""
    sample_cholesky(Sigma, rng; jitter = 1e-10) -> Vector

Draw one zero-mean Gaussian sample by Cholesky factorization: with Sigma = L L' and
z ~ N(0, I), the vector X = L z has covariance Sigma.

Arguments:
  - `Sigma`  : an n×n covariance matrix (as from `assemble_cov`).
  - `rng`    : a random-number generator; use a `StableRNG(seed)` for reproducibility.
  - `jitter` : the nugget eps added as eps*I before factoring. It restores
               positive-definiteness when Sigma is exactly singular (e.g. a grid through
               t = 0, where the Brownian-motion row R(0, s) = 0) or is nudged indefinite
               by rounding (e.g. near-duplicate grid points).

Returns a length-n sample vector.

This draws the ZERO-MEAN law X = L z and deliberately ignores any mean function on the
process. A future non-zero-mean process must add the mean back explicitly:
    X = assemble_mean(gp, t_grid) .+ L * z.
"""
function sample_cholesky(Sigma::AbstractMatrix, rng; jitter=1e-10)
    n = size(Sigma, 1)
    L = cholesky(Symmetric(Sigma) + jitter * I(n)).L
    return L * randn(rng, n)
end

# Eigenvalues of the circulant matrix that embeds a stationary covariance sequence r.
#
# For a stationary process the covariance is Toeplitz (constant along diagonals). Mirror
# its first column r = [R(0), R(dt), ...] into an even (symmetric) extension c, and c is
# the first column of a *circulant* matrix. Circulant matrices are diagonalized by the
# FFT, so their eigenvalues are simply real(fft(c)) — the spectrum we need to take a
# square root of. The extension has length m = 2(n-1) for n ≥ 2. (It also underpins the
# faithfulness check in the tests, so the two share this one helper and cannot disagree.)
function _circulant_eigenvalues(r::AbstractVector)
    isempty(r) && throw(ArgumentError("_circulant_eigenvalues needs a non-empty covariance sequence r"))
    c = vcat(r, r[end-1:-1:2])                 # even (circulant) extension of the first column
    return real(fft(c))
end

"""
    sample_circulant_embedding(r, rng) -> Vector

Draw one exact stationary Gaussian sample on a uniform grid in O(m log m), via circulant
embedding (Wood–Chan / Dietrich–Newsam). This is the FFT analogue of Cholesky: it takes
the square root of the covariance with a Fourier transform instead of a factorization.

Arguments:
  - `r`   : the stationary covariance sequence r = [R(0), R(dt), ..., R((n-1)*dt)], i.e.
            the first column of the Toeplitz covariance on the grid.
  - `rng` : a random-number generator (use `StableRNG(seed)` for reproducibility).

Returns a length-n sample path.

How it works: embed r into a circulant of size m = 2(n-1), whose eigenvalues are
lambda = real(fft(c)) (see `_circulant_eigenvalues`); the sqrt of those eigenvalues is
the square root we need. This only works if every lambda ≥ 0 — the precondition that the
embedding is a genuine covariance. It holds for the exponential/OU kernel here; for
processes like fractional Brownian motion (Unit 6) a larger, minimal-nonnegative
embedding can be required, which means padding r before calling (this r-only signature
has no size parameter to do that automatically).
"""
function sample_circulant_embedding(r::AbstractVector, rng)
    n = length(r)
    lambda = _circulant_eigenvalues(r)
    m = length(lambda)
    # Accept eigenvalues down to a *scale-relative* floor, not a fixed -1e-10. FFT roundoff
    # on the embedding grows with the amplitude of r, so a fixed absolute floor would wrongly
    # reject large-amplitude (e.g. D >> 1) but genuinely valid kernels. At O(1) amplitude this
    # is just the original -1e-10.
    tol = -1e-10 * max(1.0, maximum(abs.(lambda)))
    all(lambda .>= tol) || throw(ArgumentError(
        "circulant embedding not PSD (min eigenvalue $(minimum(lambda)), tolerance $tol); " *
        "r must be padded to a larger nonnegative-definite embedding -- this r-only " *
        "signature has no size parameter to do that automatically"))
    xi = randn(rng, m) .+ im .* randn(rng, m)         # complex standard normal noise
    Y  = fft(sqrt.(max.(lambda, 0.0)) .* xi) ./ sqrt(m)
    # The real and imaginary parts of Y are two *independent* samples with the right
    # covariance; we return the real one. (imag(Y[1:n]) would be a second, free draw.)
    return real(Y[1:n])
end

"""
    sample_kl(lambdas, eigfuncs, rng) -> Vector

Draw one zero-mean Gaussian sample by the truncated Karhunen-Loeve expansion:

    X(t) = Σ_{k=1}^{K} √(λ_k) · ξ_k · e_k(t),    ξ_k ~ N(0,1) iid.

This is a third square root of the covariance operator, alongside `sample_cholesky` (the triangular
factor) and `sample_circulant_embedding` (the Bochner/FFT factor): here the square root is taken in
the eigenbasis. With K = all modes it reproduces the (Nystrom-discretized) covariance; with K < all
it is the OPTIMAL K-term approximation, whose discarded variance fraction is
`kl_tail_energy(lambdas, K)`.

Arguments:
  - `lambdas`  : the K eigenvalues (variances of the coefficients), e.g. from `nystrom_eigen`.
  - `eigfuncs` : an n_grid×K matrix whose column k is the eigenfunction e_k sampled on the grid
                 (the second return of `nystrom_eigen`). Determines the output length n_grid.
  - `rng`      : a random-number generator; use `StableRNG(seed)` for reproducibility.

Returns a length-n_grid sample path.

Like `sample_cholesky`, this draws the ZERO-MEAN law and ignores any process mean. A tiny negative
eigenvalue (the Nystrom discretization noise floor) is clamped to zero before the square root, so it
contributes no variance rather than throwing on √(negative).
"""
function sample_kl(lambdas::AbstractVector, eigfuncs::AbstractMatrix, rng)
    K = length(lambdas)
    size(eigfuncs, 2) == K || throw(ArgumentError(
        "sample_kl: eigfuncs must have one column per eigenvalue (got $(size(eigfuncs, 2)) columns " *
        "for $K eigenvalues)"))
    return eigfuncs * (sqrt.(max.(lambdas, 0.0)) .* randn(rng, K))
end

end # module

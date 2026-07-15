# ============================================================================
#  KL — the Karhunen-Loeve eigenbasis of the covariance operator
# ----------------------------------------------------------------------------
#  The second diagonalization (after Spectral/Bochner). By Mercer's theorem the
#  covariance operator (C f)(t) = ∫ R(t,s) f(s) ds has an orthonormal eigenbasis
#  {e_k} with eigenvalues λ_k ≥ 0, and every mean-zero process expands as
#  X_t = Σ_k √λ_k ξ_k e_k(t) with ξ_k iid N(0,1) (the KL expansion). The λ_k are
#  simultaneously (i) eigenvalues of C and (ii) the variances of the expansion
#  coefficients -- one fact, two readings.
#
#  We solve the eigenproblem by NYSTROM QUADRATURE: replace the integral by a
#  quadrature rule (nodes t_i, weights w_i), turning ∫R(t,s)e(s)ds = λe(t) into
#  the discrete (K W) e = λ e with K_ij = R(t_i,t_j), W = diag(w). The catch: K W
#  is NOT symmetric even though R is, so its eigenvectors need not be orthonormal
#  or even real. The fix (the load-bearing step of this module) is to solve the
#  SYMMETRIC problem W^{1/2} K W^{1/2} g = λ g and recover e = W^{-1/2} g, which is
#  orthonormal in the discrete W-weighted L² inner product Σ_i w_i e_k(t_i)² = 1.
#
#  Numerical caution: small eigenvalues are resolved only to poor RELATIVE
#  accuracy -- Nystrom sinks them into a discretization noise floor, below which
#  the λ_k ~ k^{-2} decay flattens. That floor is a property of the discretization,
#  not a modeling error; slope/error claims are meaningful only above it.
# ============================================================================
module KL

using LinearAlgebra
export quad_nodes_weights, nystrom_eigen, trace_diag, kl_tail_energy

"""
    quad_nodes_weights(T; n, rule = :trapezoid) -> (nodes, weights)

Quadrature nodes and weights for integrating over the domain, for the Nystrom eigenproblem.

  - `rule = :trapezoid` : the interval [0, T] with n points t_i = (i-1)*T/(n-1) and composite-
                          trapezoid weights (endpoints h/2, interior h; h = T/(n-1)). Weights sum
                          to T (they integrate the constant 1 exactly).
  - `rule = :periodic`  : the circle of circumference T with n points t_i = (i-1)*T/n (no endpoint,
                          since T is identified with 0) and equal weights T/n -- the periodic
                          trapezoid / rectangle rule, the right quadrature for a torus kernel.

Both return length-n `nodes` and `weights`.
"""
function quad_nodes_weights(T; n, rule = :trapezoid)
    n >= 2 || throw(ArgumentError("quad_nodes_weights needs n >= 2 (got n = $n)"))
    T > 0 || throw(ArgumentError("quad_nodes_weights needs T > 0 (got T = $T)"))
    if rule === :trapezoid
        h = T / (n - 1)
        nodes = [(i - 1) * h for i in 1:n]
        weights = fill(h, n)
        weights[1] = h / 2
        weights[end] = h / 2
    elseif rule === :periodic
        h = T / n
        nodes = [(i - 1) * h for i in 1:n]
        weights = fill(h, n)
    else
        throw(ArgumentError("quad_nodes_weights: unknown rule $(repr(rule)); use :trapezoid or :periodic"))
    end
    return nodes, weights
end

"""
    nystrom_eigen(R, nodes, weights; nev = length(nodes)) -> (lambdas, eigfuncs)

Solve the Karhunen-Loeve eigenproblem ∫ R(t,s) e(s) ds = λ e(t) by symmetrized Nystrom quadrature.

Arguments:
  - `R`               : the covariance kernel R(t, s) (a callable).
  - `nodes`, `weights`: a quadrature rule, e.g. from `quad_nodes_weights`.
  - `nev`             : number of leading eigenpairs to return (default: all n).

Returns:
  - `lambdas`  : the `nev` largest eigenvalues, SORTED DESCENDING.
  - `eigfuncs` : an n×nev matrix; column k is the eigenfunction e_k sampled at `nodes`, normalized
                 in the discrete W-weighted L² inner product (Σ_i w_i e_k(t_i)² = 1), with a
                 canonical sign (largest-magnitude entry made positive).

Method: form K_ij = R(t_i, t_j) and W = diag(weights), then solve the SYMMETRIC problem
W^{1/2} K W^{1/2} g = λ g (symmetric because K is, even though the raw K W is not) and set
e = W^{-1/2} g. Solving K W directly is the classic Nystrom mistake: its eigenvectors are not
W-orthonormal and can come out complex.
"""
function nystrom_eigen(R, nodes, weights; nev = length(nodes))
    n = length(nodes)
    n == length(weights) || throw(ArgumentError(
        "nystrom_eigen: nodes and weights must have equal length (got $n and $(length(weights)))"))
    1 <= nev <= n || throw(ArgumentError("nystrom_eigen needs 1 <= nev <= $n (got nev = $nev)"))
    K = [R(t, s) for t in nodes, s in nodes]
    sw = sqrt.(weights)                          # W^{1/2}
    A = Symmetric((sw * sw') .* K)               # W^{1/2} K W^{1/2}: symmetric even though K W is not
    E = eigen(A)                                 # ascending real eigenvalues (guaranteed by Symmetric)
    idx = sortperm(E.values; rev = true)[1:nev]  # keep the nev largest, descending
    lambdas = E.values[idx]
    eigfuncs = E.vectors[:, idx] ./ sw           # e_k = W^{-1/2} g_k -> discrete-L² orthonormal
    for k in 1:nev                               # canonical sign: largest-magnitude entry positive
        col = @view eigfuncs[:, k]
        col[argmax(abs.(col))] < 0 && (col .*= -1)
    end
    return lambdas, eigfuncs
end

"""
    trace_diag(R, nodes, weights) -> Float64

Quadrature estimate of Tr C = ∫_0^T R(t,t) dt = Σ_i w_i R(t_i, t_i). Used as an assembly/quadrature
sanity check via the trace identity Σ_k λ_k = Tr C: for Brownian motion this is ∫_0^T t dt = T²/2
(NOT T·R(0)=0), and the trapezoid rule is exact on that linear diagonal.
"""
trace_diag(R, nodes, weights) =
    sum(weights[i] * R(nodes[i], nodes[i]) for i in eachindex(nodes))

"""
    kl_tail_energy(lambdas, K) -> Float64

Fraction of total variance discarded by truncating the KL expansion at K modes:
Σ_{k>K} λ_k / Σ_k λ_k. Expects `lambdas` sorted descending (as `nystrom_eigen` returns). K=0 gives
1 (nothing kept); K = length gives 0 (everything kept).
"""
function kl_tail_energy(lambdas, K)
    0 <= K <= length(lambdas) || throw(ArgumentError(
        "kl_tail_energy needs 0 <= K <= $(length(lambdas)) (got K = $K)"))
    total = sum(lambdas)
    total > 0 || throw(ArgumentError("kl_tail_energy: sum of eigenvalues must be positive (got $total)"))
    return sum(@view lambdas[K+1:end]) / total
end

end # module

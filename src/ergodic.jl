# ============================================================================
#  Ergodic — time-average estimators and the Green–Kubo coefficient
# ----------------------------------------------------------------------------
#  For a stationary process with an integrable correlation, the time-average of
#  ONE long path converges to the mean (an L² law of large numbers, Pavliotis
#  Prop. 1.16). This module holds the estimators that check that loop:
#
#    - path-side (consume sampled paths): the running time-average along one path,
#      the variance of the time-average across an ENSEMBLE, and the integrated
#      mean-square displacement. A path MATRIX is n_grid × N — one path per COLUMN
#      (a transpose silently estimates the wrong object).
#    - covariance-side (operate on the stationary covariance sequence r): the
#      Green–Kubo transport coefficient D* = ∫₀^∞ C, and the exact finite-T
#      variance identity of Lemma 1.17.
#
#  These sample the ZERO-MEAN law (like the samplers), so the mean-square of the
#  time-average IS its variance about the true mean μ = 0.
#
#  A note on two constants that are easy to confuse (they coincide only at α = 1):
#    R(0) = D/α           the variance at zero lag.
#    D*   = ∫₀^∞ C = D/α² the transport / Green–Kubo coefficient (Pavliotis
#                         Example 1.18). The finite-T variance decays to 2D*/T,
#                         so the rate constant carries α² — not α.
# ============================================================================
module Ergodic

export running_time_average, time_average_variance, mean_square_displacement,
       green_kubo, time_average_variance_exact

# --- private: cumulative trapezoid integral of one path ----------------------
# I_k = ∫₀^{t_k} X_s ds with t_k = (k-1)·dt, by the trapezoid rule: I[1] = 0,
# I[k] = I[k-1] + dt·(X[k-1] + X[k])/2. Shared by all three path estimators so
# they cannot drift into three subtly different discretizations.
function _cumulative_integral(path::AbstractVector, dt)
    n = length(path)
    I = zeros(float(eltype(path)), n)
    @inbounds for k in 2:n
        I[k] = I[k-1] + dt * (path[k-1] + path[k]) / 2
    end
    return I
end

"""
    running_time_average(path, dt) -> Vector

The running time-average A_T = (1/T)∫₀ᵀ X_s ds along ONE sampled path, as a function of the upper
limit T = t_k = (k-1)·dt. Returns a vector the length of `path`; `A[1] = path[1]` is the T→0 limit
(the time-average over a vanishing window is the initial value).
"""
function running_time_average(path::AbstractVector, dt)
    isempty(path) && throw(ArgumentError("running_time_average needs a non-empty path"))
    I = _cumulative_integral(path, dt)
    n = length(path)
    A = similar(I)
    A[1] = path[1]
    @inbounds for k in 2:n
        A[k] = I[k] / ((k - 1) * dt)
    end
    return A
end

"""
    time_average_variance(paths, dt) -> Vector

The ensemble variance of the running time-average at each T, across a path MATRIX (`n_grid × N`, one
path per COLUMN). Computed as the mean-square `mean(A[k,:].^2)` — which equals Var(A_T) because the
sampled law is zero-mean (repo-wide): the mean μ = 0 is known exactly, so the mean-square about 0 is
the unbiased estimator of the variance-about-the-true-mean that Lemma 1.17 predicts, with no N vs
N−1 ambiguity. Returns a vector of length `size(paths, 1)`.
"""
function time_average_variance(paths::AbstractMatrix, dt)
    (size(paths, 1) >= 1 && size(paths, 2) >= 1) ||
        throw(ArgumentError("time_average_variance needs a non-empty path matrix (n_grid × N)"))
    n, N = size(paths)
    acc = zeros(float(eltype(paths)), n)          # accumulate Σ_j A_j.^2 in one pass (no n×N matrix)
    for j in 1:N
        acc .+= abs2.(running_time_average(view(paths, :, j), dt))
    end
    return acc ./ N
end

"""
    mean_square_displacement(paths, dt) -> Vector

The integrated mean-square displacement E[(∫₀ᵗ X_s ds)²] at each t, across a path MATRIX (`n_grid ×
N`, one path per COLUMN). For a stationary process with D* = ∫₀^∞ C this grows like 2D*·t. Returns a
vector of length `size(paths, 1)`. (Exactly `t_k²·time_average_variance` on the same matrix, since
the running average is the integral divided by t_k.)
"""
function mean_square_displacement(paths::AbstractMatrix, dt)
    (size(paths, 1) >= 1 && size(paths, 2) >= 1) ||
        throw(ArgumentError("mean_square_displacement needs a non-empty path matrix (n_grid × N)"))
    n, N = size(paths)
    acc = zeros(float(eltype(paths)), n)          # accumulate Σ_j I_j.^2 in one pass (no n×N matrix)
    for j in 1:N
        acc .+= abs2.(_cumulative_integral(view(paths, :, j), dt))
    end
    return acc ./ N
end

"""
    green_kubo(r, dt) -> Float64

The Green–Kubo transport coefficient D* = ∫₀^∞ C(u) du, by the trapezoid rule over the covariance
sequence r = [C(0), C(dt), C(2dt), ...]. For the OU kernel (D/α)·exp(-α|u|) this converges to
D/α² (Pavliotis Example 1.18) — NOT R(0) = D/α (the two coincide only at α = 1). A length-1 r has a
zero-width domain, so D* = 0.
"""
function green_kubo(r::AbstractVector, dt)
    isempty(r) && throw(ArgumentError("green_kubo needs a non-empty covariance sequence r"))
    length(r) == 1 && return zero(float(eltype(r)))
    return dt * (sum(r) - (r[1] + r[end]) / 2)
end

"""
    time_average_variance_exact(r, dt) -> Vector

The exact finite-T variance identity (Pavliotis Lemma 1.17): at each T = t_k = (k-1)·dt,

    Var( (1/T)∫₀ᵀ X_s ds ) = (2/T²) ∫₀ᵀ (T-u) C(u) du,

with the inner integral discretized by the trapezoid rule over the covariance sequence r. Returns a
vector the length of `r`; `V[1] = r[1] = C(0)` is the T→0 limit.

Reduced to O(n) via cumulative sums S1_k = Σ_{j≤k} r_j and S2_k = Σ_{j≤k} j·r_j:
V_k = (2/(k-1)²)·(k·S1_k − S2_k − (k-1)·r_1/2) for k ≥ 2. (On a uniform grid dt cancels analytically;
it is kept in the signature for API symmetry with the other estimators.)
"""
function time_average_variance_exact(r::AbstractVector, dt)
    isempty(r) && throw(ArgumentError("time_average_variance_exact needs a non-empty covariance sequence r"))
    n = length(r)
    V = zeros(float(eltype(r)), n)
    V[1] = r[1]
    S1 = r[1]; S2 = r[1]                      # Σ_{j≤k} r_j and Σ_{j≤k} j·r_j, seeded with j = 1
    for k in 2:n
        S1 += r[k]
        S2 += k * r[k]
        V[k] = 2 / (k - 1)^2 * (k * S1 - S2 - (k - 1) * r[1] / 2)
    end
    return V
end

end # module

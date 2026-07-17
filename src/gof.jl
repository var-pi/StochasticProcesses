# ============================================================================
#  GOF — goodness-of-fit utilities
# ----------------------------------------------------------------------------
#  DELIBERATE ARCHITECTURAL EXCEPTION: every other src/ module is organized by
#  operation on the covariance operator C (kernels, its diagonalizations, its
#  square roots -- see the module docstring in StochasticProcesses.jl). This
#  module is NOT that: ks_statistic has nothing to do with C. It lives in src/
#  anyway, for two concrete reasons: (1) two separate units need it -- Unit 3's
#  Cramer-Wold and KL-coefficient checks here, and Unit 5's random-walk-to-BM
#  convergence checks later -- so it belongs to neither on its own, and (2) a
#  small, deterministic, hand-computed test is exactly the guard the src/-vs-
#  test/ split exists to provide (see CLAUDE.md's testing conventions). State
#  the exception plainly rather than stretching "operation on C" to cover it.
# ============================================================================
module GOF

export ks_statistic

"""
    ks_statistic(samples, cdf) -> Float64

Kolmogorov–Smirnov sup-distance between the empirical CDF of `samples` and a target
`cdf` (a callable `x -> F(x)`).

Given order statistics x_(1) ≤ x_(2) ≤ ... ≤ x_(n), the empirical CDF F_n jumps by 1/n
at each x_(i): F_n(x) = i/n for x_(i) ≤ x < x_(i+1). At the jump located at x_(i), F_n
takes BOTH values (i-1)/n just below and i/n at and above -- so the sup-distance
max_x |F_n(x) - F(x)| can be realized by either side of that jump. This function
therefore checks both gaps at every order statistic:

  - the UPPER gap  i/n - F(x_(i))       (F_n just after the jump vs. the target)
  - the LOWER gap  F(x_(i)) - (i-1)/n   (the target vs. F_n just before the jump)

and returns the max over all i and both sides. Checking only one side (a "D+"- or
"D-"-only statistic) silently misses the sup whenever it is attained on the other side
-- see the two skewed-fixture tests in the test suite, each of which pins down exactly
one side.

`samples` need not be sorted (this function sorts internally). Throws `ArgumentError`
on empty input, since the sup-distance over zero order statistics is undefined.
"""
function ks_statistic(samples, cdf)
    n = length(samples)
    n == 0 && throw(ArgumentError("ks_statistic: empty sample"))
    xs = sort(samples)
    d = 0.0
    for (i, x) in enumerate(xs)
        Fx = cdf(x)
        d = max(d, i / n - Fx, Fx - (i - 1) / n)   # both sides of the jump
    end
    return d
end

end # module

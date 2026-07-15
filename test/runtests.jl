# ============================================================================
#  Test suite — deterministic analytic identities
# ----------------------------------------------------------------------------
#  These tests pin the library against results we can work out by hand: kernel
#  symmetry, closed-form covariance entries, exact Fourier round-trips, and so on.
#  They run in CI and use tight tolerances.
#
#  The heavier Monte-Carlo demonstrations (log-log convergence slopes) live in
#  experiments/ and are run by hand, not in CI. A handful of fast statistical
#  guards (5% Frobenius error at a fixed seed) do appear here — enough to catch a
#  gross sampler-wiring bug without the cost of a full convergence study.
#
#  Structure: one nested @testset per idea, named so a failure points straight at
#  the concept it broke.
# ============================================================================
using Test, StochasticProcesses, LinearAlgebra, StableRNGs, FFTW

@testset "StochasticProcesses" begin
    @testset "package loads" begin
        @test isdefined(Main, :StochasticProcesses)
    end

    # ------------------------------------------------------------------ kernels
    @testset "Kernels" begin
        @testset "symmetry  R(t,s) == R(s,t)" begin
            @test brownian_motion_kernel(0.3, 0.7) == brownian_motion_kernel(0.7, 0.3)
            @test exponential_kernel(0.3, 0.7) == exponential_kernel(0.7, 0.3)
        end

        @testset "known closed-form values" begin
            @test brownian_motion_kernel(0.3, 0.7) == 0.3                 # min(t,s)
            @test exponential_kernel(1.0, 1.0) == 1.0                     # (D/alpha)*e^0, D=alpha=1
            @test exponential_kernel(0.0, 1.0; D = 2.0, alpha = 1.0) ≈ 2.0 * exp(-1.0)
        end

        @testset "stationarity  R depends only on |t-s|" begin
            # The defining OU property: shifting both times leaves R unchanged. The first
            # pair is bit-exact under the shift; the keyword pair differs by one unit in the
            # last place in |t-s| (0.3-0.7 vs 0.0-0.4), hence ≈ — still nowhere near any
            # non-stationary bug, which would move R by an O(1) amount.
            @test exponential_kernel(0.2, 0.5) == exponential_kernel(1.2, 1.5)
            @test exponential_kernel(0.3, 0.7; D = 2.0, alpha = 3.0) ≈
                  exponential_kernel(0.0, 0.4; D = 2.0, alpha = 3.0)
        end

        @testset "periodic_kernel (torus)" begin
            @test periodic_kernel(0.2, 0.7) == periodic_kernel(0.7, 0.2)          # symmetry
            @test periodic_kernel(0.3, 0.3; D = 2.0) == 2.0                        # R(0) = D
            # wrap-around: the distance from 0.1 to 0.9 is the SHORT arc 0.2, not 0.8. A missing wrap
            # (using |t-s| directly) would give exp(-0.8) here -- an O(1) error this test catches.
            @test periodic_kernel(0.1, 0.9; alpha = 1.0) ≈ exp(-0.2)
            @test periodic_kernel(0.1, 0.9) != exponential_kernel(0.1, 0.9)        # the wrap distinguishes it from OU
            # inside a half-period the periodic distance is just |t-s| (hand value)
            @test periodic_kernel(0.2, 0.5; D = 1.0, alpha = 2.0) ≈ exp(-2 * 0.3)
            # positive-definite on a uniform circle grid: all eigenvalues >= 0 (the full-spectrum claim)
            circle = range(0, 1; length = 17)[1:end-1]        # 16 distinct circle points (drop the wrap point)
            Σ = assemble_cov(GaussianProcess(periodic_kernel), circle)
            @test all(eigvals(Matrix(Σ)) .>= -1e-10)
        end
    end

    # -------------------------------------------------- assembly & empirical cov
    @testset "GaussianProcess: assembly & empirical covariance" begin
        t_grid = range(0, 1; length = 8)
        gp = GaussianProcess(brownian_motion_kernel)
        Sigma = assemble_cov(gp, t_grid)

        @testset "assemble_cov / assemble_mean" begin
            @test issymmetric(Sigma)
            @test size(Sigma) == (8, 8)
            @test all(eigvals(Matrix(Sigma)) .>= -1e-10)        # positive semidefinite (BM Σ is in fact PD)
            @test assemble_mean(gp, t_grid) == zeros(8)         # default zero mean
        end

        @testset "path orientation is n_grid × N (one path per column)" begin
            # Orientation is load-bearing: a transpose silently estimates the wrong matrix.
            # Use a NON-square fixture so n_grid and N cannot be confused.
            paths = [1.0 2.0 3.0;
                     4.0 5.0 6.0]                               # n_grid = 2, N = 3
            @test size(paths, 1) == 2                           # n_grid rows
            C = empirical_cov(paths)
            @test size(C) == (2, 2)                             # n_grid × n_grid, not N × N
            # Target worked out by hand (not a re-run of the function's own formula): the row
            # means are 2 and 5, so both centered rows are [-1 0 1], and (Xc*Xc')/(N-1) = [2 2; 2 2]/2.
            @test C ≈ [1.0 1.0; 1.0 1.0]
            # A single path leaves the (N-1) denominator undefined: guard, don't return NaN.
            @test_throws ArgumentError empirical_cov(zeros(3, 1))
        end

        @testset "assemble_cov returns a Symmetric wrapper" begin
            # The wrapper is part of the contract: issymmetric alone can't detect a dropped
            # wrapper, because the BM Σ is exactly symmetric as a plain Matrix too.
            @test Sigma isa Symmetric
        end

        @testset "closed-form covariance entry" begin
            # One hand-checked entry catches an assembly transpose/index bug that symmetry cannot.
            @test Sigma[2, 3] == min(t_grid[2], t_grid[3])
        end

        @testset "non-zero mean via the keyword constructor" begin
            gpm = GaussianProcess(brownian_motion_kernel; meanfn = t -> 2.0 + t)
            @test assemble_mean(gpm, [0.0, 0.5, 1.0]) == [2.0, 2.5, 3.0]
        end
    end

    # ------------------------------------------------------ Cholesky sampler
    @testset "sample_cholesky: jitter control & reproducibility" begin
        # A BM grid through t = 0 has an all-zero first row (R(0,s) = 0), so Σ is exactly
        # singular — the ideal stress test for the nugget.
        t_grid = range(0, 1; length = 64)
        gp = GaussianProcess(brownian_motion_kernel)
        Sigma = assemble_cov(gp, t_grid)

        @testset "singular Σ throws at jitter = 0 (negative control)" begin
            @test_throws PosDefException sample_cholesky(Sigma, StableRNG(1); jitter = 0.0)
        end

        @testset "the nugget restores a full-length draw" begin
            x = sample_cholesky(Sigma, StableRNG(1); jitter = 1e-10)
            @test length(x) == length(t_grid)
            @test all(isfinite, x)
        end

        @testset "reproducibility: the seed determines the draw" begin
            @test sample_cholesky(Sigma, StableRNG(42)) == sample_cholesky(Sigma, StableRNG(42))
            @test sample_cholesky(Sigma, StableRNG(42)) != sample_cholesky(Sigma, StableRNG(43))
        end
    end

    @testset "Cholesky sampler + empirical_cov reproduce Σ" begin
        # The headline consistency claim, as a fast CI guard: paths from sample_cholesky,
        # summarized by empirical_cov, must reproduce the assembled Σ. The tolerance is loose
        # (5% relative Frobenius) so it is robust to Monte-Carlo noise at a fixed seed, yet
        # still catches a sampler-wiring bug: swapping L for U, transposing, or mis-centering
        # lands around 93% error, while the correct sampler lands near 0.8%.
        gp     = GaussianProcess(brownian_motion_kernel)
        t_grid = range(0, 1; length = 16)
        Sigma  = Matrix(assemble_cov(gp, t_grid))
        rng    = StableRNG(7)
        N      = 4000
        paths  = reduce(hcat, (sample_cholesky(Sigma, rng) for _ in 1:N))
        rel_err = norm(empirical_cov(paths) .- Sigma) / norm(Sigma)
        @test rel_err < 0.05
    end

    @testset "exponential_kernel through the full GP pipeline" begin
        # A smoke test end-to-end: the OU kernel assembles to a genuinely positive-definite Σ
        # and is sampleable. isposdef also records, in test form, that this kernel is
        # well-conditioned — unlike Brownian motion, it is not a singular edge case.
        gp = GaussianProcess(exponential_kernel)
        t  = range(0, 5; length = 20)
        Σ  = assemble_cov(gp, t)
        @test isposdef(Matrix(Σ))                       # well-conditioned, unlike BM
        x  = sample_cholesky(Σ, StableRNG(1))
        @test length(x) == 20 && all(isfinite, x)
    end

    # --------------------------------------------- Bochner transform of a covariance
    @testset "Spectral — Bochner transform of the OU covariance" begin
        # Deterministic throughout: r is a fixed sampled covariance sequence, no RNG.
        # n*dt = 20 ≈ 20 correlation times, so the OU covariance is well resolved.
        D = 1.0; alpha = 1.0; dt = 0.05; n = 400
        r = [exponential_kernel(0.0, k * dt; D = D, alpha = alpha) for k in 0:n-1]   # R(0) = D/alpha = 1
        omega, Shat = bochner_forward(r, dt)
        R0 = D / alpha

        @testset "normalization  ∫ Ŝ dω ≈ R(0)" begin
            # A ~3% check, not machine precision: truncating r at n*dt, the trapezoid rule, and
            # the dropped Nyquist bin each shave a little off, so the integral lands near
            # 0.975*R(0). Deterministic, but not exact — hence rtol = 3e-2.
            @test isapprox(spectral_variance(omega, Shat), R0; rtol = 3e-2)
        end

        @testset "dropped 1/2π lands on 2π·R(0), not R(0)" begin
            # The "2π lesson" as a passing test. Scaling Shat by 2π after the fact is exactly what
            # a missing dt/(2π) factor inside bochner_forward would produce, so the integral moves
            # to 2π·R(0). (This illustrates the consequence; the real regression guard is the
            # normalization test above, which already fails hard — ~2π off — if that factor is dropped.)
            @test isapprox(spectral_variance(omega, 2pi .* Shat), 2pi * R0; rtol = 3e-2)
            @test !isapprox(spectral_variance(omega, 2pi .* Shat), R0; rtol = 0.2)   # provably not R(0)
        end

        @testset "Lorentzian shape match at interior frequencies" begin
            # The one-sided Shat approximates the one-sided density 2*S(omega) at interior omega,
            # so we compare to 2*S_analytic, not S_analytic — mixing them up is a clean factor-of-2
            # bug. The realized error is ≤ 0.11%, so rtol = 1e-2 leaves headroom without being a
            # vacuous 50×-loose placeholder.
            S_analytic(w) = D / (pi * (w^2 + alpha^2))            # two-sided OU (Lorentzian) density
            for w0 in (0.5, 1.0, 2.0)
                i = argmin(abs.(omega .- w0))
                @test isapprox(Shat[i], 2 * S_analytic(omega[i]); rtol = 1e-2)
            end
        end

        @testset "one-sided ↔ two-sided consistency" begin
            # Locks the relationship the shape match above relies on: the one-sided spectrum is the
            # two-sided one restricted to omega >= 0 with interior bins doubled and DC left alone.
            o1, S1 = omega, Shat                                  # onesided = true (the default)
            o2, S2 = bochner_forward(r, dt; onesided = false)
            keep = o2 .>= 0
            @test all(o1 .>= 0)                                   # one-sided grid is nonnegative
            @test o1 == o2[keep]                                  # same omega >= 0 grid
            @test isapprox(S1[1], S2[keep][1]; atol = 1e-12)      # DC bin is not doubled
            @test isapprox(S1[2:end], 2 .* S2[keep][2:end]; rtol = 1e-12)   # interior bins doubled
            # Both conventions carry the same total power: one-sided over omega >= 0 equals
            # two-sided over the whole line.
            @test isapprox(spectral_variance(o1, S1), R0; rtol = 3e-2)
            @test isapprox(spectral_variance(o2, S2), R0; rtol = 3e-2)
        end

        @testset "round-trip: bochner_inverse recovers the covariance" begin
            # bochner_inverse needs the *unsorted, two-sided* spectrum (see its docstring), which is
            # why it is private. We build that spectrum from the same private _raw_transform that
            # bochner_forward uses — reusing the one definition of the even-extension/FFT/scale rather
            # than re-deriving it here — and pass the matching frequency spacing dOmega = 2π/(m*dt).
            rsym, S_unsorted, _ = StochasticProcesses.Spectral._raw_transform(r, dt)
            m = length(rsym)
            dOmega = 2pi / (m * dt)
            r_rt = StochasticProcesses.Spectral.bochner_inverse(S_unsorted, dOmega)
            @test isapprox(r_rt[1:n], r; atol = 1e-10)
        end

        @testset "spectral_variance: hand-computed trapezoid on a non-uniform grid" begin
            # A tiny worked example catches off-by-one / endpoint bugs that a self-consistent test
            # cannot: trapezoid of S = [1,3,2] on omega = [0,1,3] is 0.5*(1+3)*1 + 0.5*(3+2)*2 = 7.
            @test spectral_variance([0.0, 1.0, 3.0], [1.0, 3.0, 2.0]) == 7.0
        end

        @testset "rectangular vs trapezoidal integrator: the DC weighting differs" begin
            # On a uniform grid with a non-zero DC bin the two integrators must disagree, and by a
            # known amount: rectangular counts DC in full, trapezoid halves both DC and the top bin.
            # This pins them as distinct tools.
            ow = [0.0, 1.0, 2.0, 3.0]; Sw = [1.0, 3.0, 2.0, 4.0]
            @test spectral_power(ow, Sw)    == 10.0   # dOmega*sum = 1*(1+3+2+4); DC counted in full
            @test spectral_variance(ow, Sw) == 7.5    # trapezoid halves DC (0.5*1) and the top bin
        end

        # (Folding onto one side is exercised implicitly by the tests above, since bochner_forward
        #  folds by default: interior bins doubled, DC not.)

        @testset "degenerate-input guards" begin
            # Before these guards, the empty case crashed inside FFTW planning and the single-point
            # case had an undefined integral — both now raise a clear, catchable ArgumentError.
            @test_throws ArgumentError bochner_forward(Float64[], dt)
            @test_throws ArgumentError spectral_variance([0.0], [5.0])
            @test_throws ArgumentError spectral_power([0.0], [5.0])
        end
    end

    # ------------------------------------------- periodogram & Welch estimators
    @testset "Spectral — periodogram & Welch estimators" begin
        dt = 0.1
        # A shared deterministic multi-tone fixture (no RNG): three orthogonal interior cosines,
        # reused by several tests below. Orthogonality makes its mean-square a clean Σ a_k^2 / 2.
        L0 = 256; tt = dt .* (0:L0-1)
        ks = (5, 11, 23); as = (1.0, 0.5, 0.3)                  # distinct interior DFT bins
        x  = sum(a .* cos.((2pi * k / (L0 * dt)) .* tt) for (k, a) in zip(ks, as))

        @testset "Parseval normalization on a deterministic signal" begin
            # The one-sided spectral integral of a record returns its mean-square. This pins dt, the
            # 1/2π factor, and the one-sided fold at once. We integrate with spectral_power (the
            # rectangular Parseval sum, the right tool for a discrete periodogram). Bin-aligned
            # orthogonal cosines leak nothing, so this is an exact DFT identity — measured relative
            # error ~1e-15 — and a tight rtol makes a real ~1% normalization bug fail rather than
            # hide under slack.
            omega, Shat = welch_psd(x, dt; nseg = 1, window = :none)
            ms = sum(a^2 for a in as) / 2                          # mean-square of orthogonal cosines
            @test isapprox(spectral_power(omega, Shat), ms; rtol = 1e-9)
        end

        @testset "peak location of a pure cosine" begin
            # Welch of a pure cosine places its mass at omega0. Put omega0 exactly on a DFT bin so
            # argmax is unambiguous; this catches an angular-vs-ordinary (factor of 2π) error in the
            # frequency axis.
            L = 256; k = 8; f0 = k / (L * dt); omega0 = 2pi * f0
            t = dt .* (0:L-1)
            xc = cos.(omega0 .* t)
            oc, Sc = welch_psd(xc, dt; nseg = 1, window = :none)
            @test isapprox(oc[argmax(Sc)], omega0; atol = (oc[2] - oc[1]))     # within one bin
        end

        @testset "raw_periodogram is the single-segment unwindowed Welch case" begin
            or, Sr = raw_periodogram(x, dt)
            ow, Sw = welch_psd(x, dt; nseg = 1, window = :none)
            @test or == ow
            @test Sr == Sw
        end

        @testset "Hann window normalizes by window power U, not length L" begin
            # With the Hann window, Welch integrates to the window-weighted mean-square
            # wms = sum(win.^2 .* x.^2) / sum(win.^2). A U→L bug divides by L instead of U, rescaling
            # Shat by U/L ≈ 0.37 and landing near 0.19 = 0.37*wms — so the second assertion pins that
            # the wrong normalizer is provably off.
            Lh = 256; th = dt .* (0:Lh-1); kh = 16                 # interior bin, well off DC and Nyquist
            xh  = cos.((2pi * kh / (Lh * dt)) .* th)
            win = [0.5 - 0.5 * cos(2pi * j / (Lh - 1)) for j in 0:Lh-1]   # same Hann as welch_psd
            U   = sum(abs2, win)                                   # window power
            wms = sum(win.^2 .* xh.^2) / U                         # window-weighted mean-square (exact target)
            oh, Sh = welch_psd(xh, dt; nseg = 1, window = :hann)
            # Again an exact DFT identity (relative error ~1e-15), so rtol = 1e-9 still catches a real ~1% bug.
            @test isapprox(spectral_power(oh, Sh), wms; rtol = 1e-9)              # U correct
            @test !isapprox(spectral_power(oh, (U / Lh) .* Sh), wms; rtol = 0.2)  # U→L variant provably wrong
        end

        @testset "multi-segment averaging (Welch's reason to exist)" begin
            # With k identical segments the average collapses to a single segment, so Welch(nseg=k)
            # must byte-equal the raw periodogram of one copy. Perturbing one segment must change the
            # average — which exercises the segment/averaging loop directly, no integrator involved.
            seg = Float64[1, -2, 3, 0.5, -1.5, 2, -0.5, 1, 0.25]        # any fixed real vector
            om, Sm   = welch_psd(vcat(seg, seg, seg, seg), dt; nseg = 4, window = :none)
            or1, Sr1 = raw_periodogram(seg, dt)
            @test om == or1 && Sm == Sr1
            bad = copy(seg); bad[1] += 1.0                              # perturb one of the four segments
            _, Smb = welch_psd(vcat(seg, seg, seg, bad), dt; nseg = 4, window = :none)
            @test Smb != Sr1                                            # the average must change
        end

        @testset "overlapping segments average correctly" begin
            # The experiment runs 50% Hann overlap, so assert the overlap path numerically, not just
            # for finiteness. Welch's normalization and one-sided fold are linear, so Welch of
            # overlapping segments equals the mean of the raw periodograms of those same segments — an
            # independent recompute that exercises the hop / segment-count / accumulation loop (with a
            # hand count of how many segments there are).
            xo = Float64[2, -1, 0.5, 3, -2, 1, 0.25, -0.5]             # length 8
            oo, So = welch_psd(xo, dt; nseg = 2, noverlap = 2, window = :none)   # L=4, hop=2 -> 3 segments
            starts = 1:2:(length(xo) - 4 + 1)                          # 1, 3, 5  (hand count = 3)
            @test length(starts) == 3
            raws  = [raw_periodogram(xo[s:s+3], dt) for s in starts]
            Smean = sum(r[2] for r in raws) ./ length(starts)
            @test oo == raws[1][1]                                     # same omega grid
            @test isapprox(So, Smean; rtol = 1e-12)                    # the overlap average matches
        end

        @testset "degenerate-input guards" begin
            # Each of these used to fail badly: nseg <= 0 threw an opaque DivideError; noverlap >= L
            # made the hop non-positive and hung forever (the start index never advances); noverlap > L
            # walked the index negative into a BoundsError; L == 1 silently returned NaN (0/0 in the
            # Hann formula); and an unknown window symbol silently fell through to rectangular. All now
            # raise a clear ArgumentError.
            @test_throws ArgumentError welch_psd(x, dt; nseg = 0)
            @test_throws ArgumentError welch_psd(x, dt; nseg = -1)
            @test_throws ArgumentError welch_psd(ones(4), dt; nseg = 4)                # L == 1
            @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 2, noverlap = 4)   # hop == 0 (would hang)
            @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 2, noverlap = 6)   # hop < 0
            @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 1, noverlap = -1)  # noverlap < 0
            @test_throws ArgumentError welch_psd(x, dt; nseg = 1, window = :hamming)    # unknown window
        end
    end

    # ----------------------------------------------- circulant-embedding sampler
    @testset "Spectral — circulant-embedding sampler" begin
        D = 1.0; alpha = 1.0; dt = 0.1; n = 32
        r = [exponential_kernel(0.0, k * dt; D = D, alpha = alpha) for k in 0:n-1]
        # The circulant eigenvalues, computed once via the same private helper the sampler uses
        # (so we test the shipped code path, not a re-derived copy that could drift from it).
        lambda = StochasticProcesses.Sampling._circulant_eigenvalues(r)

        @testset "the embedding is PSD for the OU kernel" begin
            @test all(lambda .>= -1e-10)
        end

        @testset "eigenvalues reconstruct the covariance exactly" begin
            @test isapprox(real(ifft(lambda))[1:n], r; atol = 1e-10)
        end

        @testset "shape & reproducibility" begin
            x = sample_circulant_embedding(r, StableRNG(1))
            @test length(x) == n && all(isfinite, x)
            @test sample_circulant_embedding(r, StableRNG(7)) == sample_circulant_embedding(r, StableRNG(7))
            @test sample_circulant_embedding(r, StableRNG(7)) != sample_circulant_embedding(r, StableRNG(8))
        end

        @testset "negative control: a non-PSD sequence throws" begin
            # bad_r is chosen so its circulant embedding has a negative eigenvalue (min ≈ -0.3),
            # tripping the PSD guard — the precondition that keeps the embedding an exact covariance.
            # This is the case that genuinely fires for processes like fractional BM (Unit 6).
            bad_r = [1.0, -0.9, 0.7, -0.9]
            @test_throws ArgumentError sample_circulant_embedding(bad_r, StableRNG(1))
        end

        @testset "statistical consistency reproduces Σ" begin
            # The sampler's sqrt-eigenvalue / sqrt-m scaling must reproduce the Toeplitz Σ. Loose 5%
            # Frobenius so it is robust to Monte-Carlo noise at a fixed seed, yet still catches a gross
            # scaling bug. (The exact reconstruction test above does not cover the *sampling* scaling.)
            Sigma = [r[abs(i - j) + 1] for i in 1:n, j in 1:n]
            rng   = StableRNG(2024)
            N     = 4000
            paths = reduce(hcat, (sample_circulant_embedding(r, rng) for _ in 1:N))
            @test norm(empirical_cov(paths) .- Sigma) / norm(Sigma) < 0.05
        end

        @testset "degenerate input: empty r throws" begin
            # An empty r would slip past the (vacuously true) PSD check and crash inside FFTW planning;
            # the guard turns that into a clear, catchable ArgumentError.
            @test_throws ArgumentError sample_circulant_embedding(Float64[], StableRNG(1))
        end
    end

    @testset "public surface" begin
        # Regression guard on the exported names: catches a dropped `export` or a re-export block that
        # falls out of sync with a submodule.
        for f in (:brownian_motion_kernel, :exponential_kernel, :periodic_kernel, :GaussianProcess,
                  :assemble_cov, :assemble_mean, :empirical_cov, :sample_cholesky,
                  :sample_circulant_embedding, :bochner_forward, :spectral_variance,
                  :spectral_power, :welch_psd, :raw_periodogram)
            @test isdefined(StochasticProcesses, f)
        end
    end
end

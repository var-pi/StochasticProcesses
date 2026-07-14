using Test, StochasticProcesses, LinearAlgebra, StableRNGs

@testset "StochasticProcesses" begin
    @testset "package loads" begin
        @test isdefined(Main, :StochasticProcesses)
    end

    @testset "Kernels" begin
        # symmetry R(t,s) == R(s,t)
        @test brownian_motion_kernel(0.3, 0.7) == brownian_motion_kernel(0.7, 0.3)
        @test exponential_kernel(0.3, 0.7) == exponential_kernel(0.7, 0.3)
        # known values
        @test brownian_motion_kernel(0.3, 0.7) == 0.3                 # min(t,s)
        @test exponential_kernel(1.0, 1.0) == 1.0                     # (D/α)·e^0, D=α=1
        @test exponential_kernel(0.0, 1.0; D = 2.0, alpha = 1.0) ≈ 2.0 * exp(-1.0)

        # Stationarity: R depends only on |t-s| (the defining OU property). The first
        # pair is bit-exact under the shift; the keyword-form pair differs by one ULP
        # in |t-s| (0.3-0.7 vs 0.0-0.4), so ≈ — still O(1) away from any non-stationary bug.
        @test exponential_kernel(0.2, 0.5) == exponential_kernel(1.2, 1.5)
        @test exponential_kernel(0.3, 0.7; D = 2.0, alpha = 3.0) ≈
              exponential_kernel(0.0, 0.4; D = 2.0, alpha = 3.0)
    end

    @testset "GaussianProcess assembly + empirical_cov" begin
        t_grid = range(0, 1; length = 8)
        gp = GaussianProcess(brownian_motion_kernel)

        Sigma = assemble_cov(gp, t_grid)
        @test issymmetric(Sigma)
        @test size(Sigma) == (8, 8)
        @test all(eigvals(Matrix(Sigma)) .>= -1e-10)            # PSD (BM Σ is PD)
        @test assemble_mean(gp, t_grid) == zeros(8)

        # --- Orientation is load-bearing: paths are n_grid × N (one path per COLUMN).
        # Use a NON-square matrix so n_grid and N cannot be confused.
        paths = [1.0 2.0 3.0;
                 4.0 5.0 6.0]                                   # n_grid = 2, N = 3
        @test size(paths, 1) == 2                               # n_grid rows
        C = empirical_cov(paths)
        @test size(C) == (2, 2)                                 # n_grid × n_grid, NOT N × N
        # Hand-computed target (NOT a re-run of the function's own formula): row means
        # are 2 and 5, so both centred rows are [-1 0 1]; (Xc*Xc')/(N-1) = [2 2; 2 2]/2.
        @test C ≈ [1.0 1.0; 1.0 1.0]
        # The (N-1) denominator is undefined for a single path: guard, don't return NaN.
        @test_throws ArgumentError empirical_cov(zeros(3, 1))

        # --- assemble_cov honours its Symmetric *type* contract (issymmetric alone
        # can't see a dropped wrapper: BM Σ is exactly symmetric as a plain Matrix too).
        @test Sigma isa Symmetric
        # A closed-form Σ entry: catches an assembly transpose/index bug that symmetry cannot.
        @test Sigma[2, 3] == min(t_grid[2], t_grid[3])
        # Non-zero mean AND the keyword constructor are exercised (not only the zero default).
        gpm = GaussianProcess(brownian_motion_kernel; meanfn = t -> 2.0 + t)
        @test assemble_mean(gpm, [0.0, 0.5, 1.0]) == [2.0, 2.5, 3.0]
    end

    @testset "sample_cholesky: jitter control + reproducibility" begin
        # BM Σ on a grid through t=0 has an all-zero first row (R(0,s)=0) → exactly
        # singular: cholesky throws at jitter = 0, and the nugget restores it.
        t_grid = range(0, 1; length = 64)
        gp = GaussianProcess(brownian_motion_kernel)
        Sigma = assemble_cov(gp, t_grid)

        @test_throws PosDefException sample_cholesky(Sigma, StableRNG(1); jitter = 0.0)

        # The nugget restores positive-definiteness → a full-length draw.
        x = sample_cholesky(Sigma, StableRNG(1); jitter = 1e-10)
        @test length(x) == length(t_grid)
        @test all(isfinite, x)

        # Reproducibility: same seed → identical draw; different seed → different.
        @test sample_cholesky(Sigma, StableRNG(42)) == sample_cholesky(Sigma, StableRNG(42))
        @test sample_cholesky(Sigma, StableRNG(42)) != sample_cholesky(Sigma, StableRNG(43))
    end

    @testset "sampler + empirical_cov reproduce Σ (statistical consistency)" begin
        # The Unit-0 headline claim, as a fast CI guard: draws from sample_cholesky,
        # estimated by empirical_cov, must reproduce the assembled Σ. Loose tolerance
        # (5% Frobenius, relative) so it is robust to MC noise at a fixed seed, yet
        # still catches sampler-wiring bugs (.L↔.U, transpose, bad centering): a .U
        # sampler lands at ~93% error here, a correct one at ~0.8%.
        gp     = GaussianProcess(brownian_motion_kernel)
        t_grid = range(0, 1; length = 16)
        Sigma  = Matrix(assemble_cov(gp, t_grid))
        rng    = StableRNG(7)
        N      = 4000
        paths  = reduce(hcat, (sample_cholesky(Sigma, rng) for _ in 1:N))
        rel_err = norm(empirical_cov(paths) .- Sigma) / norm(Sigma)
        @test rel_err < 0.05
    end

    @testset "exponential_kernel through the GP pipeline" begin
        # Smoke test through the whole pipeline: the shipped OU kernel assembles to a
        # genuinely PD Σ and is sampleable (brownian_motion is the only kernel otherwise
        # taken end-to-end; the second shipped kernel deserves the same). isposdef also
        # documents, in test form, the phase-4 claim that this kernel is well-conditioned
        # (hence NOT the singular negative control).
        gp = GaussianProcess(exponential_kernel)
        t  = range(0, 5; length = 20)
        Σ  = assemble_cov(gp, t)
        @test isposdef(Matrix(Σ))                       # well-conditioned, unlike BM
        x  = sample_cholesky(Σ, StableRNG(1))
        @test length(x) == 20 && all(isfinite, x)
    end

    @testset "Spectral (Bochner): normalization + Lorentzian + round-trip" begin
        # Deterministic throughout: r is a fixed sampled covariance sequence, no RNG.
        D = 1.0; alpha = 1.0; dt = 0.05; n = 400          # n*dt = 20 ≈ 20/alpha: well-resolved
        r = [exponential_kernel(0.0, k * dt; D = D, alpha = alpha) for k in 0:n-1]  # R(0)=D/alpha=1
        omega, Shat = bochner_forward(r, dt)

        # (1) NORMALIZATION GATE (~3% check, NOT 1e-10): int Shat dOmega -> R(0). This carries
        #     truncation (r cut at n*dt), trapezoid, and dropped-Nyquist error, so it lands near
        #     0.9749*R0 (pinned by dry-run), not machine zero -- deterministic but not exact.
        R0 = D / alpha
        @test isapprox(spectral_variance(omega, Shat), R0; rtol = 3e-2)

        # (2) DROPPED-1/2pi NEGATIVE CONTROL (the 2pi lesson, as a passing test):
        #     without the 1/2pi the integral lands on 2*pi*R(0), NOT R(0). This illustrates the
        #     consequence via spectral_variance's linearity (Shat scaled 2pi post-hoc is exactly
        #     what a missing dt/(2pi) in bochner_forward would produce) -- it is NOT what actually
        #     catches a real regression in that line: gate (1) above already fails hard (~2pi off,
        #     rtol=3e-2) if bochner_forward's own dt/(2pi) factor is ever dropped.
        @test isapprox(spectral_variance(omega, 2pi .* Shat), 2pi * R0; rtol = 3e-2)
        @test !isapprox(spectral_variance(omega, 2pi .* Shat), R0; rtol = 0.2)   # provably NOT 1

        # (3) LORENTZIAN SHAPE MATCH at a few interior omega within the resolved band.
        #     CONVENTION (F1): Shat is ONE-SIDED (interior bins folded/doubled), so at interior
        #     omega it approximates the ONE-SIDED density 2*S(omega) -- compare to 2*S_analytic,
        #     NOT S_analytic (comparing to the two-sided density is a clean factor-of-2 bug).
        #     Realized error is <=0.11% (pinned by dry-run); rtol=1e-2 leaves headroom without
        #     being a vacuous 50x-loose placeholder.
        S_analytic(w) = D / (pi * (w^2 + alpha^2))            # TWO-SIDED density
        for w0 in (0.5, 1.0, 2.0)
            i = argmin(abs.(omega .- w0))
            @test isapprox(Shat[i], 2 * S_analytic(omega[i]); rtol = 1e-2)
        end

        # (3b) ONE-SIDED <-> TWO-SIDED CONTRACT (F4 -- the test whose absence let the shape-gate
        #      convention bug through; it locks the relationship the shape gate in (3) depends on).
        #      o1, S1 reuse the (omega, Shat) computed above rather than recalling bochner_forward
        #      (onesided defaults to true there too, so they are the identical call).
        o1, S1 = omega, Shat
        o2, S2 = bochner_forward(r, dt; onesided = false)
        keep = o2 .>= 0
        @test all(o1 .>= 0)                                   # independent of _onesided's own predicate
        @test o1 == o2[keep]                                  # same omega>=0 grid
        @test isapprox(S1[1], S2[keep][1]; atol = 1e-12)      # DC bin NOT doubled
        @test isapprox(S1[2:end], 2 .* S2[keep][2:end]; rtol = 1e-12)   # interior bins doubled
        #      both conventions integrate to R(0): one-sided over omega>=0 == two-sided over R.
        @test isapprox(spectral_variance(o1, S1), R0; rtol = 3e-2)
        @test isapprox(spectral_variance(o2, S2), R0; rtol = 3e-2)

        # (4) ROUND-TRIP: bochner_inverse inverts the *unsorted, two-sided* forward transform.
        #     bochner_inverse is PRIVATE (F5) -- reach it via the qualified module path. The CAVEAT
        #     below is exactly WHY it is not exported: bochner_forward SORTS by omega and folds
        #     one-sided by default; ifft needs the natural fft ordering, so build the unsorted
        #     two-sided spectrum here via the same private _raw_transform bochner_forward itself
        #     uses (single source of truth -- avoids hand-duplicating the even-extension/FFT/scale
        #     formula in the test) and pass dOmega = 2pi/(m*dt).
        rsym, S_unsorted, _ = StochasticProcesses.Spectral._raw_transform(r, dt)
        m = length(rsym)
        dOmega = 2pi / (m * dt)
        r_rt = StochasticProcesses.Spectral.bochner_inverse(S_unsorted, dOmega)
        @test isapprox(r_rt[1:n], r; atol = 1e-10)

        # (5) HAND-COMPUTED spectral_variance on a tiny NON-uniform grid (bites off-by-one /
        #     endpoint bugs the self-consistent test cannot): trapezoid of S=[1,3,2] on
        #     omega=[0,1,3] = 0.5*(1+3)*1 + 0.5*(3+2)*2 = 2 + 5 = 7.
        @test spectral_variance([0.0, 1.0, 3.0], [1.0, 3.0, 2.0]) == 7.0

        # (5b) RECTANGULAR spectral_power vs TRAPEZOIDAL spectral_variance on a UNIFORM grid with a
        #      NON-ZERO DC bin -> the DC-weighting difference BITES (M2). This is the CANONICAL home of
        #      the DC-halving lesson (relocated from the old 'coarse Welch grid ~9%' experiment framing,
        #      now stale since the experiment's Welch grid is fine). rectangular counts DC in FULL;
        #      trapezoid halves both DC and the top bin. Pins the two integrators as DISTINCT tools.
        ow = [0.0, 1.0, 2.0, 3.0]; Sw = [1.0, 3.0, 2.0, 4.0]
        @test spectral_power(ow, Sw)    == 10.0   # dOmega*sum = 1*(1+3+2+4); DC counted in FULL
        @test spectral_variance(ow, Sw) == 7.5    # trapezoid halves DC (0.5*1) and top bin -> 7.5

        # (6) _onesided folding is exercised implicitly by (1)-(3) (bochner_forward folds by
        #     default): interior bins doubled, DC not.

        # (7) DEGENERATE-INPUT GUARDS: before the guard, these crashed with the wrong exception
        #     (BoundsError / an opaque FFTW planning error) or silently returned 0.0 instead of
        #     signaling the real precondition violation.
        @test_throws ArgumentError bochner_forward(Float64[], dt)
        @test_throws ArgumentError spectral_variance([0.0], [5.0])
        @test_throws ArgumentError spectral_power([0.0], [5.0])
    end

    @testset "Spectral (periodogram): normalization + peak + raw identity" begin
        dt = 0.1

        # (1) PARSEVAL / NORMALIZATION on a DETERMINISTIC signal (no RNG): the one-sided spectral
        #     integral of a record returns its mean-square. Pins dt, 1/2pi, and the one-sided fold.
        #     INTEGRATOR (M2): spectral_power -- the RECTANGULAR Parseval sum, the right tool for a
        #     discrete periodogram (DC-robust; exact in expectation at any grid). Because spectral_power
        #     counts DC in full, the interior-cosine FIXTURE is now belt-and-suspenders / non-load-bearing
        #     (a DC-heavy vector would also pass) -- M2 retires the N1 fixture-fragility at the source.
        #     We keep the interior cosines anyway: orthogonal cosines give a clean Sum a_k^2/2 target.
        L = 256; t = dt .* (0:L-1)
        ks = (5, 11, 23); as = (1.0, 0.5, 0.3)                 # distinct interior DFT bins
        x  = sum(a .* cos.((2pi * k / (L * dt)) .* t) for (k, a) in zip(ks, as))
        omega, Shat = welch_psd(x, dt; nseg = 1, window = :none)
        ms = sum(a^2 for a in as) / 2                          # mean-square of orthogonal cosines
        # Bin-aligned orthogonal cosines with no leakage -> an EXACT DFT identity (dry-run measures
        # relative error ~1e-15, i.e. machine precision), so pin a tight rtol: a real ~1% normalization
        # bug (wrong doubling factor, off-by-one in nused) must fail this, not hide under 5e-2 slack.
        @test isapprox(spectral_power(omega, Shat), ms; rtol = 1e-9)

        # (2) PEAK LOCATION: Welch of a pure cosine places its mass at omega0. Choose omega0 ON
        #     a DFT bin (integer cycles) so argmax is unambiguous -- catches an angular-vs-ordinary
        #     (2pi) error in the omega axis. L=256, k=8 -> f0=k/(L*dt), omega0=2pi*f0.
        L = 256; k = 8; f0 = k / (L * dt); omega0 = 2pi * f0
        t = dt .* (0:L-1)
        xc = cos.(omega0 .* t)
        oc, Sc = welch_psd(xc, dt; nseg = 1, window = :none)
        @test isapprox(oc[argmax(Sc)], omega0; atol = (oc[2] - oc[1]))     # within one bin

        # (3) STRUCTURAL IDENTITY: raw_periodogram IS the single-segment unwindowed Welch case.
        or, Sr = raw_periodogram(x, dt)
        ow, Sw = welch_psd(x, dt; nseg = 1, window = :none)
        @test or == ow
        @test Sr == Sw

        # (4) HANN WINDOW-POWER (F2 -- guards U = sum(win.^2), NOT L; N4 -- BIPOLAR, Phase 2's in-CI
        #     negative control). Welch WITH the Hann window integrates to the window-WEIGHTED mean-square
        #     wms = sum(win.^2 .* x.^2)/sum(win.^2). INTEGRATOR (M2): spectral_power (rectangular Parseval)
        #     -- the interior-cosine fixture is now belt-and-suspenders (a constant/DC spike would also pass,
        #     since spectral_power counts DC in full; the DC-halving lesson lives in the Phase-1 paired test).
        #     A U->L bug divides by L not U, i.e. rescales Shat by U/L ~ 0.37, landing at ~0.19 = 0.37*wms --
        #     the control fires regardless of integrator.
        Lh = 256; th = dt .* (0:Lh-1); kh = 16                 # interior bin, well off DC and Nyquist
        xh  = cos.((2pi * kh / (Lh * dt)) .* th)
        win = [0.5 - 0.5 * cos(2pi * j / (Lh - 1)) for j in 0:Lh-1]   # same Hann as welch_psd
        U   = sum(abs2, win)                                   # window power
        wms = sum(win.^2 .* xh.^2) / U                         # window-weighted mean-square (exact target)
        oh, Sh = welch_psd(xh, dt; nseg = 1, window = :hann)
        # Same exact-DFT-identity argument as test (1) above: dry-run relative error ~1e-15, so 1e-9
        # leaves ample headroom while still catching a real ~1% U-normalization bug.
        @test isapprox(spectral_power(oh, Sh), wms; rtol = 1e-9)              # U correct (rectangular Parseval)
        @test !isapprox(spectral_power(oh, (U / Lh) .* Sh), wms; rtol = 0.2)  # U->L variant provably wrong

        # (5) MULTI-SEGMENT AVERAGING (N2 -- welch's reason to exist, previously UNtested in CI: every
        #     other test here uses nseg=1). With k IDENTICAL segments the average collapses to a single
        #     segment, so Welch(nseg=k) MUST byte-equal the raw periodogram of one copy. Integrator-free
        #     -- it bites the hop / nused / acc averaging loop directly.
        seg = Float64[1, -2, 3, 0.5, -1.5, 2, -0.5, 1, 0.25]        # any fixed real vector
        om, Sm   = welch_psd(vcat(seg, seg, seg, seg), dt; nseg = 4, window = :none)
        or1, Sr1 = raw_periodogram(seg, dt)
        @test om == or1 && Sm == Sr1
        bad = copy(seg); bad[1] += 1.0                              # perturb ONE of the four segments
        _, Smb = welch_psd(vcat(seg, seg, seg, bad), dt; nseg = 4, window = :none)
        @test Smb != Sr1                                            # the average must change -> the loop bites

        # OVERLAP PATH is LOAD-BEARING now (the experiment runs 50% Hann overlap, M1) -> assert it
        # NUMERICALLY, not just finiteness. welch's normalization + one-sided fold are LINEAR, so
        # welch(overlapping segs) == mean of the raw periodograms of those SAME segments -- an
        # independent recompute that bites the hop / nused / acc loop (hand-counting nused too).
        xo = Float64[2, -1, 0.5, 3, -2, 1, 0.25, -0.5]             # length 8
        oo, So = welch_psd(xo, dt; nseg = 2, noverlap = 2, window = :none)   # L=4, hop=2 -> 3 segments
        starts = 1:2:(length(xo) - 4 + 1)                          # 1,3,5  (hand-count of nused)
        @test length(starts) == 3
        raws  = [raw_periodogram(xo[s:s+3], dt) for s in starts]
        Smean = sum(r[2] for r in raws) ./ length(starts)
        @test oo == raws[1][1]                                     # same omega grid
        @test isapprox(So, Smean; rtol = 1e-12)                    # the hop/nused/acc average BITES

        # (6) DEGENERATE-INPUT GUARDS (review fix): before these guards existed, nseg<=0 threw an
        #     opaque DivideError/ArgumentError from ones(L)/div, noverlap>=L made hop=L-noverlap<=0
        #     and HUNG FOREVER (start never advances past the while condition), noverlap>L walked
        #     start negative and threw an opaque BoundsError, L==1 (nseg==length(x)) silently
        #     returned NaN (0/0 in the Hann window's k/(L-1)) instead of erroring, and an unknown
        #     window symbol (a typo, or :hamming/:blackman) silently fell through to rectangular.
        @test_throws ArgumentError welch_psd(x, dt; nseg = 0)
        @test_throws ArgumentError welch_psd(x, dt; nseg = -1)
        @test_throws ArgumentError welch_psd(ones(4), dt; nseg = 4)                # L == 1
        @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 2, noverlap = 4)   # hop == 0 (was: hangs)
        @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 2, noverlap = 6)   # hop < 0 (was: BoundsError)
        @test_throws ArgumentError welch_psd(ones(8), dt; nseg = 1, noverlap = -1)  # noverlap < 0
        @test_throws ArgumentError welch_psd(x, dt; nseg = 1, window = :hamming)    # unknown window
    end

    @testset "public surface" begin
        # Regression guard on the export surface: catches a dropped `export` or a
        # re-export block that falls out of sync with a submodule.
        for f in (:brownian_motion_kernel, :exponential_kernel, :GaussianProcess,
                  :assemble_cov, :assemble_mean, :empirical_cov, :sample_cholesky,
                  :bochner_forward, :spectral_variance, :spectral_power,
                  :welch_psd, :raw_periodogram)
            @test isdefined(StochasticProcesses, f)
        end
    end
end

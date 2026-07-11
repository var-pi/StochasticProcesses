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

    @testset "public surface" begin
        # Regression guard on the export surface: catches a dropped `export` or a
        # re-export block that falls out of sync with a submodule.
        for f in (:brownian_motion_kernel, :exponential_kernel, :GaussianProcess,
                  :assemble_cov, :assemble_mean, :empirical_cov, :sample_cholesky)
            @test isdefined(StochasticProcesses, f)
        end
    end
end

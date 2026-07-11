using Test, StochasticProcesses, LinearAlgebra

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

        # --- assemble_cov honours its Symmetric *type* contract (issymmetric alone
        # can't see a dropped wrapper: BM Σ is exactly symmetric as a plain Matrix too).
        @test Sigma isa Symmetric
        # A closed-form Σ entry: catches an assembly transpose/index bug that symmetry cannot.
        @test Sigma[2, 3] == min(t_grid[2], t_grid[3])
        # Non-zero mean AND the keyword constructor are exercised (not only the zero default).
        gpm = GaussianProcess(brownian_motion_kernel; meanfn = t -> 2.0 + t)
        @test assemble_mean(gpm, [0.0, 0.5, 1.0]) == [2.0, 2.5, 3.0]
    end
end

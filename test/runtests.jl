using Test, StochasticProcesses

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
end

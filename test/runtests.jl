using Test, StochasticProcesses

@testset "StochasticProcesses" begin
    @testset "package loads" begin
        @test isdefined(Main, :StochasticProcesses)
    end
end

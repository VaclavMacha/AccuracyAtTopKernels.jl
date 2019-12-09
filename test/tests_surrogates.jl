function test_hinge()
    ϑ = rand()
    l = Hinge(ϑ)
    @testset "hinge loss" begin
        @test typeof(l) <: AbstractSurrogate
        @test l.ϑ == ϑ
        @test l.value.([-1/ϑ - 1, 0, -1/ϑ + 1])    ≈ [0, 1, ϑ]
        @test l.gradient.([-1/ϑ - 1, 0, -1/ϑ + 1]) ≈ [0, ϑ, ϑ]
    end
end


function test_quadratic()
    ϑ = rand()
    l = Quadratic(ϑ)
    @testset "quadratic loss" begin
        @test typeof(l) <: AbstractSurrogate
        @test l.ϑ == ϑ
        @test l.value.([-1/ϑ - 1, 0, -1/ϑ + 1])    ≈ [0, 1, ϑ^2]
        @test l.gradient.([-1/ϑ - 1, 0, -1/ϑ + 1]) ≈ [0, 2*ϑ, 2*ϑ^2]
    end
end
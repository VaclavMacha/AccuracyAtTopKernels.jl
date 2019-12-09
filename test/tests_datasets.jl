function test_primal()
    n    = 100
    dim  = 2
    X    = rand(n, dim)
    y    = rand(n) .>= 0.75
    data = Primal(X, y)
    @testset "primal dataset" begin
        @test typeof(data) <: AbstractData
        @test data.dim  == dim
        @test data.n    == n
        @test data.npos == sum(y)
        @test data.nneg == sum(.~y)
        @test data.n    == data.npos + data.nneg 
        @test length(data.pos) == data.npos
        @test length(data.neg) == data.nneg
    end
end


function test_dual()
    n    = 100
    nα   = 50
    nβ   = 100
    K    = rand(nα + nβ, nα + nβ)
    data = Dual(K, n, nα)
    @testset "dual dataset" begin
        @test typeof(data) <: AbstractData
        @test data.n  == n
        @test data.nα == nα
        @test data.nβ == nβ
        @test size(data.K, 1)   == data.nα + data.nβ
        @test size(data.K, 2)   == data.nα + data.nβ
        @test length(data.indα) == data.nα
        @test length(data.indβ) == data.nβ
    end
end
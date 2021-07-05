function test_projections()
    Dists = [Uniform(0,1), Normal(0,1)]
    C1s   = [0.001, 0.1, 1, 10, 100]
    C2s   = [0.001, 0.1, 1, 10, 100]
    Ks    = 1:10
    δ0s   = vcat(0:0.1:1, 10, 100)
    
    n = rand(100:200)
    m = rand(50:150)

    @testset "projections" begin
        @testset "α0 ∼ $(typeof(distα).name), β0 ∼ $(typeof(distβ).name)" for (distα, distβ) in Iterators.product(Dists, Dists)
            α0 = rand(distα, n)
            β0 = rand(distα, m)

            @testset "δ0 = $δ0, C1 = $C1, C2 = $C2" for (δ0, C1, C2) in Iterators.product(δ0s, C1s, C2s)
                test_projection(α0, β0, δ0, C1, C2)
            end
            @testset "δ0 = $δ0" for δ0 in δ0s
                test_projection(α0, β0, δ0)
            end
            @testset "C = $C, K = $K" for (C, K) in Iterators.product(C1s, Ks)
                test_projection(α0, β0, C, K)
            end
            @testset "K = $K" for K in Ks
                test_projection(α0, β0, K)
            end
        end
    end
end


function objective(α::AbstractVector, β::AbstractVector, δ::Real, α0::AbstractVector, β0::AbstractVector, δ0::Real)
    sum(abs2, α - α0)/2 + sum(abs2, β - β0)/2 + sum(abs2, δ - δ0)/2
end


function objective(α::AbstractVector, β::AbstractVector, α0::AbstractVector, β0::AbstractVector)
    sum(abs2, α - α0)/2 + sum(abs2, β - β0)/2
end


function test_projection(α0::AbstractVector, β0::AbstractVector, δ0::Real, C1::Real, C2::Real; atol::Real = 1e-5)
    α1, β1, δ1 = AccuracyAtTopKernels.projection(α0, β0, δ0, C1, C2)
    α2, β2, δ2 = AccuracyAtTopKernels.projection_exact(α0, β0, δ0, C1, C2)

    @testset "patmat hinge" begin
        @testset "feasibility" begin
            @test abs(sum(α1) - sum(β1)) <= atol 
            @test minimum(α1) >= - atol
            @test maximum(α1) <= C1 + atol
            @test minimum(β1) >= - atol
            @test maximum(β1) <= C2*δ1 + atol
        end
        @testset "optimality" begin
            @test objective(α1, β1, δ1, α0, β0, δ0) <= objective(α2, β2, δ2, α0, β0, δ0) + atol
        end
    end
end


function test_projection(α0::AbstractVector, β0::AbstractVector, δ0::Real; atol::Real = 1e-5)
    α1, β1, δ1 = AccuracyAtTopKernels.projection(α0, β0, δ0)
    α2, β2, δ2 = AccuracyAtTopKernels.projection_exact(α0, β0, δ0)

    @testset "patmat quadratic" begin
        @testset "feasibility" begin
            @test abs(sum(α1) - sum(β1)) <= atol
            @test minimum(α1) >= - atol
            @test minimum(β1) >= - atol
            @test minimum(δ1) >= - atol
        end
        @testset "optimality" begin
            @test objective(α1, β1, δ1, α0, β0, δ0) <= objective(α2, β2, δ2, α0, β0, δ0) + atol
        end
    end
end


function test_projection(α0::AbstractVector, β0::AbstractVector, C::Real, K::Integer; atol::Real = 1e-5)
    α1, β1 = AccuracyAtTopKernels.projection(α0, β0, C, K)
    α2, β2 = AccuracyAtTopKernels.projection_exact(α0, β0, C, K)

    @testset "toppushk hinge" begin
        @testset "feasibility" begin
            @test abs(sum(α1) - sum(β1)) <= atol
            @test minimum(α1) >= - atol
            @test maximum(α1) <= C + atol
            @test minimum(β1) >= - atol
            @test maximum(β1) <= sum(α1)/K + atol
        end
        @testset "optimality" begin
            @test objective(α1, β1, α0, β0) <= objective(α2, β2, α0, β0) + atol
        end
    end
end


function test_projection(α0::AbstractVector, β0::AbstractVector, K::Integer; atol::Real = 1e-5)
    α1, β1 = AccuracyAtTopKernels.projection(α0, β0, K)
    α2, β2 = AccuracyAtTopKernels.projection_exact(α0, β0, K)

    @testset "toppushk quadratic" begin
        @testset "feasibility" begin
            @test abs(sum(α1) - sum(β1)) <= atol
            @test minimum(α1) >= - atol
            @test minimum(β1) >= - atol
            @test maximum(β1) <= sum(α1)/K + atol
        end
        @testset "optimality" begin
            @test objective(α1, β1, α0, β0) <= objective(α2, β2, α0, β0) + atol
        end
    end
end
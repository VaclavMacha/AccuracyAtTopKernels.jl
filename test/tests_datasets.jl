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
        @test length(data.ind_pos) == data.npos
        @test length(data.ind_neg) == data.nneg
    end
end


function test_dual()
    models = [PatMat(0.5), TopPushK(5), TopPush()]

    @testset "$(nameof(typeof(model)))" for model in models
        test_dual(model)
    end
end

function test_dual(model::AbstractModel)
    n     = 100
    dim   = 2
    Xtrain = rand(n, dim)
    ytrain = rand(n) .>= 0.75
    
    Xvalid = rand(3*n, dim)
    yvalid = rand(3*n) .>= 0.75

    Xtest  = rand(2*n, dim)

    data_train = Dual(model, Xtrain, ytrain)
    data_valid = Dual(model, Xtrain, ytrain, Xvalid, yvalid)
    data_test  = Dual(model, Xtrain, ytrain, Xtest)

    @testset "dual train dataset" begin
        @test typeof(data_train) <: Dual{<:DTrain}
        @test issymmetric(data_train.K)
        test_dual(data_train)
    end

    @testset "dual validation dataset" begin
        @test typeof(data_valid) <: Dual{<:DValidation}
        @test size(data_valid.K, 2) == data_valid.n
        test_dual(data_valid)
    end

    @testset "dual test dataset" begin
        @test typeof(data_test) <: Dual{<:DTest}
        test_dual(data_test)
        @test size(data_test.K, 2) == data_test.n
    end
end


function test_dual(data::Dual{<:Union{DTrain, DValidation}})
    @test length(data.ind_α) == data.nα
    @test length(data.ind_β) == data.nβ
    @test size(data.K, 1)    == data.nα + data.nβ
    @test length(data.type.ind_pos) == data.type.npos
    @test length(data.type.ind_neg) == data.type.nneg
end


function test_dual(data::Dual{<:DTest})
    @test length(data.ind_α) == data.nα
    @test length(data.ind_β) == data.nβ
    @test size(data.K, 1)    == data.nα + data.nβ
end
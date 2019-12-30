function test_kernels()

    N   = rand(50:200)
    M   = rand(50:200)
    dim = rand(5:10)
    Xtrain = rand(N, dim)
    ytrain = rand(N) .>= 0.75
    Xtest  = rand(M, dim)
    ytest = rand(N) .>= 0.75

    model1 = PatMat(Hinge(1), Hinge(1), 0.9, 1.1)
    model2 = TopPushK(Hinge(1), 5, 1.1)
    model3 = TopPush(Hinge(1), 1.1)

    kernels = [LinearKernel(),
               SquaredExponentialKernel(),
               RationalQuadraticKernel()]

    @testset "PatMat with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_kernelmatrix(model1, kernel, Xtrain, ytrain, Xtest, ytest)
    end

    @testset "TopPushK with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_kernelmatrix(model2, kernel, Xtrain, ytrain, Xtest, ytest)
    end

    @testset "TopPush with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_kernelmatrix(model3, kernel, Xtrain, ytrain, Xtest, ytest)
    end
end


function getmatrix(model::PatMat, Xtrain, ytrain, kernel)
    pos = findall(ytrain)
    nα  = length(pos)
    K   = MLKernels.kernelmatrix(Val(:row), kernel, vcat(Xtrain[pos,:], Xtrain))
    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    return K
end


function getmatrix(model::PatMat, Xtrain, ytrain, Xtest, ytest, kernel)
    return getmatrix(model, Xtrain, ytrain, Xtest, kernel)
end


function getmatrix(model::PatMat, Xtrain, ytrain, Xtest, kernel)
    pos = findall(ytrain)
    nα  = length(pos)
    K   = MLKernels.kernelmatrix(Val(:row), kernel, vcat(Xtrain[pos,:], Xtrain), Xtest)
    K[nα+1:end, :] .*= -1
    return K
end

function getmatrix(model::AbstractTopPushK, Xtrain, ytrain, kernel)
    pos = findall(ytrain)
    neg = findall(.~ytrain)
    nα  = length(pos)
    K   = MLKernels.kernelmatrix(Val(:row), kernel, vcat(Xtrain[pos,:], Xtrain[neg,:]))
    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    return K
end


function getmatrix(model::AbstractTopPushK, Xtrain, ytrain, Xtest, ytest, kernel)
    return getmatrix(model, Xtrain, ytrain, Xtest, kernel)
end


function getmatrix(model::AbstractTopPushK, Xtrain, ytrain, Xtest, kernel)
    pos = findall(ytrain)
    neg = findall(.~ytrain)
    nα  = length(pos)
    K   = MLKernels.kernelmatrix(kernel, vcat(Xtrain[pos,:], Xtrain[neg,:]), Xtest)
    K[nα+1:end, :] .*= -1
    return K
end


function test_kernelmatrix(model, kernel, Xtrain, ytrain, Xtest, ytest; atol::Real = 1e-10)
    
    ClassificationOnTop.save_kernelmatrix(model, "train.bin", Xtrain, ytrain; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "valid.bin", Xtrain, ytrain, Xtest, ytest; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "test.bin", Xtrain, ytrain, Xtest; kernel = kernel, T = Float64)

    K1  = getmatrix(model, Xtrain, ytrain, kernel)
    K2, = ClassificationOnTop.kernelmatrix(model, Xtrain, ytrain; kernel = kernel)
    t3, io3, out3 = ClassificationOnTop.load_kernelmatrix("train.bin"; T = Float64)
    K3 = out3[1]

    K4  = getmatrix(model, Xtrain, ytrain, Xtest, ytest, kernel)
    K5, = ClassificationOnTop.kernelmatrix(model, Xtrain, ytrain, Xtest, ytest; kernel = kernel)
    t6, io6, out6 = ClassificationOnTop.load_kernelmatrix("valid.bin"; T = Float64)
    K6 = out6[1]

    K7  = getmatrix(model, Xtrain, ytrain, Xtest, kernel)
    K8, = ClassificationOnTop.kernelmatrix(model, Xtrain, ytrain, Xtest; kernel = kernel)
    t9, io9, out9 = ClassificationOnTop.load_kernelmatrix("test.bin"; T = Float64)
    K9 = out9[1]

    @testset "train kernel matrix" begin
        @test K1 ≈ K2 atol = atol
        @test K1 ≈ K3 atol = atol
    end

    @testset "validation kernel matrix" begin
        @test K4 ≈ K5 atol = atol
        @test K4 ≈ K6 atol = atol
    end

    @testset "test kernel matrix" begin
        @test K7 ≈ K8 atol = atol
        @test K7 ≈ K9 atol = atol
    end

    close(io3)
    close(io6)
    close(io9)
    rm("train.bin")
    rm("valid.bin")
    rm("test.bin")
end
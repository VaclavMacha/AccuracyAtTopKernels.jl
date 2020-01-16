function test_scores()

    N   = rand(50:200)
    M   = rand(50:200)
    dim = rand(5:10)
    Xtrain = rand(N, dim)
    ytrain = rand(N) .>= 0.75
    Xtest  = rand(M, dim)
    ytest = rand(N) .>= 0.75

    kernels = [LinearKernel(),
               SquaredExponentialKernel(),
               RationalQuadraticKernel()]

    @testset "PatMat with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_scores(PatMat(0.9), kernel, Xtrain, ytrain, Xtest, ytest)
    end

    @testset "TopPushK with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_scores(TopPushK(5), kernel, Xtrain, ytrain, Xtest, ytest)
    end

    @testset "TopPush with $(typeof(kernel).name) kernel" for kernel in kernels 
        test_scores(TopPush(), kernel, Xtrain, ytrain, Xtest, ytest)
    end
end


function test_scores(model, kernel, Xtrain, ytrain, Xtest, ytest; atol::Real = 1e-4)
    
    ClassificationOnTop.save_kernelmatrix(model, "train1.bin", Xtrain, ytrain; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "train2.bin", Xtrain, ytrain, Xtrain, ytrain; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "train3.bin", Xtrain, ytrain, Xtrain; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "train.bin", Xtrain, ytrain; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "valid.bin", Xtrain, ytrain, Xtest, ytest; kernel = kernel, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "test.bin", Xtrain, ytrain, Xtest; kernel = kernel, T = Float64)

    data = Dual(model, Xtrain, ytrain; kernel = kernel)
    α, β = rand(data.nα), rand(data.nβ)

    s1 = scores(model, Xtrain, ytrain, α, β; kernel = kernel)
    s2 = scores(model, Xtrain, ytrain, Xtrain, α, β; kernel = kernel)
    s3 = scores(model, Xtrain, ytrain, Xtrain, ytrain, α, β; kernel = kernel)
    s4 = scores(model, Xtrain, ytrain, Xtest, ytest, α, β; kernel = kernel)
    s5 = scores(model, Xtrain, ytrain, Xtest, α, β; kernel = kernel)

    s1m = scores(model, "train1.bin", α, β; T = Float64)
    s2m = scores(model, "train2.bin", α, β; T = Float64)
    s3m = scores(model, "train3.bin", α, β; T = Float64)
    s4m = scores(model, "valid.bin", α, β; T = Float64)
    s5m = scores(model, "test.bin", α, β; T = Float64)

    @test maximum(abs.(s1 - s2)) <= atol
    @test maximum(abs.(s1 - s3)) <= atol
    @test maximum(abs.(s4 - s5)) <= atol
    @test s1 ≈ s1m atol = atol
    @test s2 ≈ s2m atol = atol
    @test s3 ≈ s3m atol = atol
    @test s4 ≈ s4m atol = atol
    @test s5 ≈ s5m atol = atol

    rm("train.bin")
    rm("valid.bin")
    rm("test.bin")
end
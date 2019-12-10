function test_kernels()

    N   = rand(50:200)
    M   = rand(50:200)
    dim = rand(5:10)
    X   = rand(N, dim)
    Y   = rand(M, dim)
    y   = rand(N) .>= 0.75

    model1 = PatMat(Hinge(1), Hinge(1), 0.9, 1.1)
    model2 = TopPushK(Hinge(1), 5, 1.1)
    model3 = TopPush(Hinge(1), 1.1)

    kernels = [KernelFunctions.LinearKernel(),
               KernelFunctions.SqExponentialKernel(),
               KernelFunctions.RationalQuadraticKernel()]

    @testset "PatMat with $(typeof(kernel)) kernel" for kernel in kernels 
        test_kernelmatrix(model1, kernel, X, Y, y)
    end

    @testset "TopPushK with $(typeof(kernel)) kernel" for kernel in kernels 
        test_kernelmatrix(model2, kernel, X, Y, y)
    end

    @testset "PatMat with $(typeof(kernel)) kernel" for kernel in kernels 
        test_kernelmatrix(model3, kernel, X, Y, y)
    end
end


function getmatrix(model::PatMat, kernel, X::AbstractMatrix, y::BitArray{1}; ε::Real = 1e-10)
    pos = findall(y)
    nα  = length(pos)
    K   = KernelFunctions.kernelmatrix(kernel, vcat(X[pos,:], X); obsdim = 1)
    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    K[:, :] += I*ε
    return K
end


function getmatrix(model::PatMat, kernel, X::AbstractMatrix, y::BitArray{1}, Y::AbstractMatrix)
    pos = findall(y)
    nα  = length(pos)
    K   = KernelFunctions.kernelmatrix(kernel, vcat(X[pos,:], X), Y; obsdim = 1)
    K[nα+1:end, :] .*= -1
    return K
end


function getmatrix(model::AbstractTopPushK, kernel, X::AbstractMatrix, y::BitArray{1}; ε::Real = 1e-10)
    pos = findall(y)
    neg = findall(.~y)
    nα  = length(pos)
    K   = KernelFunctions.kernelmatrix(kernel, vcat(X[pos,:], X[neg,:]); obsdim = 1)
    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    K[:, :] += I*ε
    return K
end


function getmatrix(model::AbstractTopPushK, kernel, X::AbstractMatrix, y::BitArray{1}, Y::AbstractMatrix)
    pos = findall(y)
    neg = findall(.~y)
    nα  = length(pos)
    K   = KernelFunctions.kernelmatrix(kernel, vcat(X[pos,:], X[neg,:]), Y; obsdim = 1)
    K[nα+1:end, :] .*= -1
    return K
end


function test_kernelmatrix(model, kernel, X, Y, y; ε::Real = 1e-5, atol::Real = 1e-10)
    
    ClassificationOnTop.save_kernelmatrix(model, "X.bin", X, y; kernel = kernel, ε = ε, T = Float64)
    ClassificationOnTop.save_kernelmatrix(model, "XY.bin", X, y, Y; kernel = kernel, T = Float64)

    K1 = getmatrix(model, kernel, X, y; ε = ε)
    K2, = ClassificationOnTop.kernelmatrix(model, X, y; kernel = kernel, ε = ε)
    K3, n, nα, nβ, ioX = ClassificationOnTop.load_kernelmatrix("X.bin"; T = Float64)

    K4 = getmatrix(model, kernel, X, y, Y)
    K5, = ClassificationOnTop.kernelmatrix(model, X, y, Y; kernel = kernel)
    K6, n, nα, nβ, ioXY = ClassificationOnTop.load_kernelmatrix("XY.bin"; T = Float64)


    @testset "kernel matrix" begin
        @test K1 ≈ K2 atol = atol
        @test K4 ≈ K5 atol = atol
    end

    @testset "mmap kernel matrix" begin
        @test K1 ≈ K3 atol = atol
        @test K4 ≈ K6 atol = atol
    end

    close(ioX)
    rm("X.bin")
    close(ioXY)
    rm("XY.bin")
end
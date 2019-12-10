function prepare(model::PatMat, X::AbstractMatrix, y::BitArray{1})
   
    pos = findall(y)
    n   = length(y)
    nα  = length(pos)
    nβ  = n
    XX  = vcat(X[pos, :], X)

    return XX, n, nα, nβ
end


function prepare(model::AbstractTopPushK, X::AbstractMatrix, y::BitArray{1})
   
    pos = findall(y)
    neg = findall(.~y)
    n   = length(y)
    nα  = length(pos)
    nβ  = length(neg)
    XX  = vcat(X[pos, :], X[neg, :])

    return XX, n, nα, nβ
end


function kernelmatrix(model::AbstractModel,
                      Xtrain::AbstractArray,
                      ytrain::BitArray{1};
                      kernel::Kernel = LinearKernel(),
                      ε::Real = 1e-10)

    X, n, nα, nβ = prepare(model, Xtrain, ytrain)
    K = KernelFunctions.kernelmatrix(kernel, X; obsdim = 1)
    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    K[:, :] += I*ε
    return K, n, nα, nβ
end


function kernelmatrix(model::AbstractModel,
                      Xtrain::AbstractArray,
                      ytrain::BitArray{1},
                      Xtest::AbstractArray;
                      kernel::Kernel = LinearKernel())

    X, n, nα, nβ = prepare(model, Xtrain, ytrain)
    K = KernelFunctions.kernelmatrix(kernel, X, Xtest; obsdim = 1)
    K[nα+1:end, :] .*= -1
    return K, n, nα, nβ
end


function save_kernelmatrix(model::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           ytrain::BitArray{1};
                           kernel::Kernel = LinearKernel(),
                           ε::Real = 1e-10,
                           T::DataType = Float32)

    X, n, nα, nβ = prepare(model, Xtrain, ytrain)

    io = open(file, "w+");
    write(io, n)
    write(io, nα)
    write(io, nβ)
    write(io, nα + nβ)
    write(io, nα + nβ)
    K = Mmap.mmap(io, Matrix{T}, (nα + nβ, nα + nβ))
    fill_kernelmatrix!(K, kernel, X, X; obsdim = 1)

    K[1:nα, nα+1:end] .*= -1
    K[nα+1:end, 1:nα] .*= -1
    K[:, :] += I*ε
    
    Mmap.sync!(K)
    close(io)

    return 
end


function save_kernelmatrix(model::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           ytrain::BitArray{1},
                           Xtest::AbstractMatrix;
                           kernel::Kernel = LinearKernel(),
                           T::DataType = Float32)

    X, n, nα, nβ = prepare(model, Xtrain, ytrain)
    M, N = size(X,1), size(Xtest,1)

    io = open(file, "w+");
    write(io, n)
    write(io, nα)
    write(io, nβ)
    write(io, M)
    write(io, N)
    K = Mmap.mmap(io, Matrix{T}, (M, N))
    fill_kernelmatrix!(K, kernel, X, Xtest; obsdim = 1)

    K[nα+1:end, :] .*= -1
    
    Mmap.sync!(K)
    close(io)
end


function fill_kernelmatrix!(K::AbstractMatrix,
                            kernel::Kernel,
                            Xtrain::AbstractMatrix,
                            Xtest::AbstractMatrix;
                            max_chunk_size::Real = 1e8,
                            obsdim::Integer = 2)
   
    n    = size(Xtrain,1)
    Rows = Iterators.partition(1:n, floor(Int, max(max_chunk_size, n)/n))

    ProgressMeter.@showprogress "Kernel matrix calculation in progress: " for rows in Rows
        @views KernelFunctions.kernelmatrix!(K[rows, :], kernel, Xtrain[rows,:], Xtest; obsdim = obsdim)
    end
end


function load_kernelmatrix(file::AbstractString; T::DataType = Float32)
    io  = open(file, "r");
    n   = read(io, Int)
    nα  = read(io, Int)
    nβ  = read(io, Int)
    M   = read(io, Int)
    N   = read(io, Int)
    K   = Mmap.mmap(io, Matrix{T}, (M, N))
    return K, n, nα, nβ, io
end
function prepare(model::PatMat, X::AbstractMatrix, y::BitArray{1})
   
    pos = findall(y)
    n   = length(y)
    nα  = length(pos)
    nβ  = n
    XX  = vcat(X[pos, :], .- X)

    return XX, n, nα, nβ
end


function prepare(model::AbstractTopPushK, X::AbstractMatrix, y::BitArray{1})
   
    pos = findall(y)
    neg = findall(.~y)
    n   = length(y)
    nα  = length(pos)
    nβ  = length(neg)
    XX  = vcat(X[pos, :], .- X[neg, :])

    return XX, n, nα, nβ
end


function kernelmatrix(model::AbstractModel,
                      Xtrain::AbstractArray,
                      y::BitArray{1},
                      kernel::Kernel = LinearKernel();
                      ε::Real = 1e-10)

    X, n, nα, nβ = prepare(model, Xtrain, y)
    return KernelFunctions.kernelmatrix(kernel, X; obsdim = 1) + I*ε
end


function kernelmatrix(model::AbstractModel,
                      Xtrain::AbstractArray,
                      Xtest::AbstractArray,
                      y::BitArray{1},
                      kernel::Kernel = LinearKernel())

    X, n, nα, nβ = prepare(model, Xtrain, y)
    return KernelFunctions.kernelmatrix(kernel, X, Xtest; obsdim = 1)
end


function save_kernelmatrix(model::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           y::BitArray{1},
                           kernel::Kernel = LinearKernel();
                           kwargs...)

    X, n, nα, nβ = prepare(model, Xtrain, y)
    save_kernelmatrix(file, X, X, n, nα, nβ, kernel; kwargs...)
    return 
end


function save_kernelmatrix(model::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           Xtest::AbstractMatrix,
                           y::BitArray{1},
                           kernel::Kernel = LinearKernel();
                           kwargs...)

    X, n, nα, nβ = prepare(model, Xtrain, y)
    save_kernelmatrix(file, X, Xtest, n, nα, nβ, kernel; kwargs...)
    return 
end


function save_kernelmatrix(file::AbstractString,
                           Xtrain::AbstractMatrix,
                           Xtest::AbstractMatrix,
                           n::Integer,
                           nα::Integer,
                           nβ::Integer = size(Xtrain,1) - nα, 
                           kernel::Kernel = LinearKernel();
                           ε::Real = 1e-10,
                           T::DataType = Float32)

    M, N = size(Xtrain,1), size(Xtest,1)

    io = open(file, "w+");
    write(io, n)
    write(io, nα)
    write(io, nβ)
    write(io, M)
    write(io, N)
    K = Mmap.mmap(io, Matrix{T}, (M, N))
    fill_kernelmatrix!(K, kernel, Xtrain, Xtest; obsdim = 1)
    M == N && ( K[:,:] += I*ε )
    
    Mmap.sync!(K)
    close(io)
    return 
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
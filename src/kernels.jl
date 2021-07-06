# -------------------------------------------------------------------------------
# Prepare function
# -------------------------------------------------------------------------------
function prepare(::PatMat, X::AbstractMatrix, y::AbstractVector{Bool})

    ind_pos  = findall(y)
    ind_neg  = findall(.~y)
    nα       = length(ind_pos)
    nβ       = length(y)
    inv_perm = nα .+ (1:nβ)

    XX  = vcat(X[ind_pos, :], X)

    return XX, nα, nβ, ind_pos, ind_neg, inv_perm
end


function prepare(
    ::Union{AbstractTopPushK, PatMatNP},
    X::AbstractMatrix,
    y::AbstractVector{Bool}
)

    ind_pos  = findall(y)
    ind_neg  = findall(.~y)
    nα       = length(ind_pos)
    nβ       = length(ind_neg)
    inv_perm = invperm(vcat(ind_pos, ind_neg))

    XX  = vcat(X[ind_pos, :], X[ind_neg, :])

    return XX, nα, nβ, ind_pos, ind_neg, inv_perm
end


# -------------------------------------------------------------------------------
# DTrain kernel matrix
# -------------------------------------------------------------------------------
function kernelmatrix(M::AbstractModel,
                      Xtrain::AbstractArray,
                      ytrain::AbstractVector{Bool};
                      kernel::Kernel = LinearKernel(1, 0))

    X, nα, nβ, ind_pos, ind_neg, inv_perm = prepare(M, Xtrain, ytrain)
    n = length(ytrain)

    K = MLKernels.kernelmatrix(Val(:row), kernel, X, true)
    K[1:nα, nα+1:end]   .*= -1
    K[(nα+1):end, 1:nα] .*= -1
    return K, n, nα, nβ, ind_pos, ind_neg, inv_perm
end


function save_kernelmatrix(M::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           ytrain::AbstractVector{Bool};
                           kernel::Kernel = LinearKernel(1, 0),
                           T::DataType    = Float32)

    X, nα, nβ, ind_pos, ind_neg, inv_perm = prepare(M, Xtrain, ytrain)

    n, npos, nneg = length(inv_perm), length(ind_pos), length(ind_neg)
    N             = nα + nβ

    # auxiliary
    io = open(file, "w+");
    write(io, 0)
    write(io, [nα, nβ, n, npos, nneg, N, N])
    write(io, ind_pos)
    write(io, ind_neg)
    write(io, inv_perm)

    # kernel matrix
    K = Mmap.mmap(io, Matrix{T}, (N, N))
    fill_kernelmatrix!(K, kernel, X, X)
    K[1:nα, nα+1:end]   .*= -1
    K[(nα+1):end, 1:nα] .*= -1
    Mmap.sync!(K)
    close(io)

    return
end


# -------------------------------------------------------------------------------
# DValidation kernel matrix
# -------------------------------------------------------------------------------
function kernelmatrix(M::AbstractModel,
                      Xtrain::AbstractArray,
                      ytrain::AbstractVector{Bool},
                      Xvalid::AbstractArray,
                      yvalid::AbstractVector{Bool};
                      kernel::Kernel = LinearKernel(1, 0))

    X, nα, nβ, = prepare(M, Xtrain, ytrain)
    n          = length(yvalid)
    ind_pos    = findall(yvalid)
    ind_neg    = findall(.~yvalid)
    inv_perm   = 1:length(yvalid)

    K = MLKernels.kernelmatrix(Val(:row), kernel, X, Xvalid)
    K[(nα+1):end, :] .*= -1
    return K, n, nα, nβ, ind_pos, ind_neg, inv_perm
end


function save_kernelmatrix(M::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           ytrain::AbstractVector{Bool},
                           Xvalid::AbstractMatrix,
                           yvalid::AbstractVector{Bool};
                           kernel::Kernel = LinearKernel(1, 0),
                           T::DataType    = Float32)

    X, nα, nβ, = prepare(M, Xtrain, ytrain)
    ind_pos    = findall(yvalid)
    ind_neg    = findall(.~yvalid)
    inv_perm   = 1:length(yvalid)

    n, npos, nneg = length(inv_perm), length(ind_pos), length(ind_neg)
    M, N          = size(X,1), size(Xvalid,1)

    # auxiliary
    io = open(file, "w+");
    write(io, 1)
    write(io, [nα, nβ, n, npos, nneg, M, N])
    write(io, ind_pos)
    write(io, ind_neg)
    write(io, inv_perm)

    # kernel matrix
    K = Mmap.mmap(io, Matrix{T}, (M, N))
    fill_kernelmatrix!(K, kernel, X, Xvalid)
    K[nα+1:end, :] .*= -1
    Mmap.sync!(K)
    close(io)

    return
end


# -------------------------------------------------------------------------------
# DTest kernel matrix
# -------------------------------------------------------------------------------
function kernelmatrix(M::AbstractModel,
                      Xtrain::AbstractArray,
                      ytrain::AbstractVector{Bool},
                      Xtest::AbstractArray;
                      kernel::Kernel = LinearKernel(1, 0))

    X, nα, nβ, = prepare(M, Xtrain, ytrain)
    n = size(Xtest, 1)

    K = MLKernels.kernelmatrix(Val(:row), kernel, X, Xtest)
    K[(nα+1):end, :] .*= -1

    return K, n, nα, nβ
end


function save_kernelmatrix(M::AbstractModel,
                           file::AbstractString,
                           Xtrain::AbstractMatrix,
                           ytrain::AbstractVector{Bool},
                           Xtest::AbstractMatrix;
                           kernel::Kernel = LinearKernel(1, 0),
                           T::DataType = Float32)

    X, nα, nβ, = prepare(M, Xtrain, ytrain)
    n          = size(Xtest, 1)
    M, N       = size(X,1), size(Xtest,1)

    # auxiliary
    io = open(file, "w+");
    write(io, 2)
    write(io, [nα, nβ, n, M, N])

    # kernel matrix
    K = Mmap.mmap(io, Matrix{T}, (M, N))
    fill_kernelmatrix!(K, kernel, X, Xtest)
    K[nα+1:end, :] .*= -1
    Mmap.sync!(K)
    close(io)

    return
end


# -------------------------------------------------------------------------------
# Fill function for large matrices
# -------------------------------------------------------------------------------
function fill_kernelmatrix!(K::AbstractMatrix,
                            kernel::Kernel,
                            X::AbstractMatrix,
                            Y::AbstractMatrix;
                            max_chunk_size::Real = 1e6)

    n    = size(X,1)
    Rows = Iterators.partition(1:n, floor(Int, max(max_chunk_size, n)/n))

    ProgressMeter.@showprogress "Kernel matrix calculation in progress: " for rows in Rows
        K[rows, :] .= MLKernels.kernelmatrix(Val(:row), kernel, X[rows,:], Y)
    end
end


# -------------------------------------------------------------------------------
# Load function
# -------------------------------------------------------------------------------
function load_kernelmatrix(file::AbstractString; T::DataType = Float32)
    io   = open(file, "r");
    type = read(io, Int)

    type in [0,1,2] || @error "Unknown kernel matrix type $(type)"

    if type == 0 || type == 1
        nα, nβ, n, npos, nneg, M, N = [read(io, Int) for k in 1:7]

        ind_pos  = [read(io, Int) for k in 1:npos]
        ind_neg  = [read(io, Int) for k in 1:nneg]
        inv_perm = [read(io, Int) for k in 1:n]
        K        = Mmap.mmap(io, Matrix{T}, (M, N))

        return type, io, (K, n, nα, nβ, ind_pos, ind_neg, inv_perm)
    elseif type == 2
        nα, nβ, n, M, N = [read(io, Int) for k in 1:5]

        K = Mmap.mmap(io, Matrix{T}, (M, N))
        return type, io, (K, n, nα, nβ)
    end
end

# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
struct Primal{I<:Integer, V1<:AbstractVector, V2<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    X::A
    y::V1
    ind_pos::V2
    ind_neg::V2

    dim::I
    n::I
    npos::I
    nneg::I
end


function Primal(X::A, y::V) where {A<:AbstractMatrix, V<:AbstractVector}

    ybool   = Bool.(y)
    ind_pos = findall(ybool)
    ind_neg = findall(.~ybool)

    dim  = size(X,2)
    n    = length(y)
    npos = length(ind_pos)
    nneg = length(ind_neg)

    return Primal(X, y, ind_pos, ind_neg, dim, n, npos, nneg)
end


# -------------------------------------------------------------------------------
# Dual problem
# -------------------------------------------------------------------------------
abstract type DualType; end

struct Dual{T<:DualType, I<:Integer, V<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    type::T
    io::IO

    K::A
    ind_α::V
    ind_β::V
    nα::I
    nβ::I
    n::I

    function Dual(type::T, io::IO, K::A, nα::I, nβ::I, n::I) where {T<:DualType, I, A}
        ind_α = 1:nα
        ind_β = nα .+ (1:nβ)

        new{T, I, typeof(ind_α), A}(type, io, K, ind_α, ind_β, nα, nβ, n)
    end
end


# train data
struct DTrain{I<:Integer, V1<:AbstractVector, V2<:AbstractVector} <: DualType;
    ind_pos::V1
    ind_neg::V1
    inv_perm::V2
    n::I
    npos::I
    nneg::I

    function DTrain(ind_pos::V1, ind_neg::V1, inv_perm::V2) where {V1<:AbstractVector, V2<:AbstractVector} 
        n, npos, nneg = length(inv_perm), length(ind_pos), length(ind_neg)
        new{typeof(n), V1, V2}(ind_pos, ind_neg, inv_perm, n, npos, nneg)
    end
end


function Dual(model::AbstractModel,
              Xtrain::AbstractMatrix,
              ytrain::BitArray{1};
              kwargs...)
    
    K, n, nα, nβ, ind_pos, ind_neg, inv_perm = kernelmatrix(model, Xtrain, ytrain; kwargs...)

    type = DTrain(ind_pos, ind_neg, inv_perm)
    io   = IOBuffer()
    close(io)

    return Dual(type, io, K, nα, nβ, n)
end


# validation data
struct DValidation{I<:Integer, V1<:AbstractVector, V2<:AbstractVector} <: DualType;
    ind_pos::V1
    ind_neg::V1
    inv_perm::V2
    n::I
    npos::I
    nneg::I

    function DValidation(ind_pos::V1, ind_neg::V1, inv_perm::V2) where {I, V1, V2} 
        n, npos, nneg = length(inv_perm), length(ind_pos), length(ind_neg)
        new{typeof(n), V1, V2}(ind_pos, ind_neg, inv_perm, n, npos, nneg)
    end
end


function Dual(model::AbstractModel,
              Xtrain::AbstractMatrix,
              ytrain::BitArray{1},
              Xvalid::AbstractMatrix,
              yvalid::BitArray{1};
              kwargs...)
    
    K, n, nα, nβ, ind_pos, ind_neg, inv_perm = kernelmatrix(model, Xtrain, ytrain, Xvalid, yvalid; kwargs...)

    type = DValidation(ind_pos, ind_neg, inv_perm)
    io   = IOBuffer()
    close(io)

    return Dual(type, io, K, nα, nβ, n)
end


# test data
struct DTest{I<:Integer} <: DualType
    n::I
end


function Dual(model::AbstractModel,
              Xtrain::AbstractMatrix,
              ytrain::BitArray{1},
              Xtest::AbstractMatrix;
              kwargs...)
    
    K, n, nα, nβ = kernelmatrix(model, Xtrain, ytrain, Xtest; kwargs...)
 
    type = DTest(n)
    io   = IOBuffer()
    close(io)

    return Dual(type, io, K, nα, nβ, n)
end


# Dual dataset load function 
function Dual(file::AbstractString; kwargs...)

    type_int, io, out = load_kernelmatrix(file; kwargs...)

    if type_int == 0
        K, n, nα, nβ, ind_pos, ind_neg, inv_perm = out
        type = DTrain(ind_pos, ind_neg, inv_perm)
   
    elseif type_int == 1
        K, n, nα, nβ, ind_pos, ind_neg, inv_perm = out
        type = DValidation(ind_pos, ind_neg, inv_perm)

    elseif type_int == 2
        K, n, nα, nβ = out
        type = DTest(n)
    end

    return Dual(type, io, K, nα, nβ, n)
end
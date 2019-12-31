# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
@with_kw_noshow struct Primal{I<:Integer, B<:BitArray{1}, V<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    X::A
    y::B
    ind_pos::V = findall(y)
    ind_neg::V = findall(.~y)
    dim::I     = size(X,2)
    n::I       = length(y)
    npos::I    = length(ind_pos)
    nneg::I    = length(ind_neg)
end


Primal(X::AbstractMatrix, y::AbstractVector) = Primal(X = X, y = Bool.(y))


# -------------------------------------------------------------------------------
# Dual problem
# -------------------------------------------------------------------------------
abstract type DualType; end

@with_kw_noshow struct DTrain{I<:Integer, V<:AbstractVector, P<:AbstractVector} <: DualType
    ind_pos::V
    ind_neg::V
    inv_perm::P
    n::I    = length(inv_perm)
    npos::I = length(ind_pos)
    nneg::I = length(ind_neg)
end


DTrain(ipos, ineg, iperm) = DTrain(ind_pos = ipos, ind_neg = ineg, inv_perm = iperm)


@with_kw_noshow struct DValidation{I<:Integer, V<:AbstractVector, P<:AbstractVector} <: DualType
    ind_pos::V
    ind_neg::V
    inv_perm::P
    n::I    = length(inv_perm)
    npos::I = length(ind_pos)
    nneg::I = length(ind_neg)
end


DValidation(ipos, ineg, iperm) = DValidation(ind_pos = ipos, ind_neg = ineg, inv_perm = iperm)


struct DTest{I<:Integer} <: DualType
    n::I
end


# type Dual
@with_kw_noshow struct Dual{T<:DualType, I<:Integer, V<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    type::T
    io::IO

    K::A
    nα::I
    nβ::I
    n::I

    ind_α::V = 1:nα
    ind_β::V = nα .+ (1:nβ)
end

Dual(type::DualType, io::IO, K::AbstractMatrix, nα::Int, nβ::Int, n::Int) = 
    Dual(type = type, io = io, K = K, nα = nα, nβ = nβ, n = n)


# train data
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
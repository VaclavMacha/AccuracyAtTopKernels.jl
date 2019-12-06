# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
struct Primal{I<:Integer, V1<:AbstractVector, V2<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    X::A
    y::V1
    pos::V2
    neg::V2

    dim::I
    n::I
    npos::I
    nneg::I
end

function Primal(X::A, y::V) where {A<:AbstractMatrix, V<:AbstractVector}

    ybool = Bool.(y)
    pos = findall(ybool)
    neg = findall(.~ybool)

    dim  = size(X,2)
    n    = length(y)
    npos = length(pos)
    nneg = length(neg)

    return Primal(X, y, pos, neg, dim, n, npos, nneg)
end


function scores!(data::Primal, w::AbstractVector, s::AbstractVector)
    s .= data.X * w
end


# -------------------------------------------------------------------------------
# Dual problem
# -------------------------------------------------------------------------------
struct Dual{I<:Integer, V<:AbstractVector, A<:AbstractMatrix} <: AbstractData
    K::A
    indα::V
    indβ::V

    n::I
    nα::I
    nβ::I
    nαβ::I
end

function Dual(K::A, n::I, nα::I, nβ::I = size(K,1) - nα) where {A<:AbstractMatrix, I<:Integer} 
    indα = 1:nα 
    indβ = (nα + 1):(nα + nβ)

    return Dual(K, indα, indβ, n, nα, nβ, nα + nβ)
end


function scores!(data::Dual, α::AbstractVector, β::AbstractVector, s::AbstractVector)
    s .= data.K * vcat(α, β)
end

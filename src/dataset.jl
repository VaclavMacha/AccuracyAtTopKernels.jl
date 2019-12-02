# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
struct Primal{I<:Integer, V1<:AbstractVector, V2<:AbstractVector, A<:AbstractMatrix} <: Dataset
    X::A
    y::V1
    pos::V2
    neg::V2

    dim::I
    n::I
    npos::I
    nneg::I

    function Primal(X::A, y::V) where {A<:AbstractMatrix, V<:AbstractVector}

        ybool = Bool.(y)
        pos = findall(ybool)
        neg = findall(.~ybool)

        dim  = size(X,2)
        n    = length(y)
        npos = length(pos)
        nneg = length(neg)

        return new{typeof(n), V, typeof(pos), A}(X, y, pos, neg, dim, n, npos, nneg)
    end
end


function scores!(data::Primal, w::AbstractVector, s::AbstractVector)
    s .= data.X * w
end


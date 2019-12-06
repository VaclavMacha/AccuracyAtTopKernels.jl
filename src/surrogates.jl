struct Hinge{T<:Real, F1, F2, F3} <: AbstractSurrogate
    ϑ::T
    value::F1
    value_exact::F2
    gradient::F3
end


function Hinge(ϑ::Real) where {T<:Real}
    value(s::Real)    = max(0, 1 + ϑ*s)
    value_exact(s)    = Convex.max(0, 1 + ϑ*s)
    gradient(s::Real) = 1 + ϑ*s <= 0 ? zero(ϑ) : ϑ

    return Hinge(ϑ, value, value_exact, gradient)
end


struct Quadratic{T<:Real, F1, F2, F3} <: AbstractSurrogate
    ϑ::T
    value::F1
    value_exact::F2
    gradient::F3
end


function Quadratic(ϑ::T) where {T<:Real}
    value(s::Real)    = max(0, 1 + ϑ*s)^2
    value_exact(s)    = Convex.square(Convex.max(0, 1 + ϑ*s))
    gradient(s::Real) = 2*ϑ*max(0, 1 + ϑ*s)

    return Quadratic(ϑ, value, value_exact, gradient)
end
struct Exponential{T<:Real} <: Surrogate
    ϑ::T
    value::Function
    value_exact::Function
    gradient::Function

    function Exponential(ϑ::T) where {T<:Real}
        value(s::Real)    = exp(ϑ*s)
        value_exact(s)    = Convex.exp(1 + ϑ*s)
        gradient(s::Real) = ϑ*exp(ϑ*s)

        return new{T}(ϑ, value, value_exact, gradient)
    end
end


struct Hinge{T<:Real} <: Surrogate
    ϑ::T
    value::Function
    value_exact::Function
    gradient::Function

    function Hinge(ϑ::T) where {T<:Real}
        value(s::Real)    = max(0, 1 + ϑ*s)
        value_exact(s)    = Convex.max(0, 1 + ϑ*s)
        gradient(s::Real) = 1 + ϑ*s <= 0 ? zero(ϑ) : ϑ

        return new{T}(ϑ, value, value_exact, gradient)
    end
end


struct Quadratic{T<:Real} <: Surrogate
    ϑ::T
    value::Function
    value_exact::Function
    gradient::Function

    function Quadratic(ϑ::T) where {T<:Real}
        value(s::Real)    = max(0, 1 + ϑ*s)^2
        value_exact(s)    = Convex.square(Convex.max(0, 1 + ϑ*s))
        gradient(s::Real) = 2*ϑ*max(0, 1 + ϑ*s)

        return new{T}(ϑ, value, value_exact, gradient)
    end
end
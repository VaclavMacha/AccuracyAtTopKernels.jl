struct Hinge <: AbstractSurrogate
    ϑ::Real
    value::Function
    value_exact::Function
    gradient::Function

    function Hinge(ϑ::Real)
        value(s::Real)    = max(0, 1 + ϑ*s)
        value_exact(s)    = Convex.max(0, 1 + ϑ*s)
        gradient(s::Real) = 1 + ϑ*s <= 0 ? zero(ϑ) : ϑ

        return new(ϑ, value, value_exact, gradient)
    end
end


struct Quadratic <: AbstractSurrogate
    ϑ::Real
    value::Function
    value_exact::Function
    gradient::Function

    function Quadratic(ϑ::Real)
        value(s::Real)    = max(0, 1 + ϑ*s)^2
        value_exact(s)    = Convex.square(Convex.max(0, 1 + ϑ*s))
        gradient(s::Real) = 2*ϑ*max(0, 1 + ϑ*s)

        return new(ϑ, value, value_exact, gradient)
    end
end
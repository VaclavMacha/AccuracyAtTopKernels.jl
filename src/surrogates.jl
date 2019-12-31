show(io::IO, surr::T) where {T<:AbstractSurrogate} =
    print(io, "$(T.name)($(surr.ϑ))")


@with_kw_noshow struct Hinge{T<:Real} <: AbstractSurrogate
    ϑ::T = 1.0
    value::Function       = (s) -> max(0, 1 + ϑ*s)
    value_exact::Function = (s) -> Convex.max(0, 1 + ϑ*s)
    gradient::Function    = (s) -> 1 + ϑ*s <= 0 ? zero(ϑ) : ϑ
end


Hinge(ϑ) = Hinge(ϑ = ϑ)


@with_kw_noshow struct Quadratic{T<:Real} <: AbstractSurrogate
    ϑ::T = 1.0
    value::Function       = (s) -> max(0, 1 + ϑ*s)^2
    value_exact::Function = (s) -> Convex.square(Convex.max(0, 1 + ϑ*s))
    gradient::Function    = (s) -> 2*ϑ*max(0, 1 + ϑ*s)
end


Quadratic(ϑ) = Quadratic(ϑ = ϑ)
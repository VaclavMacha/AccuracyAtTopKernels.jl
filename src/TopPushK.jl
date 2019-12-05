struct TopPushK{S<:AbstractSurrogate} <: AbstractTopPushK{S}
    l::S
    K::Integer
    C::Real

    function TopPushK(l::S, K::Integer, C::Real) where {S<:AbstractSurrogate}

        @assert K >= 1 "The vaule of `K` must be greater or equal to 1."

        return new{S}(l, K, C)
    end
end


struct TopPush{S<:AbstractSurrogate} <: AbstractTopPushK{S}
    l::S
    C::Real

    function TopPush(l::S, C::Real) where {S<:AbstractSurrogate}
        return new{S}(l, C)
    end
end



function initialization(model::AbstractTopPushK, data::Primal, w0)
    w = zeros(eltype(data.X), data.dim)
    isempty(w0) || (w .= w0) 
    Δ = zero(w)
    s = data.X * w
    return w, Δ, s
end


function initialization(model::AbstractTopPushK, data::Dual, α0, β0)
    αβ   = zeros(eltype(data.K), data.n)
    α, β = @views αβ[data.indα], αβ[data.indβ]

    isempty(α0) || (α .= α0)
    isempty(β0) || (β .= β0)

    projection!(model, data, α, β)

    Δ  = zero(αβ)
    s  = data.K * vcat(α, β)
    return α, β, αβ, Δ, s
end


# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::TopPushK, data::Primal)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)
    z = Convex.Variable(data.nneg)

    objective = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t + Convex.sum(z)/model.K - Xpos*w,
                   z >= Xneg*w - t,
                   z >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver.optimizer)

    return vec(w.value), t.value + sum(z.value)/model.K
end


function optimize(solver::General, model::TopPush, data::Primal)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)

    objective = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t - Xpos*w,
                   t >= Convex.maximum(Xneg*w)]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver.optimizer)

    return vec(w.value), t.value
end


# Our solution
function objective(model::AbstractTopPushK, data::Primal, w, t, s)
    return w'*w/2 + model.C * sum(model.l.value.(t .- s[data.pos]))
end


function threshold(model::TopPushK, data::Primal, s)
   return mean(partialsort(s[data.neg], 1:model.K, rev = true))
end


function threshold(model::TopPush, data::Primal, s)
   return maximum(s[data.neg])
end


function gradient!(model::TopPushK, data::Primal, w, s, Δ)
    ind_t = partialsortperm(s[data.neg], 1:model.K, rev = true)
    t     = mean(s[data.neg[ind_t]])

    ∇l = model.l.gradient.(t .- s[data.pos])
    ∇t = vec(mean(data.X[data.neg[ind_t], :], dims = 1))

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.pos,:]'*∇l)
end


function gradient!(model::TopPush, data::Primal, w, s, Δ)
    t, ind_t = findmax(s[data.neg])

    ∇l = model.l.gradient.(t .- s[data.pos])
    ∇t = vec(data.X[data.neg[ind_t], :])

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.pos,:]'*∇l)
end


# -------------------------------------------------------------------------------
# Dual problem with hinge loss
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::M, data::Dual) where {M<:AbstractTopPushK{<:Hinge}}

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)

    objective   = - Convex.quadform(vcat(α, β), data.K)/2 + Convex.sum(α)/model.l.ϑ
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α <= model.l.ϑ*model.C,
                   α >= 0,
                   β >= 0]

    M <: TopPushK && push!(constraints, β <= Convex.sum(α)/model.K)

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.optimizer)

    return vec(α.value), vec(β.value)
end


# Graient descent + projection
function objective(model::AbstractTopPushK{<:Hinge}, data::Dual, α, β, s)
    - s'*vcat(α, β)/2 + sum(α)/model.l.ϑ
end


function gradient!(model::AbstractTopPushK{<:Hinge}, data::Dual, α, β, s, Δ)
    Δ             .= .- s
    Δ[data.indα] .+= 1/model.l.ϑ
end


function projection!(model::M, data::Dual, α, β) where {M<:AbstractTopPushK{<:Hinge}}
    K = M <: TopPushK ? model.K : 1
    αs, βs = projection(α, β, model.l.ϑ*model.C, K)
    α .= αs
    β .= βs
    return α, β 
end


# Coordinate descent


# -------------------------------------------------------------------------------
# Dual problem with truncated quadratic loss
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::M, data::Dual) where {M<:AbstractTopPushK{<:Quadratic}}

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)

    objective   = - Convex.quadform(vcat(α, β), data.K)/2 +
                    Convex.sum(α)/model.l.ϑ - Convex.sumsquares(α)/(4*model.C*model.l.ϑ^2)
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α >= 0,
                   β >= 0]

    M <: TopPushK && push!(constraints, β <= Convex.sum(α)/model.K)

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.optimizer)

    return vec(α.value), vec(β.value)
end


# Graient descent + projection
function objective(model::AbstractTopPushK{<:Quadratic}, data::Dual, α, β, s)
    - s'*vcat(α, β)/2 + sum(α)/model.l.ϑ - sum(abs2, α)/(4*model.C*model.l.ϑ^2)
end


function gradient!(model::AbstractTopPushK{<:Quadratic}, data::Dual, α, β, s, Δ)
    Δ             .= .- s
    Δ[data.indα] .+= 1/model.l.ϑ .- α/(2*model.l.ϑ^2*model.C)
end


function projection!(model::M, data::Dual, α, β) where {M<:AbstractTopPushK{<:Quadratic}}
    K = M <: TopPushK ? model.K : 1
    αs, βs = projection(α, β, K)
    α .= αs
    β .= βs
    return α, β 
end


# Coordinate descent
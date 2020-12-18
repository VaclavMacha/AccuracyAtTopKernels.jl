@with_kw_noshow struct TopPushK{S<:AbstractSurrogate, I<:Integer, T<:Real} <: AbstractTopPushK{S}
    K::I
    C::T = 1
    l::S = Hinge(ϑ = 1.0)

    @assert K >= 1 "The vaule of `K` must be greater or equal to 1."
end


TopPushK(K::Int) = TopPushK(K = K)


show(io::IO, model::TopPushK) =
    print(io, "TopPushK($(model.K), $(model.C), $(model.l))")


@with_kw_noshow struct TopPush{S<:AbstractSurrogate, T<:Real} <: AbstractTopPushK{S}
    C::T = 1
    l::S = Hinge(ϑ = 1.0)
end


show(io::IO, model::TopPush) =
    print(io, "TopPush($(model.C), $(model.l))")

@with_kw_noshow struct τFPL{S<:AbstractSurrogate, R<:Real, T<:Real} <: AbstractTopPushK{S}
    τ::R
    C::T = 1
    l::S = Hinge(ϑ = 1.0)

    @assert 0 < τ < 1  "The vaule of `τ` must lay in the interval (0,1)."
end


τFPL(τ::Real) = τFPL(τ = τ)


show(io::IO, model::τFPL) =
    print(io, "τFPL($(model.τ), $(model.C), $(model.l))")

getK(model::TopPushK, data) = model.K
getK(model::TopPush, data) = 1
getK(model::τFPL, data::Primal) = max(1, round(Int, model.τ * data.nneg))
getK(model::τFPL, data::Dual) = max(1, round(Int, model.τ * data.nβ))

# -------------------------------------------------------------------------------
# Primal problem - General solver
# -------------------------------------------------------------------------------
function optimize(solver::General, model::TopPushK, data::Primal)

    Xpos = @view data.X[data.ind_pos, :]
    Xneg = @view data.X[data.ind_neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)
    z = Convex.Variable(data.nneg)
    K = getK(model, data)

    objective = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t + Convex.sum(z)/K - Xpos*w,
                   z >= Xneg*w - t,
                   z >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(w.value), t.value + sum(z.value)/K
end


function optimize(solver::General, model::TopPush, data::Primal)

    Xpos = @view data.X[data.ind_pos, :]
    Xneg = @view data.X[data.ind_neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)

    objective = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t - Xpos*w,
                   t >= Convex.maximum(Xneg*w)]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(w.value), t.value
end

# -------------------------------------------------------------------------------
# Primal problem - Gradient descent solver
# -------------------------------------------------------------------------------
function initialization(model::AbstractTopPushK, data::Primal)
    w = zeros(eltype(data.X), data.dim)
    Δ = zero(w)
    s = scores(model, data, w)
    return w, s, Δ
end


function objective(model::AbstractTopPushK, data::Primal, w, t, s = scores(model, data, w))
    return w'*w/2 + model.C * sum(model.l.value.(t .- s[data.ind_pos]))
end


function threshold(model::AbstractTopPushK, data::Primal, s)
    K = getK(model, data)
   return mean(partialsort(s[data.ind_neg], 1:K, rev = true))
end


function threshold(model::TopPush, data::Primal, s)
   return maximum(s[data.ind_neg])
end


function gradient!(model::AbstractTopPushK, data::Primal, w, s, Δ)
    K = getK(model, data)
    ind_t = partialsortperm(s[data.ind_neg], 1:K, rev = true)
    t     = mean(s[data.ind_neg[ind_t]])

    ∇l = model.l.gradient.(t .- s[data.ind_pos])
    ∇t = vec(mean(data.X[data.ind_neg[ind_t], :], dims = 1))

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.ind_pos,:]'*∇l)
    return t
end


function gradient!(model::TopPush, data::Primal, w, s, Δ)
    t, ind_t = findmax(s[data.ind_neg])

    ∇l = model.l.gradient.(t .- s[data.ind_pos])
    ∇t = vec(data.X[data.ind_neg[ind_t], :])

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.ind_pos,:]'*∇l)
    return t
end


# -------------------------------------------------------------------------------
# Dual problem - General solver
# -------------------------------------------------------------------------------
# Hinge loss
function optimize(solver::General, model::M, data::Dual{<:DTrain}; ε::Real = 1e-6) where {M<:AbstractTopPushK{<:Hinge}}

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)
    K = getK(model, data)
    KK = data.K + ε .* I

    objective   = - Convex.quadform(vcat(α, β), KK)/2 + Convex.sum(α)/model.l.ϑ
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α <= model.l.ϑ*model.C,
                   α >= 0,
                   β >= 0]

    if M <: TopPushK || M <: τFPL
        push!(constraints, β <= Convex.sum(α)/K)
    end

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(α.value), vec(β.value)
end

# Truncated quadratic loss
function optimize(solver::General, model::M, data::Dual{<:DTrain}; ε::Real = 1e-6) where {M<:AbstractTopPushK{<:Quadratic}}

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)
    K = getK(model, data)
    KK = data.K + ε .* I

    objective   = - Convex.quadform(vcat(α, β), KK)/2 +
                    Convex.sum(α)/model.l.ϑ - Convex.sumsquares(α)/(4*model.C*model.l.ϑ^2)
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α >= 0,
                   β >= 0]

    if M <: TopPushK || M <: τFPL
        push!(constraints, β <= Convex.sum(α)/K)
    end

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(α.value), vec(β.value)
end


# -------------------------------------------------------------------------------
# Dual problem - Graient descent solver
# -------------------------------------------------------------------------------
function initialization(model::AbstractTopPushK, data::Dual{<:DTrain})
    αβ   = rand(eltype(data.K), data.nα + data.nβ)
    α, β = @views αβ[data.ind_α], αβ[data.ind_β]
    projection!(model, data, α, β)

    s  = data.K * vcat(α, β)
    return α, β, αβ, s
end


# Hinge loss
function objective(model::AbstractTopPushK{<:Hinge}, data::Dual{<:DTrain}, α, β, s = data.K * vcat(α, β))
    - s'*vcat(α, β)/2 + sum(α)/model.l.ϑ
end


function gradient!(model::AbstractTopPushK{<:Hinge}, data::Dual{<:DTrain}, α, β, s, Δ)
    Δ             .= .- s
    Δ[data.ind_α] .+= 1/model.l.ϑ
end


function projection!(model::M, data::Dual{<:DTrain}, α, β) where {M<:AbstractTopPushK{<:Hinge}}
    K = getK(model, data)
    αs, βs = projection(α, β, model.l.ϑ*model.C, K)
    α .= αs
    β .= βs
    return α, β
end


# Truncated quadratic loss
function objective(model::AbstractTopPushK{<:Quadratic}, data::Dual{<:DTrain}, α, β, s = data.K * vcat(α, β))
    - s'*vcat(α, β)/2 + sum(α)/model.l.ϑ - sum(abs2, α)/(4*model.C*model.l.ϑ^2)
end


function gradient!(model::AbstractTopPushK{<:Quadratic}, data::Dual{<:DTrain}, α, β, s, Δ)
    Δ             .= .- s
    Δ[data.ind_α] .+= 1/model.l.ϑ .- α/(2*model.l.ϑ^2*model.C)
end


function projection!(model::M, data::Dual{<:DTrain}, α, β) where {M<:AbstractTopPushK{<:Quadratic}}
    K = getK(model, data)
    αs, βs = projection(α, β, K)
    α .= αs
    β .= βs
    return α, β
end

function threshold(model::AbstractTopPushK, data::Dual, s)
    K = getK(model, data)
   return mean(partialsort(.- s[data.ind_β], 1:K, rev = true))
end


function threshold(model::TopPush, data::Dual, s)
   return maximum(.- s[data.ind_β])
end


function primal_objective(model::AbstractTopPushK, data::Dual{<:DTrain}, α, β, s = data.K * vcat(α, β))

    K = getK(model, data)
    t = threshold(model, data, s)
    z = max.(.- s[data.ind_β] .- t, 0)
    y = t + sum(z)/K  .- s[data.ind_α]
    w_norm = s'*vcat(α, β)/2
    return w_norm + model.C * sum(model.l.value.(y))
end

# -------------------------------------------------------------------------------
# Dual problem - Coordinate descent solver
# -------------------------------------------------------------------------------
function loss(model::AbstractTopPushK, data::Dual{<:DTrain}, a::Real, b::Real, Δ::Real)
    a*Δ^2/2 + b*Δ
end


function select_k(model::AbstractTopPushK, data::Dual{<:DTrain}, α, β)
    rand(1:(data.nα + data.nβ))
end


function apply!(model::AbstractTopPushK, data::Dual{<:DTrain}, best::BestUpdate, α, β, αβ, s, αsum, βsort)
    βsorted!(data, best, β, βsort)
    if best.k <= data.nα && best.l > data.nα
        αsum .+= best.Δ
    end
    αβ[best.k] = best.vars[1]
    αβ[best.l] = best.vars[2]
    scores!(data, best, s)
end


function apply!(model::TopPush, data::Dual{<:DTrain}, best::BestUpdate, α, β, αβ, s, αsum, βsort)
    αβ[best.k] = best.vars[1]
    αβ[best.l] = best.vars[2]
    scores!(data, best, s)
end


# Hinge loss
function rule_αα!(model::AbstractTopPushK{<:Hinge}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort)

    αk, αl = α[k], α[l]
    C, ϑ   = model.C, model.l.ϑ

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l]
    b = - s[k] + s[l]
    Δ = solution(a, b, max(-αk, αl - ϑ*C), min(ϑ*C - αk, αl))

    vars = (αk = αk + Δ, αl = αl - Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end


function rule_αβ!(model::M, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort) where {M<:AbstractTopPushK{<:Hinge}}

    αk, βl = α[k], β[l - data.nα]
    C, ϑ   = model.C, model.l.ϑ
    K = getK(model, data)

    a = - data.K[k,k] - 2*data.K[k,l] - data.K[l,l]
    b = - s[k] - s[l] + 1/ϑ

    if K == 1
        Δ = solution(a, b, max(-αk, -βl), C*ϑ - αk)
    else
        βmax = find_βmax(βsort, β, l - data.nα)
        Δ    = solution(a, b, max(-αk, -βl, K*βmax - αsum[1]), min(C*ϑ - αk, (αsum[1] - K*βl)/(K-1)))
    end

    vars = (αk = αk + Δ, βl = βl + Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end


function rule_ββ!(model::M, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort) where {M<:AbstractTopPushK{<:Hinge}}

    βk, βl = β[k - data.nα], β[l - data.nα]
    C, ϑ   = model.C, model.l.ϑ
    K = getK(model, data)

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l]
    b = - s[k] + s[l]

    if K == 1
        Δ = solution(a, b, -βk, βl)
    else
        Δ = solution(a, b, max(-βk, βl - αsum[1]/K), min(αsum[1]/K - βk, βl))
    end

    vars = (βk = βk + Δ, βl = βl - Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end


# Truncated quadratic loss
function rule_αα!(model::AbstractTopPushK{<:Quadratic}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort)

    αk, αl = α[k], α[l]
    C, ϑ   = model.C, model.l.ϑ

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l] - 1/(C*ϑ^2)
    b = - s[k] + s[l] - (αk - αl)/(2*C*ϑ^2)
    Δ = solution(a, b, - αk, αl)

    vars = (αk = αk + Δ, αl = αl - Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end


function rule_αβ!(model::M, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort) where {M<:AbstractTopPushK{<:Quadratic}}

    αk, βl = α[k], β[l - data.nα]
    C, ϑ   = model.C, model.l.ϑ
    K = getK(model, data)

    a = - data.K[k,k] - 2*data.K[k,l] - data.K[l,l] - 1/(2*C*ϑ^2)
    b = - s[k] - s[l] + 1/ϑ - αk/(2*C*ϑ^2)

    if K == 1
        Δ = solution(a, b, max(-αk, -βl), Inf)
    else
        βmax = find_βmax(βsort, β, l - data.nα)
        Δ    = solution(a, b, max(-αk, -βl, K*βmax - αsum[1]), (αsum[1] - K*βl)/(K-1))
    end

    vars = (αk = αk + Δ, βl = βl + Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end


function rule_ββ!(model::M, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, s, αsum, βsort) where {M<:AbstractTopPushK{<:Quadratic}}

    βk, βl = β[k - data.nα], β[l - data.nα]
    C, ϑ   = data.n, model.C, model.l.ϑ
    K = getK(model, data)

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l]
    b = - s[k] + s[l]

    if K == 1
        Δ = solution(a, b, -βk, βl)
    else
        Δ = solution(a, b, max(-βk, βl - αsum[1]/K), min(αsum[1]/K - βk, βl))
    end

    vars = (βk = βk + Δ, βl = βl - Δ)
    L    = loss(model, data, a, b, Δ)
    update!(best, k, l, Δ, L, vars)
end

struct PatMat{S<:AbstractSurrogate, T1<:Real, T2<:Real} <: AbstractModel
    l1::S
    l2::S
    τ::T1
    C::T2

    function PatMat(l1::S, l2::S, τ::T1, C::T2) where {S<:AbstractSurrogate, T1<:Real, T2<:Real}

        @assert 0 < τ < 1  "The vaule of `τ` must lay in the interval (0,1)."

        return new{S, T1, T2}(l1, l2, τ, C)
    end
end


# -------------------------------------------------------------------------------
# Primal problem - General solver
# -------------------------------------------------------------------------------
function optimize(solver::General, model::PatMat, data::Primal)

    Xpos = @view data.X[data.ind_pos, :]
    Xneg = @view data.X[data.ind_neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)
    z = Convex.Variable(data.n)

    objective   = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l1.value_exact.(y))
    constraints = [Convex.sum(model.l2.value_exact.(z)) <= data.n * model.τ,
                   y == t - Xpos*w,
                   z == data.X*w - t]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(w.value), t.value
end

# -------------------------------------------------------------------------------
# Primal problem - Gradient descent solver
# -------------------------------------------------------------------------------
function initialization(model::PatMat, data::Primal, w0)
    w = zeros(eltype(data.X), data.dim)

    isempty(w0) || (w .= w0) 
    s = scores(model, data, w)
    Δ = zero(w)
    return w, s, Δ
end


function objective(model::PatMat, data::Primal, w, t, s = data.X*w)
    w'*w/2 + model.C * sum(model.l1.value.(t .- s[data.ind_pos]))
end


function threshold(model::PatMat, data::Primal, s)
   Roots.find_zero(t -> sum(model.l2.value.(s .- t)) - data.n*model.τ, (-Inf, Inf))
end


function gradient!(model::PatMat, data::Primal, w, s, Δ)
    t   = threshold(model, data, s)
    ∇l1 = model.l1.gradient.(t .- s[data.ind_pos])
    ∇l2 = model.l2.gradient.(s .- t)
    ∇t  = data.X'*∇l2/sum(∇l2)

    Δ .= w .+ model.C .* (sum(∇l1)*∇t .- data.X[data.ind_pos,:]'*∇l1)
end


# -------------------------------------------------------------------------------
# Dual problem - General solver
# -------------------------------------------------------------------------------
# Hinge loss
function optimize(solver::General, model::PatMat{<:Hinge}, data::Dual{<:DTrain})

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)
    δ = Convex.Variable()

    objective   = - Convex.quadform(vcat(α, β), data.K)/2 + Convex.sum(α)/model.l1.ϑ + 
                    Convex.sum(β)/model.l2.ϑ - δ*data.n*model.τ
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α <= model.l1.ϑ*model.C,
                   β <= model.l2.ϑ*δ,
                   α >= 0,
                   β >= 0]

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(α.value), vec(β.value), δ.value
end


# Truncated quadratic loss
function optimize(solver::General, model::PatMat{<:Quadratic}, data::Dual{<:DTrain})

    α = Convex.Variable(data.nα)
    β = Convex.Variable(data.nβ)
    δ = Convex.Variable()

    objective   = - Convex.quadform(vcat(α, β), data.K)/2 +
                    Convex.sum(α)/model.l1.ϑ - Convex.sumsquares(α)/(4*model.C*model.l1.ϑ^2) +
                    Convex.sum(β)/model.l2.ϑ - Convex.quadoverlin(β, δ)/(4*model.l2.ϑ^2) -
                    δ*data.n*model.τ
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α >= 0,
                   β >= 0,
                   δ >= 0]

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, solver.solver)

    return vec(α.value), vec(β.value), δ.value
end


# -------------------------------------------------------------------------------
# Dual problem - Graient descent solver
# -------------------------------------------------------------------------------
function initialization(model::PatMat, data::Dual{<:DTrain}, α0, β0)
    αβδ     = rand(eltype(data.K), data.nα + data.nβ + 1)
    α, β, δ = @views αβδ[data.ind_α], αβδ[data.ind_β], αβδ[[end]]

    isempty(α0) || (α .= α0)
    isempty(β0) || (β .= β0)
    δ .= init_δ(model, data, α, β)
    projection!(model, data, α, β, δ)

    s = data.K * vcat(α, β)
    return α, β, δ, αβδ, s
end


function init_δ(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, α0, β0)
    maximum(β0)/model.l1.ϑ
end


function init_δ(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, α0, β0)
    δ0 = sqrt(sum(abs2, β0)/(4*data.n*model.τ*model.l2.ϑ^2))
    iszero(δ0) && (δ0 += 1e-8)
    return δ0
end


# Hinge loss
function objective(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, α, β, δ, s = data.K * vcat(α, β))
    - s'*vcat(α, β)/2 + sum(α)/model.l1.ϑ + sum(β)/model.l2.ϑ - δ*data.n*model.τ
end


function gradient!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, α, β, δ, s, Δ)
    Δ[1:end-1]    .= .- s
    Δ[data.ind_α] .+= 1/model.l1.ϑ
    Δ[data.ind_β] .+= 1/model.l2.ϑ
    Δ[end]         = - data.n*model.τ
end


function projection!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, α, β, δ)
     αs, βs, δs = projection(α, β, δ[1], model.l1.ϑ*model.C, model.l2.ϑ)
     α .= αs
     β .= βs
     δ .= δs
     return α, β, δ 
end


# Truncated quadratic loss
function objective(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, α, β, δ, s = data.K * vcat(α, β))
    return - s'*vcat(α, β)/2 +
           sum(α)/model.l1.ϑ - sum(abs2, α)/(4*model.C*model.l1.ϑ^2) +
           sum(β)/model.l2.ϑ - sum(abs2, β)/(4*δ*model.l2.ϑ^2) -
           δ*data.n*model.τ
end


function gradient!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, α, β, δ, s, Δ)
    Δ[1:end-1]    .= .- s
    Δ[data.ind_α] .+= 1/model.l1.ϑ .- α/(2*model.l1.ϑ^2*model.C)
    Δ[data.ind_β] .+= 1/model.l2.ϑ .- β/(2*model.l2.ϑ^2*δ[1])
    Δ[end]         = sum(abs2, β)/(4*model.l2.ϑ^2*δ[1]^2) - data.n*model.τ
end


function projection!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, α, β, δ)
     αs, βs, δs = projection(α, β, δ[1])
     α .= αs
     β .= βs
     δ .= δs
     return α, β, δ 
end


# -------------------------------------------------------------------------------
# Dual problem - Coordinate descent solver
# -------------------------------------------------------------------------------
function select_k(model::PatMat, data::Dual{<:DTrain}, α, β, δ)
    if rand() <= 0.9
        return data.nα + findmax(β)[2]
    else
        return rand(1:(data.nα + data.nβ))
    end
end


# Hinge loss
function loss(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, a::Real, b::Real, δ::Real, Δ::Real)
    a*Δ^2/2 + b*Δ - δ*data.n*model.τ
end


function apply!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, best::BestUpdate, α, β, δ, αβδ, s, βsort)
    βsorted!(data, best, β, βsort)
    αβδ[best.k] = best.vars[1]
    αβδ[best.l] = best.vars[2]
    αβδ[end]    = best.vars[3]
    scores!(data, best, s)
end


function rule_αα!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, βsort)

    αk, αl   = α[k], α[l]
    n, C, ϑ1 = data.n, model.C, model.l1.ϑ

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l]
    b = - s[k] + s[l]
    Δ = solution(a, b, max(-αk, αl - ϑ1*C), min(ϑ1*C - αk, αl))

    vars = (αk = αk + Δ, αl = αl - Δ, δ = δ[1]) 
    L    = loss(model, data, a, b, vars.δ, Δ)
    update!(best, k, l, Δ, L, vars)
end


function rule_αβ!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, βsort)

    αk, βl = α[k], β[l - data.nα]
    n, τ, C, ϑ1, ϑ2 = data.n, model.τ, model.C, model.l1.ϑ, model.l2.ϑ
    βmax   = find_βmax(βsort, β, l - data.nα)

    a = - data.K[k,k] - 2*data.K[k,l] - data.K[l,l]
    b = - s[k] - s[l] + 1/ϑ1 + 1/ϑ2

    # solution 1
    Δ = solution(a, b, max(-αk, -βl), ϑ1*C - αk)
    if βl + Δ <= βmax
        vars = (αk = αk + Δ, βl = βl + Δ, δ = βmax/ϑ2)
        L    = loss(model, data, a, b, vars.δ, Δ)
        update!(best, k, l, Δ, L, vars)
    end

    # solution 2
    Δ = solution(a, b - n*τ/ϑ2, max(-αk, -βl), ϑ1*C - αk)
    if βl + Δ >= βmax
        vars = (αk = αk + Δ, βl = βl + Δ, δ = (βl + Δ)/ϑ2)
        L    = loss(model, data, a, b - n*τ/ϑ2, vars.δ, Δ)
        update!(best, k, l, Δ, L, vars)
    end
end


function rule_ββ!(model::PatMat{<:Hinge}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, βsort)

    βk, βl   = β[k - data.nα], β[l - data.nα]
    n, τ, ϑ2 = data.n, model.τ, model.l2.ϑ
    βmax     = find_βmax(βsort, β, k - data.nα, l - data.nα)

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l]
    b = - s[k] + s[l]

    # solution 1
    Δ = solution(a, b, -βk, βl)
    if βmax >= max(βk + Δ, βl - Δ)
        vars = (βk = βk + Δ, βl = βl - Δ, δ = βmax/ϑ2)
        L    = loss(model, data, a, b, vars.δ, Δ)
        update!(best, k, l, Δ, L, vars)
    end

    # solution 2
    Δ = solution(a, b - n*τ/ϑ2, -βk, βl)
    if βk + Δ >= max(βmax, βl - Δ)
        vars = (βk = βk + Δ, βl = βl - Δ, δ = (βk + Δ)/ϑ2)
        L    = loss(model, data, a, b - n*τ/ϑ2, vars.δ, Δ)
        update!(best, k, l, Δ, L, vars)
    end

    # solution 3
    Δ = solution(a, b + n*τ/ϑ2, -βk, βl) 
    if βl - Δ >= max(βk + Δ, βmax)
        vars = (βk = βk + Δ, βl = βl - Δ, δ = (βl - Δ)/ϑ2)
        L    = loss(model, data, a, b + n*τ/ϑ2, vars.δ, Δ)
        update!(best, k, l, Δ, L, vars)
    end
end


# Truncated quadratic loss
function loss(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, a::Real, b::Real, δ::Real, Δ::Real, β2sum)
    a*Δ^2/2 + b*Δ - β2sum[1]/δ - δ*data.n*model.τ
end


function apply!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, best::BestUpdate, α, β, δ, αβδ, s, β2sum)
    if best.k > data.nα || best.l > data.nα
        if best.k <= data.nα && best.l > data.nα
            β2sum .+= best.Δ*(best.Δ + 2*β[best.l - data.nα])/(4*model.l2.ϑ^2)
        else 
            β2sum .+= best.Δ*(2*best.Δ + 2*(β[best.k - data.nα] - β[best.l - data.nα]))/(4*model.l2.ϑ^2)
        end
    end
    αβδ[best.k] = best.vars[1]
    αβδ[best.l] = best.vars[2]
    αβδ[end]    = best.vars[3]
    scores!(data, best, s)
end


function rule_αα!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, β2sum)

    αk, αl   = α[k], α[l]
    n, C, ϑ1 = data.n, model.C, model.l1.ϑ

    a = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l] - 1/(C*ϑ1^2) 
    b = - s[k] + s[l] - (αk - αl)/(2*C*ϑ1^2)
    Δ = solution(a, b, - αk, αl)

    vars = (αk = αk + Δ, αl = αl - Δ, δ = δ[1]) 
    L    = loss(model, data, a, b, vars.δ, Δ, β2sum)
    update!(best, k, l, Δ, L, vars)
end


function rule_αβ!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, β2sum)

    αk, βl = α[k], β[l - data.nα]
    n, τ, C, ϑ1, ϑ2 = data.n, model.τ, model.C, model.l1.ϑ, model.l2.ϑ

    a    = - data.K[k,k] - 2*data.K[k,l] - data.K[l,l] - 1/(2*C*ϑ1^2) - 1/(2*δ[1]*ϑ2^2) 
    b    = - s[k] - s[l] + 1/ϑ1 - αk/(2*C*ϑ1^2) + 1/ϑ2 - βl/(2*δ[1]*ϑ2^2)
    Δ    = solution(a, b, max(-αk, -βl), Inf)
    δnew = sqrt(δ[1]^2 + (Δ^2 + 2*Δ*βl)/(4*ϑ2^2*n*τ))

    a +=    (1/δ[1] - 1/δnew)/(2*ϑ2^2)
    b += βl*(1/δ[1] - 1/δnew)/(2*ϑ2^2)

    vars = (αk = αk + Δ, βl = βl + Δ, δ = δnew)
    L    = loss(model, data, a, b, vars.δ, Δ, β2sum)
    update!(best, k, l, Δ, L, vars)
end


function rule_ββ!(model::PatMat{<:Quadratic}, data::Dual{<:DTrain}, best::BestUpdate, k, l, α, β, δ, s, β2sum)

    βk, βl = β[k - data.nα], β[l - data.nα]
    n, τ, C, ϑ1, ϑ2 = data.n, model.τ, model.C, model.l1.ϑ, model.l2.ϑ

    a    = - data.K[k,k] + 2*data.K[k,l] - data.K[l,l] - 1/(δ[1]*ϑ2^2)
    b    = - s[k] + s[l] - (βk - βl)/(2*δ[1]*ϑ2^2)
    Δ    = solution(a, b, -βk, βl)
    δnew = sqrt(δ[1]^2 + (Δ^2 + Δ*(βk - βl))/(2*ϑ2^2*n*τ))

    a += (1/δ[1] - 1/δnew)/(ϑ2^2)
    b += (1/δ[1] - 1/δnew)*(βk - βl)/(2*ϑ2^2)
 
    vars = (βk = βk + Δ, βl = βl - Δ, δ = δnew)
    L    = loss(model, data, a, b, vars.δ, Δ, β2sum)
    update!(best, k, l, Δ, L, vars)
end
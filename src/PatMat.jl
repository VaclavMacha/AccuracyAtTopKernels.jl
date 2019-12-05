struct PatMat{S<:AbstractSurrogate} <: AbstractModel
    l1::S
    l2::S
    τ::Real
    C::Real

    function PatMat(l1::S, l2::S, τ::Real, C::Real) where {S<:AbstractSurrogate}

        @assert 0 < τ < 1  "The vaule of `τ` must lay in the interval (0,1)."

        return new{S}(l1, l2, τ, C)
    end
end


function initialization(model::PatMat, data::Primal, w0)
    w = zeros(eltype(data.X), data.dim)

    isempty(w0) || (w .= w0) 
    Δ = zero(w)
    s = data.X * w
    return w, Δ, s
end


function initialization(model::PatMat, data::Dual, α0, β0)
    αβδ     = rand(eltype(data.K), data.n + 1)
    α, β, δ = @views αβδ[data.indα], αβδ[data.indβ], αβδ[[end]]

    isempty(α0) || (α .= α0)
    isempty(β0) || (β .= β0)
    δ .= init_δ(model, data, α, β)
    projection!(model, data, α, β, δ)

    Δ  = zero(αβδ)
    s  = data.K * vcat(α, β)
    
    return α, β, δ, αβδ, Δ, s
end

function init_δ(model::PatMat{<:Hinge}, data::Dual, α0, β0)
    maximum(β0)/model.l1.ϑ
end

function init_δ(model::PatMat{<:Quadratic}, data::Dual, α0, β0)
    δ0 = sqrt(sum(abs2, β0)/(4*data.n*model.τ*model.l2.ϑ^2))
    iszero(δ0) && (δ0 += 1e-8)
    return δ0
end

# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::PatMat, data::Primal)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

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

# Our solution
function objective(model::PatMat, data::Primal, w, t, s)
    w'*w/2 + model.C * sum(model.l1.value.(t .- s[data.pos]))
end


function threshold(model::PatMat, data::Primal, s)
   Roots.find_zero(t -> sum(model.l2.value.(s .- t)) - data.n*model.τ, (-Inf, Inf))
end


function gradient!(model::PatMat, data::Primal, w, s, Δ)
    t   = threshold(model, data, s)
    ∇l1 = model.l1.gradient.(t .- s[data.pos])
    ∇l2 = model.l2.gradient.(s .- t)
    ∇t  = data.X'*∇l2/sum(∇l2)

    Δ .= w .+ model.C .* (sum(∇l1)*∇t .- data.X[data.pos,:]'*∇l1)
end


# -------------------------------------------------------------------------------
# Dual problem with hinge loss
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::PatMat{<:Hinge}, data::Dual)

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


# Graient descent + projection
function objective(model::PatMat{<:Hinge}, data::Dual, α, β, δ, s)
    - s'*vcat(α, β)/2 + sum(α)/model.l1.ϑ + sum(β)/model.l2.ϑ - δ*data.n*model.τ
end


function gradient!(model::PatMat{<:Hinge}, data::Dual, α, β, δ, s, Δ)
    Δ[1:end-1]    .= .- s
    Δ[data.indα] .+= 1/model.l1.ϑ
    Δ[data.indβ] .+= 1/model.l2.ϑ
    Δ[end]         = - data.n*model.τ
end


function projection!(model::PatMat{<:Hinge}, data::Dual, α, β, δ)
     αs, βs, δs = projection(α, β, δ[1], model.l1.ϑ*model.C, model.l2.ϑ)
     α .= αs
     β .= βs
     δ .= δs
     return α, β, δ 
end


# Coordinate descent


# -------------------------------------------------------------------------------
# Dual problem with truncated quadratic loss
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::PatMat{<:Quadratic}, data::Dual)

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


# Graient descent + projection
function objective(model::PatMat{<:Quadratic}, data::Dual, α, β, δ, s)
    return - s'*vcat(α, β)/2 +
           sum(α)/model.l1.ϑ - sum(abs2, α)/(4*model.C*model.l1.ϑ^2) +
           sum(β)/model.l2.ϑ - sum(abs2, β)/(4*δ*model.l2.ϑ^2) -
           δ*data.n*model.τ
end


function gradient!(model::PatMat{<:Quadratic}, data::Dual, α, β, δ, s, Δ)
    Δ[1:end-1]    .= .- s
    Δ[data.indα] .+= 1/model.l1.ϑ .- α/(2*model.l1.ϑ^2*model.C)
    Δ[data.indβ] .+= 1/model.l2.ϑ .- β/(2*model.l2.ϑ^2*δ[1])
    Δ[end]         = sum(abs2, β)/(4*model.l2.ϑ^2*δ[1]^2) - data.n*model.τ
end


function projection!(model::PatMat{<:Quadratic}, data::Dual, α, β, δ)
     αs, βs, δs = projection(α, β, δ[1])
     α .= αs
     β .= βs
     δ .= δs
     return α, β, δ 
end


# Coordinate descent
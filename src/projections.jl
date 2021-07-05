# -------------------------------------------------------------------------------
# Projection PatMat hinge
# -------------------------------------------------------------------------------
# general solver
function projection_exact(α0::AbstractVector,
                          β0::AbstractVector,
                          δ0::Real,
                          C1::Real,
                          C2::Real;
                          solver::Any = Optimizer(verbose = false))

    α = Convex.Variable(length(α0))
    β = Convex.Variable(length(β0))
    δ = Convex.Variable()

    objective   = Convex.sumsquares(α - α0)/2 + Convex.sumsquares(β - β0)/2 +
                  Convex.sumsquares(δ - δ0)/2
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α <= C1,
                   α >= 0,
                   β <= C2*δ,
                   β >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver)

    return vec(α.value), vec(β.value), δ.value
end


# our solution
function f(μ::Real, α0::AbstractVector, β0::AbstractVector, δ0::Real, C1::Real, C2::Real)
    return C2*δ0 + C2^2*sum(max.(β0 .- μ, 0)) - μ
end


function h(μ::Real, α0::AbstractVector, β0::AbstractVector, δ0::Real, C1::Real, C2::Real)
    λ = f(μ, α0, β0, δ0, C1, C2)
    return sum(min.(max.(α0 .- λ, 0), C1)) - sum(min.(max.(β0 .+ λ, 0), λ + μ))
end


function projection(α0::AbstractVector, β0::AbstractVector, δ0::Real, C1::Real, C2::Real)

    if δ0 <= - C2*sum(max.(β0 .+ maximum(α0), 0))
        return zero(α0), zero(β0), zero(δ0)
    end

    μ_ub1 = minimum(β0)
    μ_ub2 = minimum(C2*δ0 + C2^2*sum(β0) .- α0)/(C2^2*length(β0) + 1)
    μ_ub  = min(μ_ub1, μ_ub2) - 1e8
    μ_lb  = maximum(β0) + C2*max(δ0, 0) + 1e8

    μ = Roots.find_zero(μ -> h(μ, α0, β0, δ0, C1, C2), (μ_lb, μ_ub))
    λ = f(μ, α0, β0, δ0, C1, C2)

    α = @. min(max(α0 - λ, 0), C1)
    β = @. min(max(β0 + λ, 0), λ + μ)
    δ = (λ + μ)/C2

    return α, β, δ
end


# -------------------------------------------------------------------------------
# Projection PatMat quadratic
# -------------------------------------------------------------------------------
# general solver
function projection_exact(α0::AbstractVector,
                          β0::AbstractVector,
                          δ0::Real;
                          solver::Any = Optimizer(verbose = false))

    α = Convex.Variable(length(α0))
    β = Convex.Variable(length(β0))
    δ = Convex.Variable()

    objective   = Convex.sumsquares(α - α0)/2 + Convex.sumsquares(β - β0)/2 +
                  Convex.sumsquares(δ - δ0)/2
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α >= 0,
                   β >= 0,
                   δ >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver)

    return vec(α.value), vec(β.value), δ.value
end


# our solution
function f(λ::Real, α0::AbstractVector, β0::AbstractVector, δ0::Real)
    return sum(max.(α0 .- λ, 0)) - sum(max.(β0 .+ λ, 0))
end


function projection(α0::AbstractVector, β0::AbstractVector, δ0::Real)

    if - maximum(β0) > maximum(α0)
        return zero(α0), zero(β0), zero(δ0)
    end

    λ = Roots.find_zero(λ -> f(λ, α0, β0, δ0), (- maximum(β0), maximum(α0)))

    α = @. max(α0 - λ, 0)
    β = @. max(β0 + λ, 0)
    δ = max(δ0, 1e-4)

    return α, β, δ
end


# -------------------------------------------------------------------------------
# Projection TopPushK hinge
# -------------------------------------------------------------------------------
# general solver
function projection_exact(α0::AbstractVector,
                          β0::AbstractVector,
                          C::Real,
                          K::Integer;
                          solver::Any = Optimizer(verbose = false))

    α = Convex.Variable(length(α0))
    β = Convex.Variable(length(β0))

    objective   = Convex.sumsquares(α - α0)/2 + Convex.sumsquares(β - β0)/2
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α <= C,
                   α >= 0,
                   β <= Convex.sum(α)/K,
                   β >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver)

    return vec(α.value), vec(β.value)
end


# our solution
function f(μ::Real, s::AbstractVector, K::Integer)
    i, j, d  = 2, 1, 1
    λ, λ_old = s[1], 0
    g, g_old = -K*μ, -K*μ

    while g < 0
        g_old = g
        λ_old = λ

        if s[i] <= s[j] + μ
            g += d*(s[i] - λ)
            λ  = s[i]
            d += 1
            i += 1
        else
            g += d*(s[j] + μ - λ)
            λ  = s[j] + μ
            d -= 1
            j += 1
        end
    end
    return -(λ - λ_old)/(g - g_old)*g_old + λ_old
end


function h(μ::Real, s::AbstractVector, α0::AbstractVector, β0::AbstractVector, C::Real, K::Integer)
    λ = f(μ, s, K)
    return sum(min.(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- μ, 0))/K, 0), C)) - K*μ
end


function projection(α0::AbstractVector, β0::AbstractVector, C::Real, K::Real)

    if K >= length(β0)
        @error "No feasible solution: K < length(β0) needed."
        return α0, β0
    end

    if mean(partialsort(β0, 1:K; rev = true)) + maximum(α0) <= 0
        return zero(α0), zero(β0)
    end

    s    = vcat(.- sort(β0; rev = true), Inf)
    μ_lb = 1e-10
    μ_ub = length(α0)*C/K + 1e-6
 
    μ = Roots.find_zero(μ -> h(μ, s, α0, β0, C, K), (μ_lb, μ_ub))
    λ = f(μ, s, K)
    δ = sum(max.(β0 .+ λ .- μ, 0))/K

    α = @. max(min(α0 - λ + δ, C), 0)
    β = @. max(min(β0 + λ, μ), 0)

    return α, β
end


# ---------------------------------------------------------------------------------------
# Projection TopPushK quadratic
# ---------------------------------------------------------------------------------------
# general solver
function projection_exact(α0::AbstractVector,
                          β0::AbstractVector,
                          K::Integer;
                          solver::Any = Optimizer(verbose = false))

    α = Convex.Variable(length(α0))
    β = Convex.Variable(length(β0))

    objective   = Convex.sumsquares(α - α0)/2 + Convex.sumsquares(β - β0)/2
    constraints = [Convex.sum(α) == Convex.sum(β),
                   α >= 0,
                   β <= Convex.sum(α)/K,
                   β >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, solver)

    return vec(α.value), vec(β.value)
end


# our solution
function h(μ::Real, s::AbstractVector, α0::AbstractVector, β0::AbstractVector, K::Integer)
    λ = f(μ, s, K)
    return sum(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- μ, 0))/K, 0)) - K*μ
end


function projection(α0::AbstractVector, β0::AbstractVector, K::Integer)

    if K >= length(β0)
        @error "No feasible solution: K < length(β0) needed."
        return α0, β0
    end

    if mean(partialsort(β0, 1:K; rev = true)) + maximum(α0) <= 0
        return zero(α0), zero(β0)
    end

    s    = vcat(.- sort(β0; rev = true), Inf)
    μ_lb = 1e-10
    μ_ub = length(α0)*(maximum(α0) + maximum(β0))/K
 
    μ = Roots.find_zero(μ -> h(μ, s, α0, β0, K), (μ_lb, μ_ub))
    λ = f(μ, s, K)
    δ = sum(max.(β0 .+ λ .- μ, 0))/K

    α = @. max(α0 - λ + δ, 0)
    β = @. max(min(β0 + λ, μ), 0)

    return α, β
end
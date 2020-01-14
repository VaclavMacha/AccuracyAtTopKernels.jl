# General solver
function solve(solver::General, model::AbstractModel, data::Primal)
    Random.seed!(solver.seed)

    w, t  = optimize(solver, model, data)
    w     = Vector(w)
    state = State(solver, model, :optimal; w = copy(w), t = copy(t))

    return (w = w, t = t, state = state)
end


function solve(solver::General, model::PatMat, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    α, β, δ = optimize(solver, model, data)
    α       = Vector(α)
    β       = Vector(β)
    t       = exact_threshold(model, data, α, β)
    state   = State(solver, model, :optimal; α = copy(α), β = copy(β), δ = copy(δ))

    return (α = α, β = β, δ = δ, t = t, state = state)
end


function solve(solver::General, model::AbstractTopPushK, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    α, β  = optimize(solver, model, data)
    α     = Vector(α)
    β     = Vector(β)
    t     = exact_threshold(model, data, α, β)
    state = State(solver, model, :optimal; α = copy(α), β = copy(β))

    return (α = α, β = β, t = t, state = state)
end


# -------------------------------------------------------------------------------
# Primal problem - gradient solver
# -------------------------------------------------------------------------------
function solve(solver::Gradient, model::AbstractModel, data::Primal, w0 = Float64[])
    Random.seed!(solver.seed)

    w, s, Δ  = initialization(model, data, w0)
    t        = threshold(model, data, s)

    state    = State(solver, model; w = copy(w))
    progress = ProgressBar(solver, model, data, w, t, s)

    # optimization
    for iter in 1:solver.maxiter
        # update score and compute gradient
        s .= data.X * w
        t  = gradient!(model, data, w, s, Δ)

        # progress and state
        progress(solver, model, data, iter, w, t, s)

        # update solution
        minimize!(solver, w, Δ)
        state(iter; w = copy(w))
    end

    state(; w = copy(w))

    return (w = Vector(w), t = threshold(model, data, data.X * w), state = state)
end


# -------------------------------------------------------------------------------
# Dual problem - gradient solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Gradient, model::PatMat, data::Dual{<:DTrain}, α0 = Float64[], β0 = Float64[])
    Random.seed!(solver.seed)

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    Δ               = zero(αβδ)

    state    = State(solver, model; α = copy(α), β = copy(β), δ = copy(δ[1]))
    progress = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        s .= data.K * vcat(α, β)

        # progress and state
        progress(solver, model, data, iter,  α, β, δ[1], s)

        # compute gradient and update solution
        gradient!(model, data, α, β, δ, s, Δ)
        maximize!(solver, αβδ, Δ)
        projection!(model, data, α, β, δ)
        state(iter; α = copy(α), β = copy(β), δ = copy(δ[1]))
    end

    α = Vector(α)
    β = Vector(β)
    state(α = copy(α), β = copy(β), δ = copy(δ[1]))

    return (α = α, β = β, δ = δ[1], t = exact_threshold(model, data, α, β), state = state)
end


# TopPushK
function solve(solver::Gradient, model::AbstractTopPushK, data::Dual{<:DTrain}, α0 = Float64[], β0 = Float64[])
    Random.seed!(solver.seed)

    α, β, αβ, s = initialization(model, data, α0, β0)
    Δ           = zero(αβ)

    state    = State(solver, model; α = copy(α), β = copy(β))
    progress = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        s .= data.K * vcat(α, β)

        # progress and state
        progress(solver, model, data, iter,  α, β, s)

        # ccompute gradient and update solution
        gradient!(model, data, α, β, s, Δ)
        maximize!(solver, αβ, Δ)
        projection!(model, data, α, β)
        state(iter; α = copy(α), β = copy(β))
    end

    α = Vector(α)
    β = Vector(β)
    state(; α = copy(α), β = copy(β))

    return (α = Vector(α), β = Vector(β), t = exact_threshold(model, data, α, β), state = state)
end


# -------------------------------------------------------------------------------
# Dual problem - coordinate descent solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Coordinate,
               model::PatMat{<:S},
               data::Dual{<:DTrain},
               α0 = Float64[],
               β0 = Float64[]) where {S<:AbstractSurrogate}

    Random.seed!(solver.seed)

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    S <: Hinge     && ( βtmp = sort(β, rev = true) )
    S <: Quadratic && ( βtmp = [sum(abs2, β)/(4*model.l2.ϑ^2)] )
 
    state    = State(solver, model; α = copy(α), β = copy(β), δ = copy(δ[1]))
    progress = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β, δ)
        best = select_rule(model, data, k, α, β, δ, s, βtmp)
        iszero(best.Δ) || apply!(model, data, best, α, β, δ, αβδ, s, βtmp)

        # progress and state
        progress(solver, model, data, iter, α, β, δ[1], s)
        state(iter; α = copy(α), β = copy(β), δ = copy(δ[1]))
    end

    α = Vector(α)
    β = Vector(β)
    state(; α = copy(α), β = copy(β), δ = copy(δ[1]))

    return (α = α, β = β, δ = δ[1], t = exact_threshold(model, data, α, β), state = state)
end


# TopPushK
function solve(solver::Coordinate,
               model::AbstractTopPushK,
               data::Dual{<:DTrain},
               α0 = Float64[],
               β0 = Float64[])

    Random.seed!(solver.seed)

    α, β, αβ, s = initialization(model, data, α0, β0)
    αsum        = [sum(α)]
    βsort       = sort(β, rev = true)

    state    = State(solver, model; α = copy(α), β = copy(β))
    progress = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β)
        best = select_rule(model, data, k, α, β, s, αsum, βsort)
        iszero(best.Δ) || apply!(model, data, best, α, β, αβ, s, αsum, βsort)

        # progress and state
        progress(solver, model, data, iter,  α, β, s)
        state(iter; α = copy(α), β = copy(β))
    end

    α = Vector(α)
    β = Vector(β)
    state(; α = copy(α), β = copy(β))

    return (α = α, β = β, t = exact_threshold(model, data, α, β), state = state)
end


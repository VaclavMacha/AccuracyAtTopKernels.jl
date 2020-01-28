# General solver
function solve(solver::General, model::AbstractModel, data::Primal)
    Random.seed!(solver.seed)

    val, tm, = @timed optimize(solver, model, data)
    w, t     = val
    state    = ProgStateInit(solver, model, tm; w = copy(w), t = copy(t))

    return (w = copy(w), t = copy(t), state = state)
end


function solve(solver::General, model::AbstractPatMat, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    val, tm, = @timed optimize(solver, model, data)
    α, β, δ  = val
    t        = exact_threshold(model, data, α, β)
    state    = ProgStateInit(solver, model, tm; α = copy(α), β = copy(β), δ = copy(δ))

    return (α = copy(α), β = copy(β), δ = copy(δ), t = copy(t), state = state)
end


function solve(solver::General, model::AbstractTopPushK, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    val, tm, = @timed optimize(solver, model, data)
    α, β     = val
    t        = exact_threshold(model, data, α, β)
    state    = ProgStateInit(solver, model, tm; α = copy(α), β = copy(β))

    return (α = copy(α), β = copy(β), t = copy(t), state = state)
end


# -------------------------------------------------------------------------------
# Primal problem - gradient solver
# -------------------------------------------------------------------------------
function solve(solver::Gradient, model::AbstractModel, data::Primal)
    Random.seed!(solver.seed)

    w, s, Δ         = initialization(model, data)
    t               = threshold(model, data, s)
    state, progress = ProgStateInit(solver, model, data, s; w = copy(w), t = copy(t))

    # optimization
    for iter in 1:solver.maxiter
        # progress and state
        update!(state, progress, solver, model, data, iter - 1, s; w = copy(w), t = copy(t))

        # update score and compute gradient
        s .= data.X * w
        t  = gradient!(model, data, w, s, Δ)
        minimize!(solver, w, Δ)
    end

    s .= data.X * w
    t  = threshold(model, data, data.X * w)
    update!(state, progress, solver, model, data, solver.maxiter, s; w = copy(w), t = copy(t))

    return (w = copy(w), t = copy(t), state = state)
end


# -------------------------------------------------------------------------------
# Dual problem - gradient solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Gradient, model::AbstractPatMat, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    α, β, δ, αβδ, s = initialization(model, data)
    Δ               = zero(αβδ)
    state, progress = ProgStateInit(solver, model, data, s; α = copy(α), β = copy(β), δ = copy(δ[1]))

    # optimization
    for iter in 1:solver.maxiter
        # compute gradient and update solution
        gradient!(model, data, α, β, δ, s, Δ)
        maximize!(solver, αβδ, Δ)
        projection!(model, data, α, β, δ)

        # update score
        s .= data.K * vcat(α, β)
 
        # progress and state
        update!(state, progress, solver, model, data, iter, s; α = copy(α), β = copy(β), δ = copy(δ[1]))
    end

    t = exact_threshold(model, data, α, β)

    return (α = copy(α), β = copy(β), δ = copy(δ[1]), t = t, state = state)
end


# TopPushK
function solve(solver::Gradient, model::AbstractTopPushK, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    α, β, αβ, s     = initialization(model, data)
    Δ               = zero(αβ)
    state, progress = ProgStateInit(solver, model, data, s; α = copy(α), β = copy(β))

    # optimization
    for iter in 1:solver.maxiter
        # ccompute gradient and update solution
        gradient!(model, data, α, β, s, Δ)
        maximize!(solver, αβ, Δ)
        projection!(model, data, α, β)

        # update score
        s .= data.K * vcat(α, β)

        # progress and state
        update!(state, progress, solver, model, data, iter, s; α = copy(α), β = copy(β))
    end
    
    t = exact_threshold(model, data, α, β)

    return (α = copy(α), β = copy(β), t = copy(t) , state = state)
end


# -------------------------------------------------------------------------------
# Dual problem - coordinate descent solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Coordinate, model::AbstractPatMat{<:S}, data::Dual{<:DTrain}) where {S<:AbstractSurrogate}
    Random.seed!(solver.seed)

    α, β, δ, αβδ, s = initialization(model, data)
    S <: Hinge     && ( βtmp = sort(β, rev = true) )
    S <: Quadratic && ( βtmp = [sum(abs2, β)/(4*model.l2.ϑ^2)] )
 
    state, progress = ProgStateInit(solver, model, data, s; α = copy(α), β = copy(β), δ = copy(δ[1]))

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β, δ)
        best = select_rule(model, data, k, α, β, δ, s, βtmp)
        iszero(best.Δ) || apply!(model, data, best, α, β, δ, αβδ, s, βtmp)

        # progress and state
        vars = (α = copy(α), β = copy(β), δ = copy(δ[1]), k = copy(best.k), l = copy(best.l))
        update!(state, progress, solver, model, data, iter, s; vars...)
    end

    t = exact_threshold(model, data, α, β)

    return (α = copy(α), β = copy(β), δ = copy(δ[1]), t = t, state = state)
end


# TopPushK
function solve(solver::Coordinate, model::AbstractTopPushK, data::Dual{<:DTrain})
    Random.seed!(solver.seed)

    α, β, αβ, s     = initialization(model, data)
    αsum            = [sum(α)]
    βsort           = sort(β, rev = true)
    state, progress = ProgStateInit(solver, model, data, s; α = copy(α), β = copy(β))

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β)
        best = select_rule(model, data, k, α, β, s, αsum, βsort)
        iszero(best.Δ) || apply!(model, data, best, α, β, αβ, s, αsum, βsort)

        # progress and state
        vars = (α = copy(α), β = copy(β), k = copy(best.k), l = copy(best.l))
        update!(state, progress, solver, model, data, iter, s; vars...)
    end

    t = exact_threshold(model, data, α, β)

    return (α = copy(α), β = copy(β), t = copy(t), state = state)
end


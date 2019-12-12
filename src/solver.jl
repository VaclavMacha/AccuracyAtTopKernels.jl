# General solver
function solve(solver::General, model::AbstractModel, data::Primal)
    w, t = optimize(solver, model, data)
    return (w = w, t = t)
end


function solve(solver::General, model::PatMat, data::Dual{<:DTrain})
    α, β, δ = optimize(solver, model, data)
    return (α = Vector(α), β = Vector(β), δ = δ, t = exact_threshold(model, data, α, β))
end


function solve(solver::General, model::AbstractTopPushK, data::Dual{<:DTrain})
    α, β = optimize(solver, model, data)
    return (α = Vector(α), β = Vector(β), t = exact_threshold(model, data, α, β))
end


# -------------------------------------------------------------------------------
# Primal problem - gradient solver
# -------------------------------------------------------------------------------
function solve(solver::Gradient, model::AbstractModel, data::Primal, w0 = Float64[])

    w, s, Δ  = initialization(model, data, w0)
    progress = ProgressBar(solver, model, data, w, threshold(model, data, s), s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        s .= data.X * w

        # progress
        progress(solver, model, data, iter, w, threshold(model, data, s), s)

        # compute gradient and perform update
        gradient!(model, data, w, s, Δ)
        minimize!(solver, w, Δ)
    end
    return (w = w, t = threshold(model, data, data.X * w))
end


# -------------------------------------------------------------------------------
# Dual problem - gradient solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Gradient, model::PatMat, data::Dual{<:DTrain}, α0 = Float64[], β0 = Float64[])

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    Δ               = zero(αβδ)
    progress        = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        s .= data.K * vcat(α, β)

        # progress
        progress(solver, model, data, iter,  α, β, δ[1], s)

        # compute gradient and perform update
        gradient!(model, data, α, β, δ, s, Δ)
        maximize!(solver, αβδ, Δ)
        projection!(model, data, α, β, δ)
    end
    return (α = Vector(α), β = Vector(β), δ = δ[1], t = exact_threshold(model, data, α, β))
end


# TopPushK
function solve(solver::Gradient, model::AbstractTopPushK, data::Dual{<:DTrain}, α0 = Float64[], β0 = Float64[])

    α, β, αβ, s = initialization(model, data, α0, β0)
    Δ           = zero(αβ)
    progress    = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        s .= data.K * vcat(α, β)

        # progress
        progress(solver, model, data, iter,  α, β, s)

        # compute gradient and perform update
        gradient!(model, data, α, β, s, Δ)
        maximize!(solver, αβ, Δ)
        projection!(model, data, α, β)
    end
    return (α = Vector(α), β = Vector(β), t = exact_threshold(model, data, α, β))
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

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    S <: Hinge     && ( βtmp = sort(β, rev = true) )
    S <: Quadratic && ( βtmp = [sum(abs2, β)/(4*model.l2.ϑ^2)] )
 
    progress        = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β, δ)
        best = select_rule(model, data, k, α, β, δ, s, βtmp)
        iszero(best.Δ) || apply!(model, data, best, α, β, δ, αβδ, s, βtmp)

        # progress
        progress(solver, model, data, iter, α, β, δ[1], s)
    end
    return (α = Vector(α), β = Vector(β), δ = δ[1], t = exact_threshold(model, data, α, β))
end


# TopPushK
function solve(solver::Coordinate,
               model::AbstractTopPushK,
               data::Dual{<:DTrain},
               α0 = Float64[],
               β0 = Float64[])

    α, β, αβ, s = initialization(model, data, α0, β0)
    αsum        = [sum(α)]
    βsort       = sort(β, rev = true)
    progress    = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β)
        best = select_rule(model, data, k, α, β, s, αsum, βsort)
        iszero(best.Δ) || apply!(model, data, best, α, β, αβ, s, αsum, βsort)

        # progress
        progress(solver, model, data, iter,  α, β, s)
    end
    return (α = Vector(α), β = Vector(β), t = exact_threshold(model, data, α, β))
end


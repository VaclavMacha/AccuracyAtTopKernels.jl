# General solver
function solve(solver::General, model::AbstractModel, data::AbstractData)
    return optimize(solver, model, data)
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
        scores!(data, w, s)

        # progress
        progress(solver, model, data, iter, w, threshold(model, data, s), s)

        # compute gradient and perform update
        gradient!(model, data, w, s, Δ)
        minimize!(solver, w, Δ)
    end

    scores!(data, w, s)
    return w, threshold(model, data, s)
end


# -------------------------------------------------------------------------------
# Dual problem - gradient solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Gradient, model::PatMat, data::Dual, α0 = Float64[], β0 = Float64[])

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    Δ               = zero(αβδ)
    progress        = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        scores!(data, α, β, s)

        # progress
        progress(solver, model, data, iter,  α, β, δ[1], s)

        # compute gradient and perform update
        gradient!(model, data, α, β, δ, s, Δ)
        maximize!(solver, αβδ, Δ)
        projection!(model, data, α, β, δ)
    end
    return Vector(α), Vector(β), δ[1]
end


# TopPushK
function solve(solver::Gradient, model::AbstractTopPushK, data::Dual, α0 = Float64[], β0 = Float64[])

    α, β, αβ, s = initialization(model, data, α0, β0)
    Δ           = zero(αβ)
    progress    = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update score
        scores!(data, α, β, s)

        # progress
        progress(solver, model, data, iter,  α, β, s)

        # compute gradient and perform update
        gradient!(model, data, α, β, s, Δ)
        maximize!(solver, αβ, Δ)
        projection!(model, data, α, β)
    end
    return Vector(α), Vector(β)
end


# -------------------------------------------------------------------------------
# Dual problem - coordinate descent solver
# -------------------------------------------------------------------------------
# PatMat
function solve(solver::Coordinate, model::PatMat{<:S}, data::Dual, α0 = Float64[], β0 = Float64[]) where {S<:AbstractSurrogate}

    α, β, δ, αβδ, s = initialization(model, data, α0, β0)
    S <: Hinge     && ( βtmp = sort(β, rev = true) )
    S <: Quadratic && ( βtmp = [sum(abs2, β)/(4*model.l2.ϑ^2)] )
 
    progress        = ProgressBar(solver, model, data, α, β, δ[1], s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β, δ)
        best = select_rule(model, data, k, α, β, δ, s, βtmp)
        apply!(model, data, best, α, β, δ, αβδ, s, βtmp)

        # progress
        progress(solver, model, data, iter, α, β, δ[1], s)
    end
    return Vector(α), Vector(β), δ[1]
end


# TopPushK
function solve(solver::Coordinate, model::AbstractTopPushK, data::Dual, α0 = Float64[], β0 = Float64[])

    α, β, αβ, s = initialization(model, data, α0, β0)
    αsum        = [sum(α)]
    βsort       = sort(β, rev = true)
    progress    = ProgressBar(solver, model, data, α, β, s)

    # optimization
    for iter in 1:solver.maxiter
        # update coordinates
        k    = select_k(model, data, α, β)
        best = select_rule(model, data, k, α, β, s, αsum, βsort)
        apply!(model, data, best, α, β, αβ, s, αsum, βsort)

        # progress
        progress(solver, model, data, iter,  α, β, s)
    end
    return Vector(α), Vector(β)
end


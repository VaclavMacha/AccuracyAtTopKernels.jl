# General solver
function solve(solver::General, model::AbstractModel, data::AbstractData)
    return optimize(solver, model, data)
end

# Gradient solver
function solve(solver::Gradient, model::AbstractModel, data::Primal, w0 = Float64[])

    w, Δ, s  = initialization(model, data, w0)
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


function solve(solver::Gradient, model::PatMat, data::Dual, α0 = Float64[], β0 = Float64[])

    α, β, δ, αβδ, Δ, s = initialization(model, data, α0, β0)

    progress  = ProgressBar(solver, model, data, α, β, δ[1], s)

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

function solve(solver::Gradient, model::AbstractTopPushK, data::Dual, α0 = Float64[], β0 = Float64[])

    α, β, αβ, Δ, s = initialization(model, data, α0, β0)

    progress  = ProgressBar(solver, model, data, α, β, s)

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
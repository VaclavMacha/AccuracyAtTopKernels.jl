# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver
struct General <: Solver
    verbose

    function General(; verbose::Bool = false)
        new(verbose)
    end
end


function solve(solver::General, model::Model, data::Primal, w::AbstractVector = Float64[])

    w, = initialization(model, data)

    return optimize(solver, model, data, w)
end


# Our solver
struct Our{I<:Integer} <: Solver
    maxiter::I
    optimiser
    verbose

    function Our(; maxiter::I = 1000, optimiser = Optimise.ADAM(), verbose::Bool = true) where {I<:Integer}
        new{I}(maxiter, optimiser, verbose)
    end
end


function solve(solver::Our, model::M, data::Primal, w::AbstractVector = Float64[]) where {M<: Model}

    w, Δ, s  = initialization(model, data, w)
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
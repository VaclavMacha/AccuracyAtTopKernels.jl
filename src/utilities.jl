function initialization(model::Model, data::Primal, w::AbstractVector = Float64[])
    if isempty(w)
        w = zeros(eltype(data.X), data.dim)
    end
    Δ = zero(w)
    s = data.X * w
    return w, Δ, s
end

function minimize!(solver::Solver, x, Δ)
    Optimise.apply!(solver.optimiser, x, Δ)
    x .-= Δ
end

function maximize!(solver::Solver, x, Δ)
    Optimise.apply!(solver.optimiser, x, Δ)
    x .+= Δ
end


mutable struct ProgressBar
    bar::ProgressMeter.Progress
    L0::Real
    L::Real

    function ProgressBar(solver::S, model::M, data::D, args...) where {S<:Solver, M<:Model, D<:Dataset}
        msg = "$(M.name) $(D.name) problem using $(S.name) solver: "
        bar = ProgressMeter.Progress(solver.maxiter, 1, msg)
        L   = objective(model, data, args...)
        return new(bar, L, L)
    end
end 

function (progress::ProgressBar)(solver::Solver, model::Model, data::Dataset, iter::Integer, args...)
    if mod(iter, ceil(Int, solver.maxiter)) == 0
        progress.L = objective(model, data, args...)
    end
    ProgressMeter.next!(progress.bar; showvalues = [(:L0, progress.L0), (:L, progress.L)])
end

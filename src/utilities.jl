function minimize!(solver::AbstractSolver, x, Δ)
    Optimise.apply!(solver.optimizer, x, Δ)
    x .-= Δ
end

function maximize!(solver::AbstractSolver, x, Δ)
    Optimise.apply!(solver.optimizer, x, Δ)
    x .+= Δ
end


mutable struct ProgressBar
    bar::ProgressMeter.Progress
    L0::Real
    L::Real

    function ProgressBar(solver::S,
                         model::M,
                         data::D,
                         args...) where {S<:AbstractSolver, M<:AbstractModel, D<:AbstractData}
        msg = "$(M) $(D.name) loss - $(S.name) solver: "
        bar = ProgressMeter.Progress(solver.maxiter, 1, msg)
        L   = objective(model, data, args...)
        return new(bar, L, L)
    end
end 

function (progress::ProgressBar)(solver::AbstractSolver,
                                 model::AbstractModel,
                                 data::AbstractData,
                                 iter::Integer,
                                 args...)

    if mod(iter, ceil(Int, solver.maxiter/10)) == 0
        progress.L = objective(model, data, args...)
    end
    ProgressMeter.next!(progress.bar; showvalues = [(:L0, progress.L0), (:L, progress.L)])
end

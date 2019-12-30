# -------------------------------------------------------------------------------
# Progress bar utilities
# -------------------------------------------------------------------------------
mutable struct ProgressBar{P<:ProgressMeter.Progress, T<:Real}
    bar::P
    L0::T
    L::T
end 

function ProgressBar(solver::S,
                     model::M,
                     data::D,
                     args...) where {S<:AbstractSolver, M<:AbstractModel, D<:AbstractData}
    msg = "$(M.name) $(D.name) loss - $(S.name) solver: "
    bar = ProgressMeter.Progress(solver.maxiter, 1, msg)
    L   = objective(model, data, args...)
    return ProgressBar(bar, L, L)
end

function (progress::ProgressBar)(solver::AbstractSolver,
                                 model::AbstractModel,
                                 data::AbstractData,
                                 iter::Integer,
                                 args...)
    
    if solver.verbose
        if mod(iter, ceil(Int, solver.maxiter/10)) == 0
            progress.L = objective(model, data, args...)
        end
        ProgressMeter.next!(progress.bar; showvalues = [(:L0, progress.L0), (:L, progress.L)])
    end
end


# -------------------------------------------------------------------------------
# State 
# -------------------------------------------------------------------------------
struct State{T1, T2, D}
    solver::T1
    model::T2
    dict::D
end 


function State(solver::AbstractSolver,
               model::AbstractModel;
               kwargs...)

    dict = Dict(:iter_0 => values(kwargs))

    return State(convert(NamedTuple, solver), convert(NamedTuple, model), dict)
end


function (state::State)(iter::Integer;
                        kwargs...)
    
    if in(iter, state.solver.iters)
        state.dict[Symbol(:iter_, iter)] = values(kwargs)
    end
end

function (state::State)(; kwargs...)
    state.dict[Symbol(:iter_, state.solver.maxiter)] = values(kwargs)
end


# -------------------------------------------------------------------------------
# Gradient descent utilities
# -------------------------------------------------------------------------------
function minimize!(solver::AbstractSolver, x, Δ)
    Optimise.apply!(solver.optimizer, x, Δ)
    x .-= Δ
end

function maximize!(solver::AbstractSolver, x, Δ)
    Optimise.apply!(solver.optimizer, x, Δ)
    x .+= Δ
end


# -------------------------------------------------------------------------------
# Coordinate descent utilities
# -------------------------------------------------------------------------------
solution(a::Real, b::Real, Δlb::Real, Δub::Real) = min(max(Δlb, - b/a), Δub)


function update!(best::BestUpdate, k, l, Δ, L, vars)
    if L >= best.L
        best.k = k
        best.l = l
        best.Δ = Δ
        best.L = L
        best.vars = vars
    end
end


function select_rule(model::AbstractModel, data::Dual{<:DTrain}, k, args...)
    best = BestUpdate(1, 2, 0.0, -Inf, (αk = 0.0, αl = 0.0))

    for l in 1:data.n
        l == k && continue

        if k <= data.nα && l <= data.nα
            rule_αα!(model, data, best, k, l, args...)
        elseif k <= data.nα && l > data.nα
            rule_αβ!(model, data, best, k, l, args...)
        elseif k > data.nα && l <= data.nα
            rule_αβ!(model, data, best, l, k, args...)
        else
            rule_ββ!(model, data, best, k, l, args...)
        end
    end
    return best
end


function scores!(data::Dual{<:DTrain}, best::BestUpdate, s)
    if best.k <= data.nα && best.l > data.nα
        s .+= best.Δ*(data.K[:, best.k] + data.K[:, best.l])
    else 
        s .+= best.Δ*(data.K[:, best.k] - data.K[:, best.l])
    end
end


find_βmax(βsort, β, k) = βsort[1] != β[k] ? βsort[1] : βsort[2]


function find_βmax(βsort, β, k, l)
    if βsort[1] ∉ [β[k], β[l]]
        return βsort[1]
    elseif βsort[2] ∉ [β[k], β[l]]
        return βsort[2]
    else
        return βsort[3]
    end
end


function βsorted!(data::Dual{<:DTrain}, best::BestUpdate, β, βsort)
    if haskey(best.vars, :βk)
        deleteat!(βsort, searchsortedfirst(βsort, β[best.k - data.nα]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, best.vars.βk; rev = true), best.vars.βk)
    end
    if haskey(best.vars, :βl)
        deleteat!(βsort, searchsortedfirst(βsort, β[best.l - data.nα]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, best.vars.βl; rev = true), best.vars.βl)
    end
end


# -------------------------------------------------------------------------------
# Objective from named tuples
# -------------------------------------------------------------------------------
# Primal problems
function objective(model::AbstractModel, data::Primal, solution::NamedTuple)
    objective(model, data, solution.w, solution.t)
end


# Dual problems
function objective(model::PatMat, data::Dual{<:DTrain}, solution::NamedTuple)
    objective(model, data, solution.α, solution.β, solution.δ)
end


function objective(model::AbstractTopPushK, data::Dual{<:DTrain}, solution::NamedTuple)
    objective(model, data, solution.α, solution.β)
end


# -------------------------------------------------------------------------------
# Exact thresholds
# -------------------------------------------------------------------------------
function exact_threshold(model::AbstractModel, data::Dual{<:Union{DTrain, DValidation}}, w::AbstractVector, T::Real)
    exact_threshold(model, data, scores(model, data, w))
end


function exact_threshold(model::AbstractModel, data::Dual{<:Union{DTrain, DValidation}}, α::AbstractVector, β::AbstractVector)
    exact_threshold(model, data, scores(model, data, α, β))
end

# Primal problems
function exact_threshold(model::PatMat, data::Primal, s::AbstractVector)
    any(isnan.(s)) ? NaN : quantile(s, 1 - model.τ)
end


function exact_threshold(model::TopPushK, data::Primal, s::AbstractVector)
    mean(partialsort(s[data.ind_neg], 1:model.K, rev = true))
end


function exact_threshold(model::TopPush, data::Primal, s::AbstractVector)
    maximum(s[data.ind_neg])
end


# Dual problems
function exact_threshold(model::PatMat, data::Dual{<:Union{DTrain, DValidation}}, s::AbstractVector)
    any(isnan.(s)) ? NaN : quantile(s, 1 - model.τ)
end


function exact_threshold(model::TopPushK, data::Dual{<:Union{DTrain, DValidation}}, s::AbstractVector)
    mean(partialsort(s[data.type.ind_neg], 1:model.K, rev = true))
end


function exact_threshold(model::TopPush, data::Dual{<:Union{DTrain, DValidation}}, s::AbstractVector)
    maximum(s[data.type.ind_neg])
end
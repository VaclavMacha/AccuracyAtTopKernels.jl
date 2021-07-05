# -------------------------------------------------------------------------------
# Progress bar and State
# -------------------------------------------------------------------------------
mutable struct ProgressBar{P<:ProgressMeter.Progress, T<:Real}
    bar::P
    L0::T
    L::T
    L_primal::T
end


struct State{S, D<:Dict, T, A}
    seed::S
    dict::D
    time_init::T
    coordinates::A
end


function ProgStateInit(solver::S,
                       model::M,
                       data::D,
                       scores::AbstractVector;
                       kwargs...) where {S<:AbstractSolver, M<:AbstractModel, D<:AbstractData}

    msg  = "$(nameof(M)) $(nameof(D)) loss - $(nameof(S)) solver: "
    bar  = ProgressMeter.Progress(solver.maxiter, 0.1, msg)
    L    = objective(model, data, values(kwargs)..., scores)
    vals = (values(kwargs)..., time = 0, L = L)
    L_primal = Inf
    if isa(data, Dual)
        L_primal = primal_objective(model, data, values(kwargs)..., scores)
        vals = (; vals..., L_primal = L_primal)
    end
    dict = Dict{Union{Symbol, Int64}, Any}(:initial => vals)

    if S <: Coordinate
        coordinates = Array{Int64}(undef, solver.maxiter, 2)
    else
        coordinates = Int64[]
    end
    return State(solver.seed, dict, time(), coordinates), ProgressBar(bar, L, L, L_primal)
end


ProgStateInit(solver::General, model::AbstractModel, tm::Real; kwargs...) =
    State(solver.seed, Dict{Union{Symbol, Int64}, Any}(:optimal => (values(kwargs)..., time = tm)), time(), Int64[])


function update!(state::State,
                 progress::ProgressBar,
                 solver::S,
                 model::AbstractModel,
                 data::AbstractData,
                 iter::Integer,
                 scores::AbstractVector;
                 k::Int = 0,
                 l::Int = 0,
                 gap::Real = Inf,
                 kwargs...) where {S<:AbstractSolver}

    vars = values(kwargs)
    condition_1 = iter == solver.maxiter
    condition_2 = iter in solver.iters
    condition_3 = mod(iter, ceil(Int, solver.maxiter/10)) == 0 && solver.verbose

    if S <: Coordinate
        state.coordinates[iter, 1] = k
        state.coordinates[iter, 2] = l
    end

    if condition_1 || condition_2 || condition_3
        L = objective(model, data, vars..., scores)
        progress.L = L
        if isa(data, Dual)
            L_primal = primal_objective(model, data, vars..., scores)
            progress.L_primal = L_primal
        end
    end

    if solver.verbose
        vals = [(:L0, progress.L0), (:L, progress.L)]
        if isa(data, Dual)
            push!(vals, (:L_primal, progress.L_primal))
            push!(vals, (:gap, dualitygap(progress)))
        end
        ProgressMeter.next!(progress.bar; showvalues = vals)
    end

    if condition_2
        key = iter
        if isa(data, Dual)
            state.dict[key] = (vars..., time = time() - state.time_init, L = L, L_primal = L_primal)
        else
            state.dict[key] = (vars..., time = time() - state.time_init, L = L)
        end
    end
    if condition_1
        key = :optimal
        if isa(data, Dual)
            state.dict[key] = (vars..., time = time() - state.time_init, L = L, L_primal = L_primal)
        else
            state.dict[key] = (vars..., time = time() - state.time_init, L = L)
        end
    end
end

dualitygap(progress::ProgressBar) = progress.L_primal - progress.L

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
objective(model::AbstractModel, data::Primal, solution::NamedTuple) =
    objective(model, data, solution.w, solution.t)

objective(model::AbstractPatMat, data::Dual{<:DTrain}, solution::NamedTuple) =
    objective(model, data, solution.α, solution.β, solution.δ)

objective(model::AbstractTopPushK, data::Dual{<:DTrain}, solution::NamedTuple) =
    objective(model, data, solution.α, solution.β)


# -------------------------------------------------------------------------------
# Exact thresholds
# -------------------------------------------------------------------------------
exact_threshold(model::PatMat, data::Primal, s) =
    any(isnan.(s)) ? NaN : quantile(s, 1 - model.τ)

exact_threshold(model::PatMatNP, data::Primal, s) =
    any(isnan.(s)) ? NaN : quantile(s[data.ind_neg], 1 - model.τ)

function exact_threshold(model::AbstractTopPushK, data::Primal, s)
    K = getK(model, data)
    if K >= length(data.type.ind_neg)
        return mean(s[data.ind_neg])
    else
        return mean(partialsort(s[data.ind_neg], 1:K, rev = true))
    end
end

exact_threshold(model::TopPush, data::Primal, s) =
    maximum(s[data.ind_neg])

exact_threshold(model::AbstractModel, data::Primal, w, T) =
    exact_threshold(model, data, scores(model, data, w))

exact_threshold(model::PatMat, data::Dual{<:Union{DTrain, DValidation}}, s) =
    any(isnan.(s)) ? NaN : quantile(s, 1 - model.τ)

exact_threshold(model::PatMatNP, data::Dual{<:Union{DTrain, DValidation}}, s) =
    any(isnan.(s)) ? NaN : quantile(s[data.type.ind_neg], 1 - model.τ)

function exact_threshold(model::AbstractTopPushK, data::Dual{<:Union{DTrain, DValidation}}, s)
    K = getK(model, data)
    if K >= length(data.type.ind_neg)
        return mean(s[data.type.ind_neg])
    else
        return mean(partialsort(s[data.type.ind_neg], 1:K, rev = true))
    end
end

exact_threshold(model::TopPush, data::Dual{<:Union{DTrain, DValidation}}, s) =
    maximum(s[data.type.ind_neg])

exact_threshold(model::AbstractModel, data::Dual{<:Union{DTrain, DValidation}}, α, β) =
    exact_threshold(model, data, scores(model, data, α, β))

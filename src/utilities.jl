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


function select_rule(model::AbstractModel, data::Dual, k, args...)
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


function scores!(data::Dual, best::BestUpdate, s)
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


function βsorted!(data::Dual, best::BestUpdate, β, βsort)
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
# Scores
# -------------------------------------------------------------------------------
# Primal problems
function scores!(data::Primal, w::AbstractVector, s::AbstractVector)
    s .= data.X * w
end


function scores(w::AbstractVector, X::AbstractVector)
    X'*w
end


function scores(w::AbstractVector, X::AbstractMatrix)
    X*w
end


# Dual problems
function scores!(data::Dual, α::AbstractVector, β::AbstractVector, s::AbstractVector)
    s .= data.K * vcat(α, β)
end


function scores(K::AbstractMatrix, α::AbstractVector, β::AbstractVector)
    vec(vcat(α, β)'*K)
end


function scores(model::AbstractModel,
                Xtrain::AbstractMatrix,
                ytrain::BitArray{1},
                α::AbstractVector,
                β::AbstractVector,
                Xtest::AbstractMatrix = Xtrain;
                kernel::Kernel = LinearKernel())
    
    K, = kernelmatrix(model, Xtrain, ytrain, Xtest; kernel = kernel)
    return scores(K, α, β)
end


function scores(file::AbstractString, α::AbstractVector, β::AbstractVector; T::DataType = Float32)
    
    K, n, nα, nβ, io = load_kernelmatrix(file; T = T)
    s = scores(K, α, β)
    close(io)
    return s
end


# -------------------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------------------
# Primal problems
function predict(w::AbstractVector, t::Real, X)
    scores(w, t, X) .>= t 
end


# Dual problems
function predict(K::AbstractMatrix, α::AbstractVector, β::AbstractVector)
    scores(K, α, β) .>= 0
end


function predict(model::AbstractModel,
                 Xtrain::AbstractMatrix,
                 ytrain::BitArray{1},
                 α::AbstractVector,
                 β::AbstractVector,
                 Xtest::AbstractMatrix = Xtrain;
                 kernel::Kernel = LinearKernel())
    
    return scores(model, Xtrain, ytrain, α, β, Xtest; kernel = kernel) .>= 0
end


function predict(file::AbstractString, α::AbstractVector, β::AbstractVector; T::DataType = Float32)
    scores(file, α, β; T = T) .>= 0
end
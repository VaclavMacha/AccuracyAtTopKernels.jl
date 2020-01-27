# -------------------------------------------------------------------------------
# Scores
# -------------------------------------------------------------------------------
# Primal problem
function scores(model::AbstractModel, data::Primal, w::AbstractVector)
    scores(model, data.X, w)
end


function scores(model::AbstractModel, X::AbstractMatrix, w::AbstractVector)
    X * w
end


# Dual problem - train data 
function scores(model::PatMat, data::Dual{<:DTrain}, α::AbstractVector, β::AbstractVector)
    .- vec(data.K * vcat(α, β))[data.type.inv_perm]
end


function scores(model::Union{AbstractTopPushK, PatMatNP}, data::Dual{<:DTrain}, α::AbstractVector, β::AbstractVector)
    s = vec(vcat(α, β)'*data.K)
    s[data.ind_β] .*= -1
    return s[data.type.inv_perm]
end


function scores(model::AbstractModel,
                Xtrain::AbstractMatrix,
                ytrain::BitArray{1},
                α::AbstractVector,
                β::AbstractVector;
                kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain; kwargs...), α, β)
end


# Dual problem - validation and test data 
function scores(model::AbstractModel, data::Dual{<:Union{DValidation, DTest}}, α::AbstractVector, β::AbstractVector)
    vec(vcat(α, β)'*data.K)
end


function scores(model::AbstractModel,
                Xtrain::AbstractMatrix,
                ytrain::BitArray{1},
                Xvalid::AbstractMatrix,
                yvalid::BitArray{1},
                α::AbstractVector,
                β::AbstractVector;
                kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain, Xvalid, yvalid; kwargs...), α, β)
end


function scores(model::AbstractModel,
                Xtrain::AbstractMatrix,
                ytrain::BitArray{1},
                Xtest::AbstractMatrix,
                α::AbstractVector,
                β::AbstractVector;
                kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain, Xtest; kwargs...), α, β)
end


# Dual problem - load function
function scores(model::AbstractModel, file::AbstractString, α::AbstractVector, β::AbstractVector; kwargs...)

    data = Dual(file; kwargs...)
    s    = scores(model, data, α, β)
    close(data.io)
    return s
end


# -------------------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------------------
# Primal problems
function predict(model::AbstractModel, data::Primal, w::AbstractVector, t::Real)
    scores(model, data, w) .>= t 
end


function predict(model::AbstractModel, X::AbstractMatrix, w::AbstractVector, t::Real)
    scores(model, X, w) .>= t 
end


# Dual problems
function predict(model::AbstractModel, data::Dual, α::AbstractVector, β::AbstractVector, t::Real)
    scores(model, data, α, β) .>= t
end


function predict(model::AbstractModel,
                 Xtrain::AbstractMatrix,
                 ytrain::BitArray{1},
                 α::AbstractVector,
                 β::AbstractVector,
                 t::Real;
                 kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain; kwargs...), α, β) .>= t
end


function predict(model::AbstractModel,
                 Xtrain::AbstractMatrix,
                 ytrain::BitArray{1},
                 Xvalid::AbstractMatrix,
                 yvalid::BitArray{1},
                 α::AbstractVector,
                 β::AbstractVector,
                 t::Real;
                 kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain, Xvalid, yvalid; kwargs...), α, β) .>= t
end


function predict(model::AbstractModel,
                 Xtrain::AbstractMatrix,
                 ytrain::BitArray{1},
                 Xtest::AbstractMatrix,
                 α::AbstractVector,
                 β::AbstractVector,
                 t::Real;
                 kwargs...)
    
    return scores(model, Dual(model, Xtrain, ytrain, Xtest; kwargs...), α, β) .>= t
end


function predict(model::AbstractModel, file::AbstractString, α::AbstractVector, β::AbstractVector, t::Real; kwargs...)

    return  scores(model, file, α, β; kwargs...) .>= t
end
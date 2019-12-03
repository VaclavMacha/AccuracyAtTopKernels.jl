struct PatMat{T1<:Real, T2<:Real, S<:Surrogate} <: Model
    l1::S
    l2::S
    τ::T1
    C::T2

    function PatMat(l1::S, l2::S, τ::T1, C::T2) where {T1<:Real, T2<:Real, S<:Surrogate}

        @assert 0 < τ < 1  "The vaule of `τ` must lay in the interval (0,1)."

        return new{T1, T2, S}(l1, l2, τ, C)
    end
end


# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::PatMat, data::Primal)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)
    z = Convex.Variable(data.n)

    objective   = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l1.value_exact.(y))
    constraints = [Convex.sum(model.l2.value_exact.(z)) <= data.n * model.τ,
                   y == t - Xpos*w,
                   z == data.X*w - t]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = solver.verbose), verbose = solver.verbose)

    return vec(w.value), t.value
end

# Our solution
function objective(model::PatMat, data::Primal, w, t, s)
    w'*w/2 + model.C * sum(model.l1.value.(t .- s[data.pos]))
end


function threshold(model::PatMat, data::Primal, s)
   return Roots.find_zero(t -> sum(model.l2.value.(s .- t)) - data.n*model.τ, (-Inf, Inf))
end


function gradient!(model::PatMat, data::Primal, w, s, Δ)
    t   = threshold(model, data, s)
    ∇l1 = model.l1.gradient.(t .- s[data.pos])
    ∇l2 = model.l2.gradient.(s .- t)
    ∇t  = data.X'*∇l2/sum(∇l2)

    Δ .= w .+ model.C .* (sum(∇l1)*∇t .- data.X[data.pos,:]'*∇l1)
end
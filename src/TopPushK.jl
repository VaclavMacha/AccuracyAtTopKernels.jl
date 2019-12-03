struct TopPushK{I<:Integer, T<:Real, S<:Surrogate} <: Model
    l::S
    K::I
    C::T

    function TopPushK(l::S, K::I, C::T) where {I<:Integer, T<:Real, S<:Surrogate}

        @assert K >= 1 "The vaule of `K` must be greater or equal to 1."

        return new{I, T, S}(l, K, C)
    end
end


# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::TopPushK, data::Primal)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)
    z = Convex.Variable(data.nneg)

    objective   = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t + Convex.sum(z)/model.K - Xpos*w,
                   z >= Xneg*w - t,
                   z >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = solver.verbose), verbose = solver.verbose)

    return vec(w.value), t.value + sum(z.value)/model.K
end

# Our solution
function objective(model::TopPushK, data::Primal, w, t, s)
    return w'*w/2 + model.C * sum(model.l.value.(t .- s[data.pos]))
end


function threshold(model::TopPushK, data::Primal, s)
   return Statistics.mean(partialsort(s[data.neg], 1:model.K, rev = true))
end


function gradient!(model::TopPushK, data::Primal, w, s, Δ)
    ind_t = partialsortperm(s[data.neg], 1:model.K, rev = true)
    t     = Statistics.mean(s[data.neg])

    ∇l = model.l.gradient.(t .- s[data.pos])
    ∇t = vec(Statistics.mean(data.X[data.neg[ind_t], :], dims = 1))

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.pos,:]'*∇l)
end
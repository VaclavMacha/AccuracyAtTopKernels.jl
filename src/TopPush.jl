struct TopPush{T<:Real, S<:Surrogate} <: Model
    l::S
    C::T

    function TopPush(l::S, C::T) where {T<:Real, S<:Surrogate}
        return new{T, S}(l, C)
    end
end


# -------------------------------------------------------------------------------
# Primal problem
# -------------------------------------------------------------------------------
# General solver solution
function optimize(solver::General, model::TopPush, data::Primal, w::AbstractVector)

    Xpos = @view data.X[data.pos, :]
    Xneg = @view data.X[data.neg, :]

    w = Convex.Variable(data.dim)
    t = Convex.Variable()
    y = Convex.Variable(data.npos)

    objective   = Convex.sumsquares(w)/2 + model.C * Convex.sum(model.l.value_exact.(y))
    constraints = [y == t - Xpos*w,
                   t >= Convex.maximum(Xneg*w)]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = solver.verbose), verbose = solver.verbose)

    return vec(w.value), t.value
end


# Our solution
function objective(model::TopPush, data::Primal, w, t, s)
    return w'*w/2 + model.C * sum(model.l.value.(t .- s[data.pos]))
end


function threshold(model::TopPush, data::Primal, s)
   return maximum(s[data.neg])
end


function gradient!(model::TopPush, data::Primal, w, s, Δ)
    t, ind_t = findmax(s[data.neg])

    ∇l = model.l.gradient.(t .- s[data.pos])
    ∇t = vec(data.X[data.neg[ind_t], :])

    Δ .= w .+ model.C .* (sum(∇l)*∇t .- data.X[data.pos,:]'*∇l)
end
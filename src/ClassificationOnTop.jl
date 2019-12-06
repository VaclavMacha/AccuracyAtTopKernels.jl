module ClassificationOnTop

using Statistics, LinearAlgebra
import Convex, ECOS, Roots, ProgressMeter

export solve,
       AbstractSolver, General, Gradient, Coordinate,
       AbstractSurrogate, Hinge, Quadratic, Exponential,
       AbstractModel, AbstractTopPushK, PatMat, TopPushK, TopPush,
       AbstractData, Primal, Dual

import Flux.Optimise
import Flux.Optimise: Descent, ADAM, Momentum, Nesterov, RMSProp,
                      ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

abstract type AbstractSurrogate end
abstract type AbstractData end
abstract type AbstractModel end
abstract type AbstractTopPushK{AbstractSurrogate} <: AbstractModel end
abstract type AbstractSolver end

struct General{S} <: AbstractSolver
    solver::S

    function General(; solver = ECOS.ECOSSolver(verbose = false))
        new{typeof(solver)}(solver)
    end
end

struct Gradient{I<:Integer, B<:Bool, O} <: AbstractSolver
    maxiter::I
    verbose::B
    optimizer::O

    function Gradient(; maxiter::I   = 1000,
                        verbose::B   = true,
                        optimizer::O = Optimise.ADAM()) where {I<:Integer, B<:Bool, O<:Any}
        new{I, B, O}(maxiter, verbose, optimizer)
    end
end

struct Coordinate{I<:Integer, B<:Bool} <: AbstractSolver
    maxiter::Integer
    verbose::Bool

    function Coordinate(; maxiter::I = 1000,
                          verbose::B = true) where {I<:Integer, B<:Bool}
        new{I, B}(maxiter, verbose)
    end
end

mutable struct BestUpdate{I<:Integer, T<:Real}
    k::I
    l::I
    Δ::T
    L::T
    vars::NamedTuple
end

include("surrogates.jl")
include("dataset.jl")

include("PatMat.jl")
include("TopPushK.jl")

include("utilities.jl")
include("solver.jl")
include("projections.jl")


end # module
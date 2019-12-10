module ClassificationOnTop

using Statistics, LinearAlgebra
import Convex, ECOS, Roots, Mmap, ProgressMeter

export solve, scores, predict,
       AbstractSolver, General, Gradient, Coordinate,
       AbstractSurrogate, Hinge, Quadratic, Exponential,
       AbstractModel, AbstractTopPushK, PatMat, TopPushK, TopPush,
       AbstractData, Primal, Dual

import Flux.Optimise
import Flux.Optimise: Descent, ADAM, Momentum, Nesterov, RMSProp,
                      ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

import KernelFunctions 
import KernelFunctions: Kernel, LinearKernel

# types 
abstract type AbstractSurrogate end
abstract type AbstractData end
abstract type AbstractModel end
abstract type AbstractTopPushK{AbstractSurrogate} <: AbstractModel end
abstract type AbstractSolver end


struct General{S} <: AbstractSolver
    solver::S
end


function General(; solver::Any = ECOS.ECOSSolver(verbose = false))
    return General(solver)
end


struct Gradient{I<:Integer, B<:Bool, O} <: AbstractSolver
    maxiter::I
    verbose::B
    optimizer::O
end


function Gradient(; maxiter::Integer = 1000, verbose::Bool = true, optimizer::Any = Optimise.ADAM())
    return Gradient(maxiter, verbose, optimizer)
end

struct Coordinate{I<:Integer, B<:Bool} <: AbstractSolver
    maxiter::I
    verbose::B
end


function Coordinate(; maxiter::Integer = 1000, verbose::Bool = true)
    return Coordinate(maxiter, verbose)
end


mutable struct BestUpdate{I<:Integer, T<:Real}
    k::I
    l::I
    Δ::T
    L::T
    vars::NamedTuple
end


# includes
include("surrogates.jl")
include("dataset.jl")

include("PatMat.jl")
include("TopPushK.jl")

include("utilities.jl")
include("solver.jl")
include("projections.jl")
include("kernels.jl")

end # module
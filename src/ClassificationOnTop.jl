module ClassificationOnTop

using Statistics, LinearAlgebra
import Convex, ECOS, Roots, ProgressMeter

export solve,
       AbstractSolver, General, Gradient,
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

struct General <: AbstractSolver
    verbose
    optimizer

    function General(; verbose::Bool = false,
                       optimizer     = ECOS.ECOSSolver(verbose = false))
        new(verbose, optimizer)
    end
end

struct Gradient <: AbstractSolver
    maxiter
    optimizer
    verbose

    function Gradient(; maxiter::Integer = 1000,
                        optimizer        = Optimise.ADAM(),
                        verbose::Bool    = true)
        new(maxiter, optimizer, verbose)
    end
end

include("surrogates.jl")
include("dataset.jl")

include("PatMat.jl")
include("TopPushK.jl")

include("utilities.jl")
include("solver.jl")
include("projections.jl")


end # module
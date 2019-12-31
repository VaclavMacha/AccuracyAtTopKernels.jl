module ClassificationOnTop


# -------------------------------------------------------------------------------
# Used packages
# -------------------------------------------------------------------------------
using Statistics, LinearAlgebra, Random, Parameters
import Convex, Roots, Mmap, ProgressMeter
import ECOS: ECOSSolver
import Base: convert, show


import Flux.Optimise
import Flux.Optimise: Descent,
                      ADAM,
                      Momentum,
                      Nesterov,
                      RMSProp,
                      ADAGrad,
                      AdaMax,
                      ADADelta,
                      AMSGrad,
                      NADAM,
                      ADAMW,
                      RADAM

import MLKernels
import MLKernels: Orientation,
                  Kernel,
                  ExponentialKernel,
                  LaplacianKernel,
                  SquaredExponentialKernel,
                  GaussianKernel,
                  RadialBasisKernel,
                  GammaExponentialKernel,
                  RationalQuadraticKernel,
                  MaternKernel,
                  LinearKernel,
                  PolynomialKernel,
                  ExponentiatedKernel,
                  PeriodicKernel,
                  PowerKernel,
                  LogKernel,
                  SigmoidKernel


# -------------------------------------------------------------------------------
# Export 
# -------------------------------------------------------------------------------
export
    solve,
    scores,
    predict,
    exact_threshold,
    
    # Solvers
    AbstractSolver,
        General,
        Gradient,
        Coordinate,
    
    # Surrogates
    AbstractSurrogate,
        Hinge,
        Quadratic,
        Exponential,
    
    # Models
    AbstractModel,
        PatMat,
        AbstractTopPushK,
            TopPushK,
            TopPush,

    # Data        
    AbstractData,
        Primal,
        Dual,
            DTrain,
            DValidation,
            DTest, 

    # Gradient descent optimizers (reexport Flux.Optimise)
    Descent,
    ADAM,
    Momentum,
    Nesterov,
    RMSProp,
    ADAGrad,
    AdaMax,
    ADADelta,
    AMSGrad,
    NADAM,
    ADAMW,
    RADAM,

    # Kernels (reexport MLKernels)
    Kernel,
    ExponentialKernel,
    LaplacianKernel,
    SquaredExponentialKernel,
    GaussianKernel,
    RadialBasisKernel,
    GammaExponentialKernel,
    RationalQuadraticKernel,
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    ExponentiatedKernel,
    PeriodicKernel,
    PowerKernel,
    LogKernel,
    SigmoidKernel


# -------------------------------------------------------------------------------
# types  
# -------------------------------------------------------------------------------
abstract type AbstractSurrogate end
abstract type AbstractData end
abstract type AbstractModel end
abstract type AbstractTopPushK{AbstractSurrogate} <: AbstractModel end
abstract type AbstractSolver end


@with_kw_noshow struct General{I<:Integer,S} <: AbstractSolver
    seed::I   = rand(1:10000)
    solver::S = ECOSSolver(verbose = false)
end


show(io::IO, solver::General) =
    print(io, "General($(typeof(solver.solver).name))")


@with_kw_noshow struct Gradient{I<:Integer, B<:Bool, O, V} <: AbstractSolver
    seed::I      = rand(1:10000)
    maxiter::I   = 1000
    verbose::B   = true
    optimizer::O = ADAM()
    iters::V     = Int[]
end


function optimizername(opt::O) where {O} 
    args = [getfield(opt, key) for key in fieldnames(O) if !(fieldtype(O, key) <: IdDict)]
    return "$(O.name)($(join(args, ",")))"
end


show(io::IO, solver::Gradient) =
    print(io, "Gradient($(solver.maxiter), $(optimizername(solver.optimizer)))")


@with_kw_noshow  struct Coordinate{I<:Integer, B<:Bool, V} <: AbstractSolver
    seed::I    = rand(1:10000)
    maxiter::I = 1000
    verbose::B = true
    iters::V   = Int[]
end


show(io::IO, solver::Coordinate) =
    print(io, "Coordinate($(solver.maxiter))")


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
include("scores_predict.jl")

end # module
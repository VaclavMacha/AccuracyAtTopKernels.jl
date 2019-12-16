module ClassificationOnTop


# -------------------------------------------------------------------------------
# Used packages
# -------------------------------------------------------------------------------
using Statistics, LinearAlgebra
import Convex, ECOS, Roots, Mmap, ProgressMeter

import Flux.Optimise
import Flux.Optimise: Descent, ADAM, Momentum, Nesterov, RMSProp,
                      ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

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
include("scores_predict.jl")

end # module
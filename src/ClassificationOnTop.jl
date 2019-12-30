module ClassificationOnTop


# -------------------------------------------------------------------------------
# Used packages
# -------------------------------------------------------------------------------
using Statistics, LinearAlgebra, Random
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


struct General{I<:Integer,S} <: AbstractSolver
    seed::I
    solver::S
end


function General(; solver::Any   = ECOSSolver(verbose = false),
                   seed::Integer = rand(1:10000))
    return General(seed, solver)
end


function show(io::IO, solver::General)
    print(io, "General(", typeof(solver.solver), ")")
end


function convert(::Type{NamedTuple}, solver::General)
    (solver    = "General",
     optimizer = string(typeof(solver.solver).name),
     seed      = solver.seed)
end



struct Gradient{I<:Integer, B<:Bool, O, V} <: AbstractSolver
    seed::I
    maxiter::I
    verbose::B
    optimizer::O
    iters::V
end


function Gradient(; maxiter::Integer      = 1000,
                    verbose::Bool         = true,
                    optimizer::Any        = Optimise.ADAM(),
                    iters::AbstractVector = [],
                    seed::Integer         = rand(1:10000))

    return Gradient(seed, maxiter, verbose, optimizer, iters)
end


function show(io::IO, solver::Gradient)
    opt  = solver.optimizer
    T    = typeof(opt)
    vals = [getfield(opt, field) for field in fieldnames(T) if !(fieldtype(T, field) <: IdDict)]
    name = string(T.name, "(", join(vals, ","), ")")

    print(io, "Gradient(", join([name, solver.maxiter], ","), ")")
end


function convert(::Type{NamedTuple}, solver::Gradient)
    (solver    = "Gradient",
     optimizer = string(typeof(solver.optimizer).name),
     maxiter   = solver.maxiter,
     iters     = solver.iters,
     seed      = solver.seed)
end


struct Coordinate{I<:Integer, B<:Bool, V} <: AbstractSolver
    seed::I
    maxiter::I
    verbose::B
    iters::V
end


function Coordinate(; maxiter::Integer      = 1000,
                      verbose::Bool         = true,
                      iters::AbstractVector = [],
                      seed::Integer         = rand(1:10000))

    return Coordinate(seed, maxiter, verbose, iters)
end


function show(io::IO, solver::Coordinate)
    print(io, "Coordinate(", solver.maxiter, ")")
end


function convert(::Type{NamedTuple}, solver::Coordinate)
    (solver    = "Coordinate",
     maxiter   = solver.maxiter,
     iters     = solver.iters,
     seed      = solver.seed)
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
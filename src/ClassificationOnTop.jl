module ClassificationOnTop


# -------------------------------------------------------------------------------
# Used packages
# -------------------------------------------------------------------------------
using Statistics, LinearAlgebra, Random, Parameters
import Convex, Roots, Mmap, ProgressMeter
import Base: convert, show

import ECOS
import ECOS: ECOSSolver

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
    ECOSSolver,

    # Surrogates
    AbstractSurrogate,
        Hinge,
        Quadratic,
        Exponential,

    # Models
    AbstractModel,
        AbstractPatMat,
          PatMat,
          PatMatNP,
        AbstractTopPushK,
            TopPushK,
            TopPush,
            τFPL,

    # Data
    AbstractData,
        Primal,
        Dual,
            DTrain,
            DValidation,
            DTest,

    # Gradient descent optimizers (reexport Flux.Optimise)
    AbstractOptimizer,
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
abstract type AbstractPatMat{AbstractSurrogate}   <: AbstractModel end
abstract type AbstractTopPushK{AbstractSurrogate} <: AbstractModel end
abstract type AbstractSolver end

AbstractOptimizer = Union{Descent, ADAM, Momentum, Nesterov, RMSProp, ADAGrad,
                          AdaMax, ADADelta, AMSGrad, NADAM, RADAM}


function show(io::IO, opt::O) where {O<:AbstractOptimizer}
    args   = [getfield(opt, field) for field in fieldnames(O) if !in(field, (:velocity, :acc, :state))]
    print(io, "$(O.name)($(join(args, ",")))")
end


@with_kw_noshow struct General{I<:Integer,S} <: AbstractSolver
    seed::I   = rand(1:10000)
    solver::S = ECOSSolver(verbose = false)
end


show(io::IO, solver::General) =
    print(io, "General($(typeof(solver.solver).name))")


@with_kw_noshow struct Gradient{I<:Integer, B<:Bool, O<:AbstractOptimizer, V} <: AbstractSolver
    seed::I      = rand(1:10000)
    maxiter::I   = 1000
    verbose::B   = true
    optimizer::O = ADAM()
    iters::V     = Int[]
end


show(io::IO, solver::Gradient) =
    print(io, "Gradient($(solver.maxiter), $(solver.optimizer))")


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

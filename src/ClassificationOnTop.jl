module ClassificationOnTop

using Statistics, LinearAlgebra
import Convex, ECOS, Roots, ProgressMeter

export solve,
       Solver, General, Our,
       Surrogate, Hinge, Quadratic, Exponential,
       Model, PatMat, TopPushK, TopPush,
       Dataset, Primal, Dual

import Flux.Optimise
import Flux.Optimise: Descent, ADAM, Momentum, Nesterov, RMSProp,
                      ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

export Descent, ADAM, Momentum, Nesterov, RMSProp,
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM

abstract type Model end
abstract type Dataset end
abstract type Solver end
abstract type Surrogate end

include("surrogates.jl")
include("dataset.jl")
include("utilities.jl")
include("solver.jl")

include("PatMat.jl")
include("TopPush.jl")
include("TopPushK.jl")

end # module
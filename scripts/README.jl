# ##  AccuracyAtTopKernels.jl
# This repository is a complementary material to our paper ["General Framework for Nonlinear Binary Classification on Top Samples"]().
#
# # Installation
#
# This package can be installed using pkg REPL as follows
# ```julia
# (v1.2) pkg> add https://github.com/VaclavMacha/AccuracyAtTopKernels_new.jl
# ```

# # Usage
#
# This package provides following methods:
# * Methods:
#     1. `TopPush(C, surrogate)`
#     2. `TopPushK(K, C, surrogate)`
#     3. `PatMat(τ C, surrogate, surrogate)`
#     4. `PatMatNP(τ C, surrogate, surrogate)`
# * Surrogates:
#     1. `Hinge(ϑ)`
#     2. `Quadratic(ϑ)`
# * Problems:
#     1. primal
#     2. dual
# * Solvers:
#     1. `General`
#     2. `Gradient`
#     3. `Coordinate`
#
# ## Simple example
# 
# In this example we use our package and StatsPlots to visualize the solution.
using AccuracyAtTopKernels, StatsPlots, Random

Random.seed!(1234)

function random_circle(n::Int; radius::Tuple = (0,1), origin::Tuple = (0,0))
    rmin, rmax = round.(Int, radius.^2)
    r2 = (rmax - rmin) .* rand(n) .+ rmin
    θ  = 2π .* rand(n)

    x = origin[1] .+ sqrt.(r2) .* cos.(θ) 
    y = origin[2] .+ sqrt.(r2) .* sin.(θ)
    return x, y
end

nneg, npos = 1000, 200
xneg, yneg = random_circle(nneg; radius = (3, 5));
xpos, ypos = random_circle(npos; radius = (0, 2));

X = [xneg yneg; xpos ypos];
y = 1:(nneg+npos) .> nneg;

#  As an example, we use the following linearly nonseparable data

plt1 = scatter(X[y .== 0, 1], X[y .== 0, 2], label = "positives")
scatter!(X[y .== 1, 1], X[y .== 1, 2], label = "negatives", dpi = 300)
savefig(plt1, "data.png")

# ![Simple example](./scripts/data.png)

# The following example shows, how to compute the primal and dual problem for ToppushK method. To solve, we use our gradient descent based solver for the primal problem and our coordinate descent based solver for the dual problem. 

model  = TopPushK(10, 1, Quadratic(1))
data_p = Primal(X, y);
data_d = Dual(model, X, y; kernel = GaussianKernel());

solution_p = solve(Gradient(maxiter = 1000, optimizer = Descent(0.0001)), model, data_p);
solution_d = solve(Coordinate(maxiter = 1000), model, data_d);

# For simple visual verification that methods works, we use density estimates of classification scores.
scores_p = scores(model, data_p, solution_p.w);
scores_d = scores(model, data_d, solution_d.α, solution_d.β);

plt_p = density(scores_p[y .== 0]; trim = true, fill = (0, 0.15), title = "primal", legend = false)
density!(scores_p[y .== 1]; trim = true, fill = (0, 0.15))
plt_d = density(scores_d[y .== 0]; label = "negatives", trim = true, fill = (0, 0.15), title = "dual with kernel")
density!(scores_d[y .== 1]; label = "positives", trim = true, fill = (0, 0.15))
plt = plot(plt_p, plt_d, layout = (1,2), size = (800, 400), dpi = 300)
savefig(plt, "scores.png")

# ![Comparison of scores.](./scripts/scores.png)
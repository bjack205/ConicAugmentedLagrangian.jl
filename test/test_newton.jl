using StaticArrays

include("../src/newton_solver.jl")
include("../src/augmented_lagrangian.jl")
include("../src/di_problem.jl")

# rosenbrock
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [2,2.]
x,stats = newton_solver_AD(rosenbrock, x0, ϵ=1e-10, verbose=true, newton_iters=20)
norm(x - [1,1]) < 1e-10
rosenbrock(x) < 1e-10
stats.iters

# hs50
hs50_obj(x) = (x[1]-x[2])^2 + (x[2]-x[3])^2 + (x[3]-x[4])^4 + (x[4]-x[5])^2
hs50_con(x) = SA[
    x[1] + 2*x[2] + 3*x[3] - 6,
    x[2] + 2*x[3] + 3*x[4] - 6,
    x[3] + 2*x[4] + 3*x[5] - 6
]

x0 = rand(5)
λ0 = zeros(3)
x,y = augmented_lagrangian_AD(x0, λ0, hs50_obj, hs50_con, verbose=false)
hs50_obj(x) < 1e-10
norm(hs50_con(x),Inf) < 1e-10

# double integrator
D,N = 2,11
di_obj, di_con = DoubleIntegrator(D,N)
n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
augmented_lagrangian_AD(x0, λ0, di_obj, di_con)
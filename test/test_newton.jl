using StaticArrays

include("../src/newton_solver.jl")
include("../src/augmented_lagrangian.jl")
include("../src/di_problem.jl")

## rosenbrock
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [2,2.]
x,stats = newton_solver_AD(rosenbrock, x0, ϵ=1e-10, verbose=true, newton_iters=20)
norm(x - [1,1]) < 1e-10
rosenbrock(x) < 1e-10
stats.iters

## hs50
hs50_obj(x) = (x[1]-x[2])^2 + (x[2]-x[3])^2 + (x[3]-x[4])^4 + (x[4]-x[5])^2
hs50_con(x) = SA[
    x[1] + 2*x[2] + 3*x[3] - 6,
    x[2] + 2*x[3] + 3*x[4] - 6,
    x[3] + 2*x[4] + 3*x[5] - 6
]

x0 = rand(5)
x,y = augmented_lagrangian_AD(x0, hs50_obj, hs50_con, verbose=false)
hs50_obj(x) < 1e-10
norm(hs50_con(x),Inf) < 1e-10

## double integrator
D,N = 2,11
di_obj, di_con = DoubleIntegrator(D,N)
n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
x, = augmented_lagrangian_AD(x0, di_obj, di_con)
u = reshape(x,:,N)[2D+1:end,:]
unorm = norm.(eachcol(u))

## socp
n = 15
m = 5 
P = Diagonal(ones(n))
A = rand(m,n)
b = rand(m)
lin_soc_obj(x) = x'P*x
lin_soc_con(x) = A*x + b
lin_soc_q(x) = [SA[x[1], x[2], x[3], x[4], 0.1]]

x0 = rand(n)
x, = augmented_lagrangian_AD(x0, lin_soc_obj, lin_soc_con, lin_soc_q)
norm(x[1:4])

## double integrator w/ socp
D,N = 2,11
di_obj, di_con = DoubleIntegrator(D,N)
di_q(x) = [
    SA[x[5], x[6], 6.0], 
    SA[x[11], x[12], 6.0], 
    SA[x[53], x[54], 6.0], 
    SA[x[59], x[60], 6.0]
]
function di_q(x)
    x_ = reshape(x,:,N)
    us = x_[2D+1:end,:]
    [[u; 6.0] for u in eachcol(us)]
end
n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
x,y,z = augmented_lagrangian_AD(x0, di_obj, di_con, di_q)
z .- Πsoc.(z)
u = reshape(x,:,N)[2D+1:end,:]
unorm = norm.(eachcol(u))
abs(unorm[1] - 6)  < 1e-6
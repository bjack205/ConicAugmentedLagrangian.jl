using StaticArrays

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
di_obj, di_con = DoubleIntegratorFuns(D,N)
n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
x, = augmented_lagrangian_AD(x0, di_obj, di_con)
u = reshape(x,:,N)[2D+1:end,:]
unorm = norm.(eachcol(u))

prob = DoubleIntegrator(D,N)
x, = augmented_lagrangian(prob, x0)

## socp
n = 15
m = 5 
P = Diagonal(ones(n))
A = rand(m,n)
b = rand(m)
lin_soc_obj(x) = x'P*x
lin_soc_con(x) = A*x + b
lin_soc_q(x) = [SA[x[1], x[2], x[3], x[4], 0.1]]
qinds = [[1,2,3,4]]

x0 = rand(n)
x, = augmented_lagrangian_AD(x0, lin_soc_obj, lin_soc_con, lin_soc_q, qinds)
norm(x[1:4])

## double integrator w/ socp
D,N = 2,11
prob = DoubleIntegrator(D,N, qinds=:control, s=fill(6.0, N))
n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
x,y,z = augmented_lagrangian(prob, x0)
z .- Πsoc.(z)
u = reshape(x,:,N)[2D+1:end,:]
unorm = norm.(eachcol(u))
abs(unorm[1] - 6)  < 1e-6

qinds = [vec(LinearIndices(zeros(3D,N))[1+2D:3D,:])]
prob = DoubleIntegrator(D,N, qinds=qinds, conetype=NegativeOrthant())
x0 = rand(n)
x,y,z = augmented_lagrangian(prob, x0)
u = reshape(x,:,N)[2D+1:end,:]

## Rocket Landing Problem
D = 3
N = 11
prob = RocketLanding(N, qinds=:control, s=fill(400,N))

n = 3D*N
m = 2D*N
x0 = rand(n)
λ0 = zeros(m)
x,y,z = augmented_lagrangian(prob, x0,
    ϵ=1e-2, ϵ_feas=1e-6, verbose=false
)
u = reshape(x,:,N)[2D+1:end,:]
unorm = norm.(eachcol(u))
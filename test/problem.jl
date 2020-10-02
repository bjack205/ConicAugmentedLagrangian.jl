using StaticArrays
using ForwardDiff
include("../src/problem.jl")
include("../src/di_problem.jl")

hs50_obj(x) = (x[1]-x[2])^2 + (x[2]-x[3])^2 + (x[3]-x[4])^4 + (x[4]-x[5])^2
hs50_con(x) = SA[
    x[1] + 2*x[2] + 3*x[3] - 6,
    x[2] + 2*x[3] + 3*x[4] - 6,
    x[3] + 2*x[4] + 3*x[5] - 6
]
prob = ADProblem(5, hs50_obj, hs50_con)
x = rand(5)

cost(prob, x) == hs50_obj(x)
con_eq(prob, x) == hs50_con(x)
jac_eq(prob, x) ≈ ForwardDiff.jacobian(hs50_con, x)
grad_obj(prob, x) ≈ ForwardDiff.gradient(hs50_obj, x)
hess_obj(prob, x) ≈ ForwardDiff.hessian(hs50_obj, x)

# Double Integrator
D,N = 2,11
prob = DoubleIntegrator(D,N)
di_obj, di_con = DoubleIntegratorFuns(D,N)
x = rand(num_vars(prob))
cost(prob,x) ≈ di_obj(x)
ForwardDiff.gradient(di_obj, x) ≈ grad_obj(prob, x)
ForwardDiff.hessian(di_obj, x) ≈ hess_obj(prob, x)

con_eq(prob,x) ≈ di_con(x)
jac_eq(prob,x) ≈ ForwardDiff.jacobian(di_con, x)


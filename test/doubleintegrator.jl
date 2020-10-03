using ForwardDiff
include("../src/augmented_lagrangian.jl")
include("../src/di_problem.jl")

function Πsoc(x)
    v = x[1:end-1]
    s = x[end]
    a = norm(v)
    if a <= -s
        return zero(x)
    elseif a <= s
        return x
    elseif a >= abs(s)
        x̄ = append!(v, a)
        return 0.5*(1 + s/a) * x̄
    end
    throw(ErrorException("Invalid second-order cone"))
end

function auglag(f,h,g, x,y,z,μ)
    ceq = h(x)
    L0 = f(x) + y'ceq + 0.5*μ*ceq'ceq

    p = num_cones(prob)
    cones = con_soc(prob, x)
    p = length(cones)
    for i = 1:p
        cone = cones[i]
        proj = Πsoc(z[i] - μ*cones[i])
        pen = proj'proj - z[i]'z[i]
        L0 += 1 / (2μ) * pen
    end
    return L0
end

##
D,N = 2,11
n,m = 2D,D
x0 = @SVector fill(0., n) 
xf = [(@SVector fill(1.,D)); (@SVector fill(0, D))]
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-1, m))
Qf = (N-1)*Q * 100
qinds = DI_cones(D,N)
u_bnd = 6.0

##
H_obj,g_obj,c_obj = DI_objective(Q,R,Qf,xf)
Adyn,bdyn = DI_dynamics(x0,N)

di_obj(x) = 0.5*x'H_obj*x + g_obj'x + c_obj
di_dyn(x) = Adyn*x + bdyn
di_soc(x) = [push(x[qi],u_bnd) for qi in qinds]

##
prob = DoubleIntegrator(D,N; Qd=Q.diag, Rd=R.diag, Qfd=Qf.diag, x0=x0, xf=xf, 
    qinds=:control, s=fill(u_bnd, N))
NN = num_vars(prob)
P = num_cons(prob)
x = rand(NN)
obj(prob,x) ≈ di_obj(x)
grad_obj(prob,x) ≈ ForwardDiff.gradient(di_obj,x)
con_eq(prob,x) ≈ di_dyn(x)
con_soc(prob,x) ≈ di_soc(x)

y = rand(P)
z = rand.(length.(di_soc(x)))
μ = 2.5
alprob = ALProblem(prob, y, z, μ)
obj(alprob, x) ≈ auglag(di_obj, di_dyn, di_soc, x, y, z, μ)
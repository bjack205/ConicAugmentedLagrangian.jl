using ForwardDiff
include("../src/augmented_lagrangian.jl")
include("../src/di_problem.jl")

function auglag(prob, x,y,z,μ)
    ceq = con_eq(prob,x)
    L0 = cost(prob, x) + y'ceq + 0.5*μ*ceq'ceq

    p = num_cones(prob)
    if p > 0
        # csoc = [Πsoc(zi - μ*q) for (zi,q) in zip(z, q(x))]
        cones = con_soc(prob, x)
        csoc = [Πsoc(z[i] - μ*cones[i]) for i = 1:p]
        pen = [csoc[i]'csoc[i] - z[i]'z[i] for i = 1:p]
        L0 += 1 / (2μ) * sum(pen)
    end
    return L0
end

function grad_LA(prob, x,y,z,μ)
    ceq = con_eq(prob,x)
    ∇f = grad_obj(prob, x)
    ∇ceq = jac_eq(prob, x)
    grad = ∇f + ∇ceq'*(y + μ*ceq)
    
    p = num_cones(prob)
    if p > 0
        cones = con_soc(prob, x)
        pi = length(cones[1])
        ∇cone = Matrix(I,pi,pi)[:,1:end-1]   # derivative of cone wrt x
        for i = 1:p
            qi = prob.qinds[i]
            csoc = Πsoc(z[i] - μ*cones[i])
            ∇csoc = jac_soc(z[i] - μ*cones[i])*(-μ*∇cone)
            ∇pen = ∇csoc'csoc / μ
            grad[qi] += ∇pen
        end
    end
    return grad
end

function hess_LA(prob, x,y,z,μ)
    hess_f = hess_obj(prob,x)
    ∇ceq = jac_eq(prob, x)
    p = num_cones(prob)
    hess = hess_f + μ*∇ceq'∇ceq

    if p > 0
        cones = con_soc(prob, x)
        pi = length(cones[1])
        ∇cone = Matrix(I,pi,pi)[:,1:end-1]   # derivative of cone wrt x
        for i = 1:p
            qi = prob.qinds[i]
            csoc = Πsoc(z[i] - μ*cones[i])
            ∇csoc = jac_soc(z[i] - μ*cones[i])*(-μ*∇cone)
            ∇²csoc = μ^2*∇cone'hess_soc(z[i] - μ*cones[i], csoc)*∇cone  # plus the 2nd order cone function 
            ∇pen = (∇csoc'∇csoc + ∇²csoc) / μ
            hess[qi,qi] += ∇pen
        end
    end
    return hess
end

# Test derivatives w/o cones
D,N = 2,11
prob = DoubleIntegrator(D,N, gravity=true)
x = rand(num_vars(prob))
cones = con_soc(prob, x)

y = rand(num_cons(prob))
z = rand.(cones)
μ = rand()*10
auglag(prob,x,y,z,μ)

LA(x) = auglag(prob, x,y,z,μ)
ForwardDiff.gradient(LA, x) ≈ grad_LA(prob, x,y,z,μ)
ForwardDiff.hessian(LA, x) ≈ hess_LA(prob,x,y,z,μ)

# Add in cones
qinds = [z[1+2D:end] for z in eachcol(LinearIndices(zeros(3D,N)))]
prob = DoubleIntegrator(D,N, gravity=true, qinds=qinds)
cones = con_soc(prob, x)
z = rand.(length.(cones)) 
num_cones(prob) == N

t = 1.0
i = 1
jac_soc.(cones) ≈ ForwardDiff.jacobian.(Πsoc, cones)
pen(x) = Πsoc(z[i] - μ*[x;t])
u1 = x[1+2D:3D]
∇cone = Matrix(I,D+1,D+1)[:,1:end-1]
ForwardDiff.jacobian(pen, u1) ≈ -μ*jac_soc(z[i] - μ*cones[i])*∇cone

# Move it outside the cone
u1 *= 10
cones[i] = [u1;1]
norm(u1) > 1
ForwardDiff.jacobian(pen, u1) ≈ -μ*jac_soc(z[i] - μ*cones[i])*∇cone
ForwardDiff.jacobian(pen, u1) ≈ -μ*jac_soc(z[i] - μ*cones[i])[:,1:end-1]

# Test the augmented Lagrangian expansion
x = rand(num_vars(prob)) * 10  # make it go outside the cone
ForwardDiff.gradient(LA, x) ≈ grad_LA(prob, x,y,z,μ)
ForwardDiff.hessian(LA, x) ≈ hess_LA(prob, x,y,z,μ)

# Test soc projection derivatives
v = rand(4)
b = rand(5)
s = norm(v) * 0.9
a = [v;s]
Πsoc(a) ≈ 0.5*[
    v+s*v/norm(v); 
    norm(v) + s
]
jac_soc(a) ≈ 0.5*[
    I + s*(I - (v*v')/(v'v))/norm(v)     v/norm(v);
    v'/norm(v)                            1
]

bv = b[1:end-1]
bs = b[end]
jac_soc(a)'b ≈ 0.5*[
    bv + s/norm(v)*(I - (v*v')/(v'v))*bv + v/norm(v)*bs;
    v'bv/norm(v) + bs
]

ForwardDiff.jacobian(x->jac_soc(x)'b, a) ≈ hess_soc(a,b) 


# Test norm cone
x = rand(5) 
b = rand(5)
t = norm(x) * 0.8
cone = NormCone(t)
projection(cone, x) == x/norm(x)*t
jacobian(cone, x) ≈ ForwardDiff.jacobian(x->projection(cone,x), x)
hessian(cone, x, b) ≈ ForwardDiff.jacobian(x->jacobian(cone,x)'b, x)
hessian(cone, x, b) ≈ ForwardDiff.hessian(x->projection(cone,x)'b, x)
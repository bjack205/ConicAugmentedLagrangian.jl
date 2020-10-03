using ForwardDiff
using SparseArrays

abstract type ProblemDef end

num_vars(::ProblemDef) = throw(ErrorException("Not implemented"))
num_cons(::ProblemDef) = throw(ErrorException("Not implemented"))
num_cones(::ProblemDef) = throw(ErrorException("Not implemented"))
get_cones(::ProblemDef) = throw(ErrorException("Not implemented"))
obj(::ProblemDef, x) = throw(ErrorException("Not implemented"))
con_eq!(::ProblemDef, ceq, x) = throw(ErrorException("Not implemented"))
con_soc!(::ProblemDef, cones, x) = throw(ErrorException("Not implemented"))

function con_eq(prob::ProblemDef, x)
    ceq = zeros(num_cons(prob))
    con_eq!(prob, ceq, x)
end

function con_soc(prob::ProblemDef, x)
    cones = zeros.(num_cones(prob))
    con_eq!(prob, cones, x)
end

function grad_obj(prob::ProblemDef, x)
    grad = zero(x)
    grad_obj!(prob, grad, x)
end

function grad_obj!(prob::ProblemDef, grad, x)
    f(x) = obj(prob, x)
    ForwardDiff.gradient!(grad, f, x)
end

function hess_obj(prob::ProblemDef, x)
    n = length(x)
    if num_vars(prob) > 100
        H = spzeros(eltype(x), n, n)
    else
        H = zeros(eltype(x), n, n)
    end
    hess_obj!(prob::ProblemDef, H, x)
end

function hess_obj!(prob::ProblemDef, H, x)
    f(x) = obj(prob, x)
    ForwardDiff.hessian!(H, f, x)
end

function jac_eq(prob::ProblemDef, x)
    c(x) = con_eq(prob, x)
    ForwardDiff.jacobian(c, x)
end

function jac_soc(prob::ProblemDef, x)
    c(x) = con_soc(prob, x)
    jac = ForwardDiff.jacobian.(c, x) 
end


struct ADProblem <: ProblemDef
    n::Int
    m::Int
    p::Int
    ps::Vector{Int}
    f::Function   # objective
    h::Function   # equality constraints
    q::Function   # conic constraints
    qinds::Vector{Vector{Int}}
    function ADProblem(n::Int, f::Function, h::Function=x->zeros(0), 
            q::Function=x->Vector{Float64}[], qinds=Vector{Int}[])
        x = rand(n)
        @assert f(x) isa Real
        @assert h(x) isa AbstractVector{<:Real}
        @assert q(x) isa AbstractVector{<:AbstractVector{<:Real}}
        @assert length(q(x)) == length(qinds)
        m = length(h(x))
        p = length(q(x))   # number of cones
        ps = length.(q(x))  # cone sizes
        if p > 0
            @assert ps == fill(ps[1],p) 
        end
        new(n, m, p, ps, f, h, q, qinds)
    end
end

num_vars(prob::ADProblem) = prob.n
num_cons(prob::ADProblem) = prob.m
num_cones(prob::ADProblem) = prob.p 
get_cones(prob::ADProblem) = [SecondOrderCone() for i = 1:prob.p]
obj(prob::ADProblem, x) = prob.f(x)
con_eq(prob::ADProblem, x) = prob.h(x)
con_soc(prob::ADProblem, x) = prob.q(x)



mutable struct ALProblem{P,T} <: ProblemDef
    prob::P
    y::Vector{T}
    z::Vector{Vector{T}}
    μ::T
end

function obj(alprob::ALProblem, x)
    y,z = alprob.y, alprob.z
    μ = alprob.μ
    prob = alprob.prob
    ceq = con_eq(prob,x)
    L0 = obj(prob, x) + y'ceq + 0.5*μ*ceq'ceq

    p = num_cones(prob)
    conevals = con_soc(prob, x)
    cone = get_cones(prob)
    for i = 1:p
        g = conevals[i]
        csoc = projection(cone[i], z[i] - μ*g)
        pen = csoc'csoc - z[i]'z[i]
        L0 += 1 / (2μ) * pen 
    end
    return L0
end

num_vars(prob::ALProblem) = num_vars(prob.prob)
num_cons(prob::ALProblem) = num_cons(prob.prob)
num_cones(prob::ALProblem) = num_cones(prob.prob)
get_cones(prob::ALProblem) = get_cones(prob.prob)
con_eq(prob::ALProblem) = con_eq(prob.prob)
con_soc(prob::ALProblem) = con_soc(prob.prob)

function grad_obj(alprob::ALProblem, x)
    prob = alprob.prob
    y,z,μ = alprob.y, alprob.z, alprob.μ
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

function hess_obj(alprob::ALProblem, x)
    prob = alprob.prob
    y,z,μ = alprob.y, alprob.z, alprob.μ

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
            ∇csoc = -jac_soc(z[i] - μ*cones[i])*∇cone
            ∇²csoc = ∇cone'hess_soc(z[i] - μ*cones[i], csoc)*∇cone  # plus the 2nd order cone function 
            ∇pen = (∇csoc'∇csoc + ∇²csoc) * μ
            hess[qi,qi] += ∇pen
        end
    end
    return hess
end
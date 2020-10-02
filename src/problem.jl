using ForwardDiff
using SparseArrays

abstract type ProblemDef end

num_vars(::ProblemDef) = throw(ErrorException("Not implemented"))
num_cons(::ProblemDef) = throw(ErrorException("Not implemented"))
num_cones(::ProblemDef) = throw(ErrorException("Not implemented"))
cost(::ProblemDef, x) = throw(ErrorException("Not implemented"))
con_eq!(::ProblemDef, ceq, x) = throw(ErrorException("Not implemented"))
con_soc!(::ProblemDef, cones, x) = throw(ErrorException("Not implemented"))

function con_eq(prob::ProblemDef, x)
    ceq = zeros(num_cons(prob))
    con_eq!(prob, ceq, x)
end

function con_soc(prob::ProblemDef, x)
    cones = zeros.(num_cones(prob))
    con_eq!(prob, ceq, x)
end

function grad_obj(prob::ProblemDef, x)
    grad = zero(x)
    grad_obj!(prob, grad, x)
end

function grad_obj!(prob::ProblemDef, grad, x)
    f(x) = cost(prob, x)
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
    f(x) = cost(prob, x)
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
    function ADProblem(n::Int, f::Function, h::Function=x->zeros(0), q::Function=x->Vector{Float64}[])
        x = rand(n)
        @assert f(x) isa Real
        @assert h(x) isa AbstractVector{<:Real}
        @assert q(x) isa AbstractVector{<:AbstractVector{<:Real}}
        m = length(h(x))
        p = length(q(x))   # number of cones
        ps = length.(q(x))  # cone sizes
        new(n, m, p, ps, f, h, q)
    end
end

num_vars(prob::ADProblem) = prob.n
num_cons(prob::ADProblem) = prob.m
num_cones(prob::ADProblem) = prob.p 
cost(prob::ADProblem, x) = prob.f(x)
con_eq(prob::ADProblem, x) = prob.h(x)
con_soc(prob::ADProblem, x) = prob.q(x)
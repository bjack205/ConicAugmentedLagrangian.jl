using LinearAlgebra
using StaticArrays

struct DoubleIntegrator{n,m,p,T} <: ProblemDef
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    Qf::Diagonal{T,SVector{n,T}}
    x0::SVector{n,T}
    xf::SVector{n,T}
    gravity::Bool
    xinds::Vector{SVector{n,Int}}
    uinds::Vector{SVector{m,Int}}
    qinds::Vector{SVector{p,Int}}
    s::Vector{T}
    N::Int

    grad::Vector{T}
    hess::Diagonal{T,Vector{T}}
    A::Matrix{T}
    b::Vector{T}
    function DoubleIntegrator(D::Int, N::Int;
            Qd = (@SVector ones(2D)),
            Rd = (@SVector fill(0.1, D)),
            Qfd = Qd*N*100,
            x0 = (@SVector zeros(2D)),
            xf = [(@SVector fill(1, D)); (@SVector zeros(D))],
            gravity::Bool=false,
            qinds = Vector{Int}[],
            s = ones(length(qinds))
        )
        n,m = 2D,D
        NN = 3D*N
        T = promote_type(eltype(Qd), eltype(Rd), eltype(x0), eltype(xf))
        inds = LinearIndices(zeros(n+m, N))
        xinds = SVector{n}.([z[1:n] for z in eachcol(inds)])
        uinds = SVector{m}.([z[1+n:end] for z in eachcol(inds)])

        # Cones
        p = length(qinds)
        ps = length.(qinds)
        @assert length(s) == p
        if p > 0
            pi = ps[1]
            @assert ps == fill(pi, p)  # ensure they're all the same size of cone
            qinds = SVector{pi}.(qinds)
        else
            pi = 0
            qinds = SVector{0,Int}[]
        end
        
        # Pre-allocation
        grad = zeros(NN)
        hess = Diagonal(zeros(NN))

        # Dynamics
        A,b = DI_dynamics(x0, N, gravity)

        new{n,m,pi,T}(Diagonal(Qd), Diagonal(Rd), Diagonal(Qfd), x0, xf, gravity, 
            xinds, uinds, qinds, s, N, grad, hess, A,b
        )
    end
end
num_vars(prob::DoubleIntegrator) = length(prob.grad)
num_cons(prob::DoubleIntegrator) = length(prob.b)
num_cones(prob::DoubleIntegrator) = length(prob.qinds) 


function cost(prob::DoubleIntegrator, x)
    J = zero(eltype(x))
    xf = prob.xf
    R = prob.R
    for k = 1:prob.N
        Q = k == N ? prob.Qf : prob.Q
        xk,uk = x[prob.xinds[k]], x[prob.uinds[k]]
        J += 0.5*(xk-xf)'Q*(xk-xf) + 0.5*uk'R*uk
    end
    return J
end

function grad_obj!(prob::DoubleIntegrator, grad, x)
    xf = prob.xf
    R = prob.R
    for k = 1:prob.N
        Q = k == N ? prob.Qf : prob.Q
        xk,uk = x[prob.xinds[k]], x[prob.uinds[k]]
        grad[prob.xinds[k]] = Q*(xk-xf)
        grad[prob.uinds[k]] = R*uk
    end
    return grad
end
grad_obj(prob::DoubleIntegrator, x) = grad_obj!(prob, prob.grad, x)

function grad_hess!(prob::DoubleIntegrator, H, x)
    R = prob.R
    for k = 1:prob.N
        Q = k == N ? prob.Qf : prob.Q
        ix = CartesianIndex.(prob.xinds[k])
        iu = CartesianIndex.(prob.uinds[k])
        H[ix] = Q.diag
        H[iu] = R.diag
    end
    return H
end
hess_obj(prob::DoubleIntegrator, x) = hess_obj!(prob, prob.hess, x)

con_eq(prob::DoubleIntegrator, x) = prob.A*x + prob.b
jac_eq(prob::DoubleIntegrator, x) = prob.A

con_soc(prob::DoubleIntegrator, x) = [push(x[qi],si) for (qi,si) in zip(prob.qinds,prob.s)]

function DoubleIntegratorFuns(D,N; kwargs...)
    prob = DoubleIntegrator(D,N; kwargs...)

    """
    Objective
    """
    n,m = 2D,D
    Q,R,Qf = prob.Q, prob.R, prob.Qf 
    x0,xf = prob.x0, prob.xf
    gravity = prob.gravity
    g = Float64[]
    h = Float64[]
    c = 0.0
    for k = 1:N-1
        append!(g, -Q*xf)
        append!(g, zeros(m))
        append!(h, Q.diag)
        append!(h, R.diag)
        c += 0.5*xf'Q*xf
    end
    append!(g, -Qf*xf)
    append!(g, zeros(m))
    append!(h, Qf.diag)
    append!(h, R.diag)
    c += 0.5*xf'Qf*xf
    H = Diagonal(h)

    """
    Dynamics
    """
    A,b = DI_dynamics(x0, N, gravity)

    di_obj(x) = 0.5*x'H*x + g'x + c
    di_dyn(x) = A*x + b
    return di_obj, di_dyn
end

function DI_dynamics(x0, N, gravity::Bool=false)
    n = length(x0)
    m = n ÷ 2
    D = m
    NN = N*(n+m)

    #  Continuous dynamics
    Ac = zeros(n,n)
    Bc = zeros(n,m)
    for i = 1:D
        Ac[i,D+i] = 1
        Bc[D+i,i] = 1
    end

    # Euler integration
    dt = 0.1
    Ad = I + Ac*dt
    Bd = Bc*dt

    iA = 1:n
    jA = iA
    jB = n .+ (1:m)
    jA2 = jA .+ (n+m)
    
    A = zeros(N*n, NN)
    b = zeros(N*n)
    A[iA,jA] .= I(n)
    b[iA] .= -x0
    iA = iA .+ n
    for k = 1:N-1
        A[iA,jA] .= Ad
        A[iA,jB] .= Bd
        A[iA,jA2] .= -I(n)
        if gravity
            b[iA[D]] = -9.81
        end

        # Advance inds
        iA = iA .+ n
        jA = jA .+ (n+m)
        jB = jB .+ (n+m)
        jA2 = jA2 .+ (n+m)
    end
    return A,b
end

function RocketLanding(N)
    n,m = 6,3
    Qd = SA_F64[1,1,1,10,10,10]
    Rd = @SVector fill(0.1, m) 
    Qfd = Qd*N*100
    x0 = SA_F64[1,1,10,0,0,-1]
    xf = @SVector zeros(n)
    DoubleIntegrator(3,N; gravity=true,
        Qd=Qd, Rd=Rd, Qfd=Qfd, x0=x0, xf=xf,
    )
end

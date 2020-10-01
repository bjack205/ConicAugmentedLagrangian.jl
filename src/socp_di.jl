using StaticArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using Random
using SolverLogging

abstract type Cone end
struct Equality <: Cone end
struct Inequality <: Cone end
struct SecondOrder <: Cone end
struct SecondOrder2 <: Cone end
struct NormCone <: Cone end

function len2inds(lens)
    a = insert!(cumsum(lens),1,0)
    return [a[i]+1:a[i+1] for i = 1:length(lens)] 
end

function soc_projection(x, dual=false)
    s = x[end]
    v = x[1:end-1]
    a = norm(v)
    d = dual ? -1 : 1
    if a <= -s*d        # below the cone
        return zero(x) 
    elseif a <= s*d     # in the cone
        return x
    elseif a >= abs(s)  # outside the cone
        return 0.5 * (1 + s/a) * push!(v, a)
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
end

function soc_penalty(c, λ)
    inactive = norm(λ, Inf) < 1e-9
    in_cone = in_soc(c)
    if in_cone && !inactive
        return c  # this seems odd to me. Should check if this actually helps
    else
        c_proj = soc_projection(c) 
        return c - c_proj
    end
end

function in_soc(c)
    s = x[end]
    v = x[1:end-1]
    return norm(v) <= s
end

##
function build_objective(n,m,N)
    Q = Diagonal(@SVector ones(n))
    R = Diagonal(@SVector fill(0.1, m))
    Qf = Diagonal(@SVector ones(n))*N*100
    xf = [(@SVector fill(1, n ÷ 2)); (@SVector zeros(n ÷ 2))]
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
    return H,g,c
end

function build_dynamics(D,N) 
    n,m = 2D,D
    NN = N*(n+m)
    x0 = zeros(n)

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
        iA = iA .+ n
        jA = jA .+ (n+m)
        jB = jB .+ (n+m)
        jA2 = jA2 .+ (n+m)
    end
    return A,b
end

function build_funcs(N,D=1)
    n,m = 2D, D

    H,g,cost_const = build_objective(n,m,N)
    A,b = build_dynamics(D,N)

    f(x) = 0.5*x'H*x + g'x + cost_const 
    c1(x) = A*x + b

    ix,iu = 1:n, n .+ (1:m)

    u_bnd = 50.0
    
    # control bounds
    function c2(Z)
        Z_ = reshape(Z,n+m,N)
        U = view(Z_,iu,:)
        Uvec = reshape(U,N*m)

        cval = zeros(eltype(Z), 2*m*N)
        cval[1:N*m] .= Uvec .- u_bnd
        cval[1+N*m:end] .= -u_bnd .- Uvec
        return cval
    end

    # norm-squared
    function c3(Z)
        Z_ = reshape(Z,n+m,N)
        U = view(Z_,iu,:)

        U2 = U .* U
        cval = vec(sum(U2, dims=1))
        cval .-= u_bnd*u_bnd
        return cval
    end

    # second-order cone
    function c4(Z)
        Z_ = reshape(Z,n+m,N)
        U = Z_[iu,:] 
        C = [U; fill(u_bnd, 1, N)]
        return vec(C)
    end

    function c5(Z)
        Z_ = reshape(Z,n+m,N)
        U = Z_[iu,:] 
        C = zeros(eltype(Z), m+1,N)
        for k = 1:N
            u = append!(U[:,k], u_bnd)
            C[:,k] = u - soc_projection(u)
        end
        vec(C)
    end

    penalty(::Equality, cval, λ) = cval
    function penalty(::Inequality, cval, λ)
        a = @. (cval >= 0) | (λ < 0) 
        return a .* cval
    end
    penalty(::SecondOrder, cval, λ) = soc_penalty(cval, λ)
    penalty(::SecondOrder2, cval, λ) = cval

    projection(::Equality, cval, ::Val{false}) = zero(cval)
    projection(::Equality, cval, ::Val{true}) = cval 
    projection(::Inequality, cval, ::Val{false}) = min.(0, cval)
    projection(::Inequality, cval, ::Val{true}) = min.(0, cval)
    function projection(::Union{SecondOrder,SecondOrder2}, cval, ::Val{dual}) where dual 
        @assert length(cval) % (m+1) == 0
        cmat = reshape(cval,m+1,:)
        proj = copy(cmat)
        for i = 1:size(cmat,2)
            u = view(cmat, :, i)
            proj[:,i] = soc_projection(u, false)
        end
        return vec(proj)
    end

    # Functions that act on all the constraints
    cons = (c1,c2,c3,c4,c5)
    conlen = [N*n, 2*N*m, N, N*(m+1), N*(m+1)]
    contype = (Equality(), Inequality(), Inequality(), SecondOrder(), SecondOrder2())

    coninds = [1,4]
    conpart = len2inds(conlen[coninds]) 
    function c(x)
        vcat([cons[i](x) for i in coninds]...)
    end

    function penalty(x,λ)
        vcat([
            penalty(contype[j], cons[j](x), view(λ, conpart[i])) 
            for (i,j) in enumerate(coninds)
        ]...)
    end

    function projection(cval, dual=Val(false))
        vcat([
            projection(contype[j], view(cval, conpart[i]), dual)
            for (i,j) in enumerate(coninds)
        ]...)
    end

    # Augmented Lagrangian 
    function L(x, λ, μ)
        p = penalty(x, λ)
        f(x) - λ'c(x) + 0.5*μ*p'p
    end

    function dual_update(λ, x, μ)
        cval = c(X)
        λbar = λ - μ .* cval
        return projection(λbar, Val(false))
    end

    return f, L, c, penalty, projection, dual_update
end
N = 11
D = 2
f, L, c, penalty, projection, dual_update = build_funcs(N,D)

##
NN = N*3D
x = rand(NN)
P = length(c(x))
λ = rand(P)
μ = rand()

f(x)
c(x)
penalty(x,λ)
L(x,λ,μ)
c(x) - projection(c(x))
dual_update(λ, c(x), μ)

L_(x) = L(x,λ,μ)
g = ForwardDiff.gradient(L_, x)
H = ForwardDiff.hessian(L_, x)
dx = -H\g
xbar = x + dx
norm(g,1)
norm(ForwardDiff.gradient(L_, xbar),1)
x .= xbar
λ = dual_update(λ, c(x), μ)
norm(c(x) - projection(c(x)),1)
μ *= 10

##
function newton_solver_AD(f,x0; 
        newton_iters=10, ls_iters=10, verbose=false, ϵ=1e-6
    )
    x = copy(x0)
    xbar = copy(x0)
    ρ = 1e-6
    ngrad = Inf
    α = 1.0
    for i = 1:newton_iters
        # Solve for the step direction
        g = ForwardDiff.gradient(f, x)
        H = ForwardDiff.hessian(f, x)
        Hinv = factorize(H + ρ*I)
        dx = -(Hinv\g)

        # Iterative Refinement
        # r = H*dx + g
        # nr = norm(r,1)
        # cr = Inf
        # m = 1
        # while nr > 1e-12 && m < 5 && cr > 1.2
        #     dx += -(Hinv\r)
        #     r = H*dx + g
        #     nr_ = norm(r,1)
        #     cr = log(nr_) / log(nr)  # convergence rate
        #     nr = nr_
        #     m += 1
        # end

        # Line Search
        J0 = f(x)
        ngrad0 = norm(g,1)
        α = 1.0
        for j = 1:ls_iters
            xbar .= x + α*dx 
            J = f(xbar)
            grad = ForwardDiff.gradient(f, xbar)
            ngrad = norm(grad,1)

            # Wolfe conditions
            if (J <= J0 + 1e-4*α*dx'g) && (dx'grad >= 0.9*dx'g)
                break                
            else
                α /= 2
            end
            if j == ls_iters
                ρ *= 10
                @warn "Max Line Search Iterations"
            else
                if ρ > 1e-6
                    ρ /= 10
                end
            end
        end
        
        # Accept the step
        x .= xbar

        if verbose
            println("  Iteration $i")
            println("    cost: ", f(x))
            println("    grad: ", ngrad)
            println("       α: ", α)
        end

        # Convergence check
        if ngrad < ϵ
            break
        end
    end
    return x
end

function augmented_lagrangian(x0, y0, L, c, projection, dual_update;
        μ = 1.0,
        ϕ = 10.0,
        al_iters=10,
        kwargs...
    )
    n = length(x0)
    m = length(y0)

    x = copy(x0)  # primals
    y = copy(y0)  # duals
    for j = 1:al_iters
        _L(x) = L(x, y, μ)
        x = newton_solver_AD(_L, x; kwargs...)

        # Update constraints
        cval = c(x)
        viol = cval - projection(cval) 
        feas = norm(viol, Inf)

        # Dual update
        y = dual_update(y, x, μ)

        # Penalty Update
        μ *= ϕ

        println("Outer Loop Update $j: feas = ", feas)
        if feas < 1e-6
            break
        end
    end
    return x, y
end

##
f, L, c, penalty, projection, dual_update = build_funcs(N,D)
NN = 3D*N
x = rand(NN)
x[2D+1:3D] .= 100
λ = rand(length(c(x)))*0
zsol, ysol = augmented_lagrangian(x, λ, L, c, projection, dual_update, 
    verbose=true, al_iters=6, newton_iters=20)

c(zsol)
projection(c(zsol))
norm(c(zsol) - projection(c(zsol)),Inf)
x = reshape(zsol,:,N)[1:2D,:]
u = reshape(zsol,:,N)[2D+1:end,:]
[norm(u) for u in eachcol(u)]

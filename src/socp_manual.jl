using StaticArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using Random

const n = 2
const m = 1
const N = 11
const CONS = :soc
@enum CONTYPE EQ INEQ SOC


function get_obj()
    Q = Diagonal(@SVector ones(n))
    R = Diagonal(@SVector fill(0.1, m))
    Qf = Diagonal(@SVector ones(n))*N*100
    xf = [(@SVector fill(10, n ÷ 2)); (@SVector zeros(n ÷ 2))]
    return Q,R,Qf,xf
end

function f(Z)
    Q,R,Qf,xf = get_obj()
    J = zero(eltype(Z))
    for k = 1:N-1
        x,u = state(Z,k), control(Z,k)
        Jk = 0.5*(x-xf)'Q*(x-xf) + 0.5*u'R*u
        J += Jk
    end
    x,u = state(Z,N), control(Z,N)
    JN = 0.5*(x-xf)'Qf*(x-xf) + 0.5*u'R*u
    J += JN
    return J
end

state(Z,i) = Z[SVector{n}(1:n) .+ (n+m)*(i-1)]
control(Z,i) = Z[SVector{m}(1:m) .+ n .+ (n+m)*(i-1)]
states(Z) = [state(Z, k) for k = 1:N]
controls(Z) = [control(Z, k) for k = 1:N]

function grad_f(Z)
    ForwardDiff.gradient(f, Z)
end

function hess_f(Z)
    Q,R,Qf = get_obj()
    d = [Q.diag; R.diag]
    H = d 
    for i = 2:N-1
        H = [H; d]
    end
    df = [Qf.diag; R.diag]
    H = [H; df]
    return Diagonal(H)
end

function c!(c_val, Z, c_type=fill(EQ, length(c_val)))
    x0 = @SVector zeros(n)
    Ac = SA[0 1; 0 0]
    Bc = SMatrix{2,1}(0,1) 
    dt = 0.1

    # Euler Integration
    A = I + Ac*dt
    B = Bc*dt

    # Dynamics constraints
    Pdyn = N*n
    inds = 1:n
    c_val[inds] .= state(Z,1) - x0
    inds = inds .+ n
    for k = 1:N-1
        x,u = state(Z,k), control(Z,k)
        x_next = state(Z,k+1)
        c_val[inds] = A*x + B*u - x_next 
        inds = inds .+ n
    end
    c_type[1:Pdyn] .= EQ 

    # Control norm constraints
    #   ||u|| < val
    val = 50.0 
    inds = Pdyn .+ (1:m+1)
    for k = 1:N-1
        u = control(Z,k)
        if CONS == :norm2
            c_val[Pdyn + k] = u'u - val*val
            # c_val[Pdyn + k] = u[1] - val
            c_type[Pdyn + k] = INEQ  # inequality constraint
        end
        if CONS == :soc
            c_val[inds] = push(u, val)
            c_type[inds] .= SOC
            inds = inds .+ (m+1)
        end
    end
    return nothing
end

function c(Z)
    c_val = zeros(eltype(Z), num_duals())
    c!(c_val, Z)
    return c_val
end

function projection!(c_proj, c_val)
    NN = num_primals()
    Pdyn = N*n 
    c_proj[1:Pdyn] .= 0  # dynamics constraints
    if CONS == :norm2 
        for i = Pdyn + 1:length(c_val)
            c_proj[i] = min(0, c_val[i])
        end
    elseif CONS == :soc
        inds = Pdyn .+ (1:m+1)
        for k = 1:N-1
            c_proj[inds] .= soc_projection(c_val[inds])
            inds = inds .+ (1+m)
        end
    end
end

function jac_c!(jac_c, c, Z)
    ForwardDiff.jacobian!(jac_c, c!, c, Z)
end

function ∇c(Z)
    NN = num_primals() 
    P = num_duals()
    jac = zeros(P,NN)
    c_val = zeros(P)
    jac_c!(jac, c_val, Z)
    return jac
end


num_primals() = N*(n+m)
function num_duals()
    if CONS == :norm2
        con = N-1
    elseif CONS == :soc
        con = (N-1)*(m+1)
    else
        con = 0
    end
    return N*n + con 
end
primals(V) = view(V, 1:num_primals())
duals(V) = view(V, num_primals()+1:length(V))

## Augmented Lagrangian
function LA(V, μ=1)
    Z, λ = primals(V), duals(V)
    c_val = c(Z)
    pen = penalty(c_val, λ)
    f(Z) + λ'c_val + 0.5*μ * pen'pen
end

function penalty(c, λ)
    pen = copy(c)
    Pdyn = N*n

    # Dynamics constraints
    pen[1:Pdyn] .= c[1:Pdyn]

    # Control constraints
    if CONS == :norm2
        a = @. (c >= 0) | (λ > 0)
        inds = (Pdyn + 1:length(c))
        pen[inds] .= a[inds] .* c[inds]
    elseif CONS == :soc
        inds = Pdyn .+ (1:m+1)
        for k = 1:N-1
            ci = view(c,inds)
            λi = view(λ,inds)
            inactive = norm(λi,Inf) < 1e-8
            if in_soc(ci) && !inactive
                pen[inds] .= ci
            else
                pen[inds] .= ci .- soc_projection(ci)
            end
            inds = inds .+ (m+1)
        end
    end
    return pen
end

function ∇penalty(c, λ)
    p(c) = penalty(c, λ)
    ForwardDiff.jacobian(p, c)
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

function in_soc(x)
    s = x[end]
    v = x[1:end-1]
    a = norm(v)
    return a <= s
end

function grad_LA(V, μ)
    Z, λ = primals(V), duals(V)
    c_val = c(Z)
    jac = ∇c(Z)
    jac_p = ∇penalty(c_val, λ) * jac  # jacobian of combined penalty term
    grad_f(Z) + jac'λ + μ * jac_p'c_val
end

function hess_LA(V, μ)
    Z, λ = primals(V), duals(V)
    c_val = c(Z)
    jac = ∇c(Z)
    jac_p = ∇penalty(c_val, λ) * jac  # jacobian of combined penalty term
    hess_f(Z) + μ * jac_p'jac_p
end

function dual_update(λ, μ, c_val)
    Pdyn = N*n 
    λ .+= μ * c_val

    # Inequality constraints
    if CONS == :norm2
        λcon = view(λ, Pdyn + 1 : num_duals())
        λcon .= max.(0, λcon)
    elseif CONS == :soc
        inds = (1:m+1) .+ Pdyn
        for i = 1:N-1
            λcon = view(λ, inds)
            λcon .= soc_projection(λcon, true)
            inds = inds .+ (m+1)
        end
    end
    return λ
end

## Solver
function newton_solver(x0,y0; newton_iters=10)
    n,m = length(x0), length(y0)
    z = [x0;y0]
    x = primals(z)  # creates a view
    y = duals(z)    # creates a view
    zbar = copy(z)
    xbar = primals(zbar)
    ybar = duals(zbar)
    ρ = 1e-6        # regularization
    ngrad_bar = Inf

    c_vals = zero(y0)
    α = 1.0
    for i = 1:newton_iters
        grad = grad_LA(z,μ)
        hess = hess_LA(z,μ)
        dx = -(hess + I*ρ)\grad
        ngrad = norm(grad,1)
        
        # line search
        α = 1.0
        for j = 1:25
            zbar .= z
            xbar .+= α * dx
            ngrad_bar = norm(grad_LA(zbar, μ),1)
            if ngrad_bar <= (1 - α*0.1)*ngrad
                ρ /= 10
                break
            else
                α *= 0.5
            end
            if j == 25
                @warn "max linesearch iters"
                ρ *= 10
            end
        end
        x .= xbar
        println("  Iteration $i")
        println("    cost: ", f(x))
        println("    grad: ", ngrad_bar)
        println("       α: ", α)

        if ngrad_bar < 1e-6
            break
        end
    end
    return x
end

function augmented_lagrangian(x0, y0;
        al_iters=10,
        kwargs...
    )
    μ = 1.0
    ϕ = 10.0
    x = copy(x0)  # primals
    y = copy(y0)  # duals
    cval = zero(y0)
    cproj = zero(y0)  # projected constraint values
    viol = zero(y0)   # constraint violations
    for j = 1:al_iters
        x = newton_solver(x, y; kwargs...)

        # Update constraints
        c!(cval, x)
        projection!(cproj, cval)
        viol .= cval .- cproj
        feas = norm(viol, 1)

        # Dual update
        y = dual_update(λ, μ, cval)

        # Penalty Update
        μ *= ϕ

        println("Outer Loop Update $j: feas = ", feas)
        if feas < 1e-6
            break
        end
    end
    return x, y
end

## Initialization
Random.seed!(1)
NN = N*(n+m)
P = num_duals() 

Z = rand(NN)*0
λ = rand(P)*0
c_val = zeros(P)
jac_c = zeros(P,NN)

##
Zans, λans = augmented_lagrangian(Z, λ, al_iters=6, newton_iters=20)
c!(c_val, Zans)
c_val[N*n+1:end]
U = controls(Zans)
norm.(U)
c_proj = zero(c_val)
projection!(c_proj, c_val)
viol = c_val - c_proj
states(Zans)[end]

##
f(Z)
grad_f(Z)
hess_f(Z) ≈ ForwardDiff.hessian(f,Z)

c!(c_val, Z)
jac_c!(jac_c, c_val, Z)

# Augmented Lagrangian
Z = rand(NN)
λ = zeros(P)
V = [Z; λ]
primals(V) == Z
duals(V) == λ
jac_c!(jac_c, c_val, Z)

μ = rand()
λ *= 0
pen = penalty(c_val, λ)
∇pen = ∇penalty(c_val, λ) * jac_c
c_proj = zero(c_val)
projection!(c_proj, c_val)
c_viol = c_val - c_proj
penalty(c_val, λ) ≈ c_viol

LA(V, μ) ≈ f(Z) + λ'c_val + 0.5 * μ * c_viol'c_viol
grad_LA(V, μ) ≈ ForwardDiff.gradient(x->LA(x,μ), V)[1:NN]
hess_LA(V, μ) ≈ hess_f(Z) + μ * ∇pen'∇pen 

# Test newton
V = rand(NN+P)
Z = primals(V)
λ = duals(V)

Vbar = copy(V)
Zbar = primals(Vbar)

g = grad_LA(V, μ)
H = hess_LA(V, μ)
dZ = -H\g

Zbar .= Z + dZ
norm(g,1)
norm(grad_LA(Vbar, μ),1)
V .= Vbar

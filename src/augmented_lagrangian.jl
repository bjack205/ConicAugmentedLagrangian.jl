include("problem.jl")
include("newton_solver.jl")

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

function in_soc(x)
    v = x[1:end-1]
    s = x[end]
    a = norm(v)
    return a <= s
end

function jac_soc(x)
    n = length(x)
    J = zeros(eltype(x), n,n)
    jac_soc!(J, x)
    return J
end

function jac_soc!(J, x)
    n = length(x)
    s = x[end]
    v = x[1:end-1] 
    a = norm(v)
    if a <= -s                               # below cone
        J .*= 0
    elseif a <= s                            # in cone
        J .*= 0
        for i = 1:n
            J[i,i] = 1.0
        end
    elseif a >= abs(s)                       # outside cone
        # scalar
        b = 0.5 * (1 + s/a)   
        dbdv = -0.5*s/a^3 * v
        dbds = 0.5 / a

        # dvdv = dbdv * v' + b * oneunit(SMatrix{n-1,n-1,T})
        for i = 1:n-1, j = 1:n-1
            J[i,j] = dbdv[i] * v[j]
            if i == j
                J[i,j] += b
            end
        end

        # dvds
        J[1:n-1,n] .= dbds * v

        # ds
        dsdv = dbdv * a + b * v / a 
        dsds = dbds * a
        # ds = push(dsdv, dsds)
        ds = [dsdv; dsds]
        J[n,:] .= ds
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
    return J
end

function hess_soc(x,b)
    n = length(x)
    s = x[end]
    v = x[1:end-1] 
    bs = b[end]
    bv = b[1:end-1] 
    a = nv = norm(v)

    if a <= -s
        return zeros(n,n)
    elseif a <= s
        return zeros(n,n)
    elseif a > abs(s)
        dvdv = -s/norm(v)^2/norm(v)*(I - (v*v')/(v'v))*bv*v' + 
            s/norm(v)*((v*(v'bv))/(v'v)^2 * 2v' - (I*(v'bv) + v*bv')/(v'v)) + 
            bs/norm(v)*(I - (v*v')/(v'v))
        dvds = 1/norm(v)*(I - (v*v')/(v'v))*bv;
        dsdv = bv'/norm(v) - v'bv/norm(v)^3*v'
        dsds = 0
        return 0.5*[dvdv dvds; dsdv dsds]
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
end

function dual_update_eq(y, ceq, μ)
    ybar = y + μ * ceq 
    return ybar
end

function dual_update_soc(z, qx, μ)
    # zbar = [zi - μ * q for (zi,q) in zip(z,qx)]
    p = length(z)
    zbar = [z[i] - μ * qx[i] for i = 1:p]
    # zbar = [Πsoc(z[i] - μ * qx[i]) for i = 1:p]
    return zbar
    # return Πsoc(zbar)
end

function augmented_lagrangian_AD(x0, f, h, q=x->Vector{Float64}[]; kwargs...)
    prob = ADProblem(length(x0), f, h, q)
    augmented_lagrangian(prob, x0; kwargs...)
end
function augmented_lagrangian(prob::ProblemDef, x0;
        y0 = zeros(num_cons(prob)),
        z0 = zero.(con_soc(prob, x0)),
        μ = 1.0,
        ϕ = 10.0,
        al_iters=10,
        ϵ_feas=1e-6,
        kwargs...
    )
    set_logger()
    ldata = global_logger().leveldata[OuterLoop]
    if :verbose ∈ keys(kwargs)
        ldata.freq = 1
    else
        ldata.freq = 10
    end
    SolverLogging.clear_cache!(ldata)

    n = length(x0)
    m = length(y0)
    p = length(z0)  # number of cones

    x = copy(x0)  # primals
    y = copy(y0)  # duals on equality constraint h(x)
    z = copy(z0)  # duals on soc constraint q(x) (vector of vectors)

    iters = 0
    for j = 1:al_iters
        function L(x)
            ceq = con_eq(prob,x)
            L0 = cost(prob, x) + y'ceq + 0.5*μ*ceq'ceq
            if p > 0
                # csoc = [Πsoc(zi - μ*q) for (zi,q) in zip(z, q(x))]
                cones = con_soc(prob, x)
                csoc = [Πsoc(z[i] - μ*cones[i]) for i = 1:p]
                pen = [csoc[i]'csoc[i] - z[i]'z[i] for i = 1:p]
                L0 += 1 / (2μ) * sum(pen)
            end
            return L0
        end
        x, stats = newton_solver_AD(L, x; kwargs...)
        iters += stats.iters

        # Update constraints
        ceq = con_eq(prob, x)
        feas = norm(ceq, Inf)

        # Dual update
        y = dual_update_eq(y, ceq, μ)
        if p > 0
            qx = con_soc(prob, x)
            viol = [norm(qx[i] - Πsoc(qx[i]),Inf) for i = 1:p]
            feas_soc = maximum(viol) 
            feas = max(feas, feas_soc) 
            z = dual_update_soc(z, qx, μ)
            @logmsg OuterLoop :feas_soc value=feas_soc
        end

        # Penalty Update
        μ *= ϕ

        @logmsg OuterLoop :iter value=j
        @logmsg OuterLoop :cost value=cost(prob,x)
        @logmsg OuterLoop :cost_AL value=L(x)
        @logmsg OuterLoop :feas value=feas
        @logmsg OuterLoop :μ value=μ
        @logmsg OuterLoop :iter_total value=iters
        print_level(OuterLoop, global_logger())
        if feas < ϵ_feas 
            break
        end
    end
    return x, y, z
end
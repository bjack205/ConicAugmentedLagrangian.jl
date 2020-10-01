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

function augmented_lagrangian_AD(x0, f, h, q=x->Vector{Float64}[];
        y0 = zero(h(x0)),
        z0 = zero.(q(x0)),
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
            ceq = h(x)
            L0 = f(x) + y'ceq + 0.5*μ*ceq'ceq
            if p > 0
                # csoc = [Πsoc(zi - μ*q) for (zi,q) in zip(z, q(x))]
                cones = q(x)
                csoc = [Πsoc(z[i] - μ*cones[i]) for i = 1:p]
                pen = [csoc[i]'csoc[i] - z[i]'z[i] for i = 1:p]
                L0 += 1 / (2μ) * sum(pen)
            end
            return L0
        end
        x, stats = newton_solver_AD(L, x; kwargs...)
        iters += stats.iters

        # Update constraints
        ceq = h(x)
        feas = norm(ceq, Inf)

        # Dual update
        y = dual_update_eq(y, ceq, μ)
        if p > 0
            qx = q(x)
            viol = [norm(qx[i] - Πsoc(qx[i]),Inf) for i = 1:p]
            feas_soc = maximum(viol) 
            feas = max(feas, feas_soc) 
            z = dual_update_soc(z, qx, μ)
            @logmsg OuterLoop :feas_soc value=feas_soc
        end

        # Penalty Update
        μ *= ϕ

        @logmsg OuterLoop :iter value=j
        @logmsg OuterLoop :cost value=f(x)
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
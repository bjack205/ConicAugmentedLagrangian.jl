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

function dual_update_eq(y, ceq, μ)
    ybar = y + μ * ceq 
    return ybar
end

function dual_update_soc(z, qx, μ)
    zbar = z - μ * qx
    return zbar
    # return Πsoc(zbar)
end

function augmented_lagrangian_AD(x0, f, h, q=x->Float64[];
        y0 = zero(h(x0)),
        z0 = zero(q(x0)),
        μ = 1.0,
        ϕ = 10.0,
        al_iters=10,
        kwargs...
    )
    set_logger()
    SolverLogging.clear_cache!(global_logger().leveldata[OuterLoop])

    n = length(x0)
    m = length(y0)
    p = length(z0)

    x = copy(x0)  # primals
    y = copy(y0)  # duals on equality constraint h(x)
    z = copy(z0)  # duals on soc constraint q(x)

    iters = 0
    for j = 1:al_iters
        function L(x)
            ceq = h(x)
            L0 = f(x) + y'ceq + 0.5*μ*ceq'ceq
            if p > 0
                csoc = Πsoc(z - μ*q(x))
                L0 += 1 / (2μ) * (csoc'csoc - z'z)
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
            viol = qx - Πsoc(qx)
            feas_soc = norm(viol, Inf)
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
        if feas < 1e-6
            break
        end
    end
    return x, y
end
function dual_update_eq(y, ceq, μ)
    ybar = y + μ * ceq 
    return ybar
end
function augmented_lagrangian_AD(x0, y0, f, h;
        μ = 1.0,
        ϕ = 10.0,
        al_iters=10,
        kwargs...
    )
    set_logger()
    SolverLogging.clear_cache!(global_logger().leveldata[OuterLoop])

    n = length(x0)
    m = length(y0)

    x = copy(x0)  # primals
    y = copy(y0)  # duals on equality constraint h(x)

    iters = 0
    for j = 1:al_iters
        L(x) = f(x) + y'h(x) + 0.5*μ*h(x)'h(x)
        x, stats = newton_solver_AD(L, x; kwargs...)
        iters += stats.iters

        # Update constraints
        ceq = h(x)
        feas = norm(ceq, Inf)

        # Dual update
        y = dual_update_eq(y, ceq, μ)

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
include("problem.jl")
include("newton_solver.jl")
include("cones.jl")

function dual_update_eq(y, ceq, μ)
    ybar = y + μ * ceq 
    return ybar
end

function dual_update_soc(z, qx, μ)
    # zbar = [zi - μ * q for (zi,q) in zip(z,qx)]
    p = length(z)
    # zbar = [z[i] - μ * qx[i] for i = 1:p]
    zbar = [Πsoc(z[i] - μ * qx[i]) for i = 1:p]
    return zbar
    # return Πsoc(zbar)
end

function augmented_lagrangian_AD(x0, f, h, q=x->Vector{Float64}[], qinds=Vector{Int}[]; kwargs...)
    prob = ADProblem(length(x0), f, h, q, qinds)
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
    if :verbose ∈ keys(kwargs) && kwargs[:verbose] == true
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
    z = Vector.(z)


    iters = 0
    conetypes = get_cones(prob)
    for j = 1:al_iters
        function L(x)
            ceq = con_eq(prob,x)
            L0 = obj(prob, x) + y'ceq + 0.5*μ*ceq'ceq
            if p > 0
                # csoc = [Πsoc(zi - μ*q) for (zi,q) in zip(z, q(x))]
                cones = con_soc(prob, x)
                csoc = [projection(conetypes[i], z[i] - μ*cones[i]) for i = 1:p]
                pen = [csoc[i]'csoc[i] - z[i]'z[i] for i = 1:p]
                L0 += 1 / (2μ) * sum(pen)
            end
            return L0
        end
        alprob = ALProblem(prob, y, Vector.(z), μ)
        x, stats = newton_solver(alprob, x; kwargs...)
        # x, stats = newton_solver_AD(L, x; kwargs...)
        iters += stats.iters

        # Update constraints
        ceq = con_eq(prob, x)
        feas = norm(ceq, Inf)

        # Dual update
        y = dual_update_eq(y, ceq, μ)
        if p > 0
            qx = con_soc(prob, x)
            viol = [norm(qx[i] - projection(conetypes[i], qx[i]),Inf) for i = 1:p]
            feas_soc = maximum(viol) 
            feas = max(feas, feas_soc) 
            z = dual_update_soc(z, qx, μ)
            @logmsg OuterLoop :feas_soc value=feas_soc
        end

        # Penalty Update
        μ *= ϕ

        @logmsg OuterLoop :iter value=j
        @logmsg OuterLoop :cost value=obj(prob,x)
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
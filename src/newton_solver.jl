using LinearAlgebra
using ForwardDiff
using SolverLogging
using Logging

function set_logger()
    if !(global_logger() isa SolverLogger)
        logger = SolverLogger(InnerLoop)
        add_level!(logger, InnerLoop, print_color=:green, indent=4)
        add_level!(logger, OuterLoop, print_color=:blue, indent=0)
        global_logger(logger)
    end
end

struct iLQRStats{T}
    J0::T        # initial cost
    J::T         # final cost
    grad::T      # terminal gradient
    iters::Int   # iterations
end

function newton_solver_AD(f,x0; kwargs...)
    h(x) = zeros(0)
    prob = ADProblem(length(x0), f) 
    newton_solver(prob, x0; kwargs...)
end
function newton_solver(prob::ProblemDef, x0; 
        newton_iters=10, 
        ls_iters=10, 
        verbose=false, 
        ϵ=1e-6,
        ir_tol=1e-12,
        ir_iter=5,
        ir_rate_thresh=1.2,
        reg_init=1e-6
    )
    set_logger()
    SolverLogging.clear_cache!(global_logger().leveldata[InnerLoop])

    x = copy(x0)
    xbar = copy(x0)

    Jinit = cost(prob, x0)
    J = Inf

    ρ = reg_init 
    ngrad = Inf
    α = 1.0
    iters = 0
    for i = 1:newton_iters
        # Solve for the step direction
        g = grad_obj(prob, x)
        H = hess_obj(prob, x)
        Hinv = factorize(H + ρ*I)
        dx = -(Hinv\g)

        # Iterative Refinement
        r = H*dx + g
        nr = norm(r,1)
        cr = Inf
        m = 0
        while nr > ir_tol && m < ir_iter && cr > ir_rate_thresh 
            dx += -(Hinv\r)
            r = H*dx + g
            nr_ = norm(r,1)
            cr = log(nr_) / log(nr)  # convergence rate
            nr = nr_
            m += 1
        end

        # Line Search
        J0 = cost(prob, x)
        ngrad0 = norm(g,1)
        α = 1.0
        for j = 1:ls_iters
            xbar .= x + α*dx 
            J = cost(prob, xbar)
            grad = grad_obj(prob, xbar)
            ngrad = norm(grad,1)

            # Wolfe conditions
            if (J <= J0 + 1e-4*α*dx'g) && (dx'grad >= 0.9*dx'g)
                if ρ > 1e-6
                    @logmsg InnerLoop "Regularization Decreased"
                    ρ /= 10
                end
                break                
            else
                α /= 2
            end
            if j == ls_iters
                ρ *= 10
                # @warn "Max Line Search Iterations"
                @logmsg InnerLoop "Max Line Search Iterations"
            end
        end
        
        # Accept the step
        x .= xbar

        @logmsg InnerLoop :iter value=i 
        @logmsg InnerLoop :cost value=cost(prob, x)
        @logmsg InnerLoop :grad value=ngrad 
        @logmsg InnerLoop :α value=α
        @logmsg InnerLoop :ir_steps value=m
        @logmsg InnerLoop :ρ value=ρ
        if verbose
            print_level(InnerLoop, global_logger())            
        end

        # Convergence check
        if ngrad < ϵ
            @logmsg OuterLoop "Cost Criteria Met"
            iters = i
            break
        end
        if i == newton_iters
            iters = newton_iters
            @logmsg OuterLoop "Hit Max Newton Iterations"
        end
    end
    stats = iLQRStats(Jinit, J, ngrad, iters)
    return x, stats
end
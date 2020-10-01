using LinearAlgebra, ForwardDiff
# using Convex, SCS, ECOS

# """
#  min x'Px
#  st Ax = b
#     ||x|| <= t
# """

n = 66 
mc = 44 
m = mc
P = Diagonal(rand(n))
p = rand(n)
A = rand(mc,n)
b = rand(mc)
t = 0.3 

function get_cones(x)
    x_ = reshape(x,:,11)[2D+1:end,:]
    return [[x;t] for x in eachcol(x_)]
end


D = 2
N = 11
n = 3D*N
mc = 2D*N
m = mc
P,p,c_cost = build_objective(2D,D,N)
A,b = build_dynamics(D,N)
t = 100.0

function get_cones(x)
    x_ = reshape(x,:,N)[2D+1:end,:]
    return [[x;t] for x in eachcol(x_)]
end

# "Convex.jl"
# x = Variable(n)
# prob = minimize(quadform(x,P))
# prob.constraints += A*x == b
# prob.constraints += norm(x) <= t
# solve!(prob,SCS.Optimizer)

# @show prob.status
# @show x.value
# prob.optval

"Second-order cone projection"
function Π(v,s)
	if norm(v) <= -s
		# @warn "below cone"
		return zero(v), 0.0
	elseif norm(v) <= s
		# @warn "in cone"
		return v, s
	elseif norm(v) > abs(s)
		# @warn "outside cone"
		a = 0.5*(1.0 + s/norm(v))
		return a*v, a*norm(v)
	else
		@warn "soc projection error"
		return zero(v), 0.0
	end
end
function Π(x)
    v,s = Π(x[1:end-1], x[end])
    return push!(v,s)
end

"Augmented Lagrangian"
f(x) = x'*P*x + p'x + c_cost
c1(x) = A*x - b
function c2(x)
    cones = get_cones(x)
    conesp = Π.(cones)
    return vcat((conesp - cones)...)
end
# c(x) = [c1(x);c2(x)]
c(x) = c1(x)

L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function LA(x,λ,ρ)
    f(x)
end

function solve(x, λ=zero(c(x)))
	x = copy(x)

	ρ = 1.0

	k = 1
	while k < 10
        _L(z) = L(z, λ, ρ)
        x = newton_al(_L, x)
		λ[1:m] += λ[1:m] + ρ*c1(x)

		# tmp = λ[m .+ (1:n+1)] + ρ*([x;t])
		# λxp, λtp = Π(tmp[1:n],tmp[n+1])
        # λ[m .+ (1:n+1)] = [λxp;λtp]

        if norm(c(x)) < 1e-6
            println("AL Took $k steps")
            break
        end

		ρ *= 10.0

		k += 1
	end

	return x, λ, ρ
end
function newton_al(_L, x)
    i = 1
    while i < 10
        f = _L(x)
        ∇L = ForwardDiff.gradient(_L,x)
        if norm(∇L) < 1e-5
            println("  Newton took $i steps")
            break
        end
        ∇²L = ForwardDiff.hessian(_L,x)
        Δx = -(∇²L + 1.0e-5*I)\∇L

        α = 1.0

        j = 1
        while j < 10
            if (_L(x + α*Δx) <= f + 1.0e-4*α*Δx'*∇L) && (-Δx'*ForwardDiff.gradient(_L,x+α*Δx) <= -0.9*Δx'*∇L)
                break
            else
                α *= 0.5
                j += 1
            end
        end
        j == 10 && @warn "line search failed"

        x .+= α*Δx
        i += 1
    end
    return x
end

x0 = randn(n)
x_sol, λ_sol, ρ_sol = solve(x0)

@show x_sol
@assert norm(x_sol - x.value) < 1.0e-5

projection(cval) = zero(cval)  # for calculating feasibility
function dual_update(λ0,x,ρ)
    λ = copy(λ0)
    λ[1:m] = λ[1:m] + ρ*c1(x)
    return λ

    cones = get_cones(x)
    n_cones = length(cones)
    λcones = reshape(λ[m+1:end],:,n_cones)
    λcones = collect(eachcol(λcones))
    λup = λcones .+ ρ .* cones
    λproj = Π.(λup)
    λ[m+1:end] .= vcat(λproj...)
    # tmp = λ[m .+ (1:n+1)] + ρ*([x;t])
    # λxp, λtp = Π(tmp[1:n],tmp[n+1])
    # λ[m .+ (1:n+1)] = [λxp;λtp]
    return λ
end



x0 = randn(n)
zsol, ysol = augmented_lagrangian(x0, λ0, L, c, projection, dual_update, 
    verbose=true, al_iters=10, newton_iters=20)
ysol

x = reshape(zsol,:,N)[1:2D,:]
u = reshape(zsol,:,N)[1+2D:end,:]
norm.(eachcol(u))
t = 35
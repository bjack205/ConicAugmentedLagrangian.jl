using LinearAlgebra
using Random
using ForwardDiff
# size
n = 3

# Objective
p0 = [1., 1., 1.]
f(p) = 0.5*(p - p0)'*(p - p0)

# utils
function pack_z(p,λ,ρ)
	return [p; λ; ρ]
end
# utils
function unpack_z(z)
	return z[1:n], z[n+1:2n], z[end]
end

# Lagrangian
function L(p,λ,ρ)
	cone_pen = cone_penalty(p,λ)
	lag = f(p) + λ'*p + 0.5*ρ*cone_pen'*cone_pen
	return lag
end
# Lagrangian
L(z) = L(unpack_z(z)...)
# Residual, also gradient of the Lagrangian
function G_res(z)
	return ForwardDiff.gradient(L, z)[1:n]
end
# Jacobian of the residual, also Hessian of the Lagrangian
function H_res(z)
	return ForwardDiff.hessian(L, z)[1:n,1:n]
end
# Dual ascent and projection of the duals on the -dual cone
function dual_ascent(p, λ, ρ)
	λ_ = λ + ρ*p
	λ_ = Πsoc_dual(λ)
	return λ_
end

# Cone penalty
function cone_penalty(p,λ)
    primal_status = primal_cone_status(p)
    dual_status = dual_cone_status(λ)
	pen = zero(p)

	if primal_status == :below || dual_status == :inside
		pen = p
	elseif primal_status == :inside && dual_status == :origin
		pen = zero(p)
	elseif primal_status == :outside || dual_status == :boundary
		pen = p - Πsoc(p)
		# DELETE
		# if dual_status == :origin # can remove this test
		# 	pen = p - Πsoc(p)
		# else
		# 	# pen = cone_surface_dist(p, λ) #not working ####################################################
		# 	pen = p - Πsoc(p)
		# end
		# \DELETE
	end
    return pen
end

"Second-order cone projection of the primals on the primal cone"
function Πsoc(p)
	pv = p[1:end-1]
	ps = p[end]
	if norm(pv) <= -ps
		# @warn "below cone"
		return zero(pv), 0.0
	elseif norm(pv) <= ps
		# @warn "in cone"
		return p
	elseif norm(pv) > abs(ps)
		# @warn "outside cone"
		a = 0.5*(1.0 + ps/norm(pv))
		return [a*pv; a*norm(pv)]
	else
		@warn "soc projection error"
		return zero(p)
	end
end

"Second-order cone projection of the duals on the -dual cone"
function Πsoc_dual(λ)
	v = λ[1:end-1]
	s = λ[end]
	if norm(v) <= -s
		# @warn "in dual cone"
		return λ
	elseif norm(v) <= s
		# @warn "above dual cone"
		return zero(λ)
	elseif norm(v) > abs(s)
		# @warn "outside dual cone"
		a = 0.5*(1.0 + -s/norm(v))
		return [a*v; -a*norm(v)]
	else
		@warn "mult projection error"
		return zero(λ)
	end
end

# Not working , this is useless
# function cone_surface_dist(p, λ)
# 	dual_ray = λ / norm(λ,2)
# 	primal_ray = [dual_ray[1:end-1]; - dual_ray[end]]
#
# 	p1 = p  - p'*primal_ray * primal_ray
# 	p2 = p1 - p1'*dual_ray  * dual_ray
# 	pen = p2#####################################################################
# 	# pen = p1# nope#####################################################################
# 	return pen
# end

# Give the status of λ wrt to the -dual cone.
function dual_cone_status(λ)
	λv = λ[1:end-1]
	λs = λ[end]
	status = :unknown
	eps = 1e-8
	if (λs == 0. && λv == zero(λv))
		# @warn "on the dual cone origin"
		status = :origin
	elseif norm(λv) + λs + eps < 0
		# @warn "strictly in dual cone"
		status = :inside
	elseif abs(norm(λv) + λs) <= eps
		# @warn "on the boundary of dual cone"
		status = :boundary
	else
		@warn "strictly outside dual cone"
		status = :outside
	end
	return status
end

# Give the status of the primals p, wrt the primal cone
function primal_cone_status(p)
	pv = p[1:end-1]
	ps = p[end]
	status = :unknown
	eps = 1e-8
	if norm(pv) <= -ps
		# @warn "below primal cone"
		status = :below
	elseif norm(pv) <= ps
		# @warn "in primal cone"
		status = :inside
	elseif norm(pv) > abs(ps)
		# @warn "outside primal cone"
		status = :outside
	else
		@warn "primal cone status error"
		status = :error
	end
	return status
end

# Solver
function newton_solver()
    rate = 10.0

	p = rand(n)
    λ = 1e-8*rand(n)
	ρ = 1.0
    z = pack_z(p, λ, ρ)

	N = 6
    M = 20
    for k = 1:N
        for l = 1:M
            grad = G_res(z)
            @show norm(grad, 1)
            norm(grad, 1) <= 1e-10 && l > 10 ? break : nothing
            hess = H_res(z)
            Δz = - (hess + 1e-8*I) \ grad

            α = 1.0
            j = 1
            while j < 25
                z_trial = deepcopy(z)
                z_trial[1:n] += α*Δz
                if norm(G_res(z_trial),1) <= (1.0-α*0.1)*norm(grad,1)
                    break
                else
                    α *= 0.5
                    j += 1
                end
            end
            z[1:n] += α*Δz

        end
        p,λ,ρ = unpack_z(z)
        k == N ? break : nothing
		# Dual ascent
		λ = dual_ascent(p, λ, ρ)
        ρ *= rate
        z = pack_z(p, λ, ρ)
    end
    return z
end



# Test
Random.seed!(100)
# primals
p_test = rand(n)
# duals
λ_test = rand(n)
# penalty
ρ_test = 10.0

p_test = [1., 1., 0.1]
λ_test = [1., 1., -sqrt(2)]
z_test = pack_z(p_test, λ_test, ρ_test)

# objective
f_test = f(p_test)
# Lagragian
L_test = L(p_test, λ_test, ρ_test)
L_test = L(z_test)
grad_test = G_res(z_test)
hess_test = H_res(z_test)
isposdef(hess_test)

# Check status
primal_cone_status(p_test)
dual_cone_status(λ_test)


# Test the solver
z_sol = newton_solver()

p_sol, λ_sol, ρ_sol = unpack_z(z_sol)

L(z_sol)
norm(G_res(z_sol), 1) <= 1e-6
H_res(z_sol)

# Check constraint satisfaction
# The primals are close the the cone, slightly outside
primal_cone_status(p_sol) # it should be slighlty outside
abs(norm(p_sol[1:end-1], 2) - p_sol[end]) < 1e-5
abs(norm(p_sol[1:end-1], 2) - p_sol[end])
# the primals should be on the boundary on the -dual cone
dual_cone_status(λ_sol)
abs(norm(λ_sol[1:end-1], 2) + λ_sol[end]) < 1e-5

@show p_sol - Πsoc(p0) # should be close to [0,0,0]




# Test 1
p_test = [1., 1., 2.]
λ_test = zeros(n)
cone_penalty(p_test, λ_test) == zeros(n)

# Test 2
p_test = [1., 1., -2.]
λ_test = zeros(n)
cone_penalty(p_test, λ_test) == p_test

p_test = [1., 1., -2.]
λ_test = [1., 1., -sqrt(2)]
cone_penalty(p_test, λ_test) == p_test

p_test = [1., 1., -2.]
λ_test = [1., 1., -2.]
cone_penalty(p_test, λ_test) == p_test

p_test = [1., 1., 0.1]
λ_test = [1., 1., -2.]
cone_penalty(p_test, λ_test) == p_test

p_test = [1., 1., 10.]
λ_test = [1., 1., -2.]
cone_penalty(p_test, λ_test) == p_test

# Test 3
p_test = [1., 1., 0.1]
λ_test = zeros(n)
cone_penalty(p_test, λ_test) == p_test - Πsoc(p_test)

p_test = [1., 1., 0.1]
λ_test = [1., 1., -sqrt(2)]
cone_penalty(p_test, λ_test) == p_test - Πsoc(p_test)

p_test = [1., 1., 10.]
λ_test = [1., 1., -sqrt(2)]
cone_penalty(p_test, λ_test) == p_test - Πsoc(p_test)

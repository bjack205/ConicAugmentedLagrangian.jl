abstract type AbstractCone end
struct SecondOrderCone <: AbstractCone end
struct NormCone{T} <: AbstractCone t::T end
struct NegativeOrthant <: AbstractCone end

projection(::NegativeOrthant, x) = min.(0, x)
jacobian(::NegativeOrthant, x) = Diagonal(x .< 0)

projection(::SecondOrderCone, x) = Πsoc(x)
Base.in(x, ::SecondOrderCone) = in_soc(x)
jacobian(::SecondOrderCone, x) = jac_soc(x)
hessian(::SecondOrderCone, x, b) = hess_soc(x, b)

function projection(cone::NormCone, x)
    t = cone.t 
    if norm(x) <= t
        return x
    else
        return x / norm(x) * t
    end
end

function jacobian(cone::NormCone, x)
    if norm(x) <= t
        return I(length(x))
    else
        return t*(I - (x*x')/(x'x))/norm(x)
    end
end

function hessian(cone::NormCone, x, b)
    t = cone.t
    n = length(x)
    if norm(x) <= t
        return zeros(n,n)
    else
        return -t*(I - (x*x')/(x'x))'b/norm(x)^3 * x' +
            t*((x*(x'b))/(x'x)^2 * 2x' - (x*b' + I*(x'b))/(x'x))/norm(x)
    end
end

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

function DoubleIntegrator(D,N)
    """
    Objective
    """
    n,m = 2D,D
    Q = Diagonal(@SVector ones(n))
    R = Diagonal(@SVector fill(0.1, m))
    Qf = Diagonal(@SVector ones(n))*N*100
    xf = [(@SVector fill(1, n ÷ 2)); (@SVector zeros(n ÷ 2))]
    g = Float64[]
    h = Float64[]
    c = 0.0
    for k = 1:N-1
        append!(g, -Q*xf)
        append!(g, zeros(m))
        append!(h, Q.diag)
        append!(h, R.diag)
        c += 0.5*xf'Q*xf
    end
    append!(g, -Qf*xf)
    append!(g, zeros(m))
    append!(h, Qf.diag)
    append!(h, R.diag)
    c += 0.5*xf'Qf*xf
    H = Diagonal(h)

    """
    Dynamics
    """
    n,m = 2D,D
    NN = N*(n+m)
    x0 = zeros(n)

    #  Continuous dynamics
    Ac = zeros(n,n)
    Bc = zeros(n,m)
    for i = 1:D
        Ac[i,D+i] = 1
        Bc[D+i,i] = 1
    end

    # Euler integration
    dt = 0.1
    Ad = I + Ac*dt
    Bd = Bc*dt

    iA = 1:n
    jA = iA
    jB = n .+ (1:m)
    jA2 = jA .+ (n+m)
    
    A = zeros(N*n, NN)
    b = zeros(N*n)
    A[iA,jA] .= I(n)
    b[iA] .= -x0
    iA = iA .+ n
    for k = 1:N-1
        A[iA,jA] .= Ad
        A[iA,jB] .= Bd
        A[iA,jA2] .= -I(n)
        iA = iA .+ n
        jA = jA .+ (n+m)
        jB = jB .+ (n+m)
        jA2 = jA2 .+ (n+m)
    end
    di_obj(x) = 0.5*x'H*x + g'x + c
    di_dyn(x) = A*x + b
    return di_obj, di_dyn
end

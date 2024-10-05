using NLsolve, Dates
#求解
function nls(func, params...; ini = [0.0])
    if typeof(ini) <: Number
        r = nlsolve((vout,vin)->vout[1]=func(vin[1],params...), [ini],ftol = 1.0e-8)
        v = r.zero[1]
    else
        r = nlsolve((vout,vin)->vout .= func(vin,params...), ini, ftol = 1.0e-8)
        v = r.zero
    end
    return v, r.f_converged
end
#空間前進差分
function diff_for(u, Dx)
    N = length(u)
    return vcat([u[1]/Dx],[(u[i+1]-u[i])/Dx  for i in 1:(N-2)],[-u[N-1]/Dx])
end
#空間中心1次差分
function diff1(u, Dx)
    N = length(u)
    return vcat([u[2]/(2*Dx)],[(u[i+1] - u[i-1])/(2*Dx) for i in 2:(N-2)],[-u[N-2]/(2*Dx)])
end
#空間中心2次差分
function diff2(u, Dx)
    N = length(u)
    return vcat([(u[2] - 2*u[1])/(2*Dx)],[(u[i+1] - 2*u[i] + u[i-1])/(Dx^2) for i in 2:(N-2)],[(-2*u[N-1] + u[N-2])/(2*Dx)])
end
#1段DVDM(u,v)
function h_DVDM1(U, U0, eps, Dt, Dx) #U=(u,v)
    N = Int64(length(U0)/2)
    u = vcat([U[k] for k in 1:(N-1)],[0.0])
    v = vcat([U[k] for k in N:(2*N-2)],[0.0])
    u0 = vcat([U0[k] for k in 1:(N-1)],[0.0])
    v0 = vcat([U0[k] for k in (N+1):(2*N-1)],[0.0])
    cd_u = diff1(u,Dx)
    cd_u0 = diff1(u0,Dx)
    cdd_u = diff2(u,Dx)
    cdd_u0 = diff2(u0,Dx)
    ret = zeros(Float64,2*(N-1))
    for k in 1:(N-1)
        ret[k] = (u[k]-u0[k])/Dt - (v[k]+v0[k])/2
        ret[N-1+k] = (v[k]-v0[k])/Dt - (cdd_u[k]+cdd_u0[k])/2 - eps*( 2*cdd_u[k]* cd_u[k] + cdd_u[k]* cd_u0[k] + cdd_u0[k]*cd_u[k] + 2*cdd_u0[k]* cd_u0[k])/6
    end
    return ret
end
#1段DVDM保存量
function G_DVDM1(U,eps,Dx)
    N = Int(length(U)/2)
    u = [U[k] for k in 1:N]
    v = [U[k] for k in (N+1):(2*N)]
    fd_u = diff_for(u,Dx)
    return Dx*sum([v[k]^2/2 + (fd_u[k])^2/2 + eps*(fd_u[k])^3/6 for k in 1:N])
end
#2段DVDM(u)
function h_DVDM2(u, u_now, u_prev, eps, Dt, Dx) #U=(u,v)
    N = length(u_now)
    u = vcat(u,[0.0])
    fa_u = (u + u_now)/2
    ba_u = (u_now + u_prev)/2
    cd_fa_u = diff1(fa_u,Dx)
    cd_ba_u = diff1(ba_u,Dx)
    cdd_fa_u = diff2(fa_u,Dx)
    cdd_ba_u = diff2(ba_u,Dx)
    faba_cdd_u = (cdd_fa_u + cdd_ba_u)/2
    return [(u[k] - 2*u_now[k] + u_prev[k])/(Dt^2) - faba_cdd_u[k] - eps*(2*cdd_fa_u[k]*cd_fa_u[k] + cdd_fa_u[k]*cd_ba_u[k] + cdd_ba_u[k]*cd_fa_u[k] + 2*cdd_ba_u[k]* cd_ba_u[k])/6 for k in 1:(N-1)]
end
#1段DVDM保存量
function G_DVDM2(u,u_now, eps, Dt, Dx)
    N = length(u)
    fd_fa_u = diff_for((u + u_now)/2,Dx)
    return Dx*sum([(u[k] - u_now[k])^2/(2*Dt^2) + (fd_fa_u[k])^2/2 + eps*(fd_fa_u[k])^3/6 for k in 1:N])
end
#2段スキームRSC
function h_RSC(u, u_now, u_prev, eps, Dt, Dx) #U=(u,v)
    N = length(u_now)
    u = vcat(u,[0.0])
    cd_u = diff1(u,Dx)
    cdd_u = diff2(u,Dx)
    au = (u + u_now + u_prev)/3
    cd_au = diff1(au,Dx)
    cdd_au = diff2(au,Dx)
    cdd_ca_u = diff2((u + u_prev)/2,Dx)
    return [(u[k] - 2*u_now[k] + u_prev[k])/(Dt^2) - cdd_ca_u[k] - eps*( cdd_u[k]*cd_au[k] + cd_u[k]*cdd_au[k] )/6 for k in 1:(N-1)]
end
#2段スキーム保存量
function G_RSC(u, u_now, eps, Dt, Dx)
    N = length(u)
    fd_u = diff_for(u,Dx)
    fd_u_now = diff_for(u_now,Dx)
    return Dx*sum([(u[k]-u_now[k])^2/(Dt^2) + 0.5*(fd_u[k])^2 + 0.5*(fd_u_now[k])^2 + eps*(fd_u[k]^2*fd_u_now[k] + fd_u_now[k]^2*fd_u[k])/6 for k in 1:N])
end

#テスト関数
function com_test(N, m, Dt, Dx)
    return [(1 + π^2 + 0.25*π^3*cos(π*k*Dx)*exp(-m*Dt))*sin(π*k*Dx)*exp(-m*Dt) for k in 1:N]
end
#1段DVDMテスト
function h_DVDM1_test(U, U0, eps, m, Dt, Dx)
    N = Int64(length(U0)/2)
    ret = h_DVDM1(U, U0, eps, Dt, Dx)
    dif = com_test(N, m+0.5, Dt, Dx)
    for k in 1:(N-1)
        ret[N-1+k] -= dif[k]
    end
    return ret
end
#2段DVDMテスト
function h_DVDM2_test(u, u_now, u_prev, eps, m, Dt, Dx)
    N = length(u_now)
    ret = h_DVDM2(u, u_now, u_prev, eps, Dt, Dx)
    dif = com_test(N, m, Dt, Dx)
    for k in 1:(N-1)
        ret[k] -= dif[k]
    end
    return ret
end
#2段RSCテスト
function h_RSC_test(u, u_now, u_prev, eps, m, Dt, Dx)
    N = length(u_now)
    ret = h_RSC(u, u_now, u_prev, eps, Dt, Dx)
    dif = com_test(N, m, Dt, Dx)
    for k in 1:(N-1)
        ret[k] -= dif[k]
    end
    return ret
end
#1段DVDM
function FPU_DVDM1(U0, eps, M, Dt, Dx, test = false) # U0 = (u0,ut0)
    t_start = time()
    ulist = [U0]
    U_now = U0
    for m in 1:M
        ini0 = zeros(length(U0) - 2)
        if test
            U_next = nls(h_DVDM1_test, U_now, eps, m-1, Dt, Dx, ini = ini0)
        else
            U_next = nls(h_DVDM1, U_now, eps, Dt, Dx, ini = ini0)
        end
        U_next = U_next[1]
        u_next = U_next[1:(Int64(length(U0)/2)-1)]
        v_next = U_next[Int64(length(U0)/2):(length(U0) - 2)]
        U_next = vcat(u_next,[0.0],v_next,[0.0])
        println("DVDVM1, ", M, " ", m, " ", G_DVDM1(U_next, eps, Dx))
        push!(ulist,U_next)
        U_now = U_next
    end
    t_end = time()
    println("DVDN1_time ", t_end-t_start)
    return ulist
end

#2段DVDM
function FPU_DVDM2(u0, ut0, eps, M, Dt, Dx, test = false)
    t_start = time()
    ulist = [u0]
    N = length(u0)
    # u1 を計算
    cd_u0 = diff1(u0,Dx)
    cdd_u0 = diff2(u0,Dx)
    if test
        dif = com_test(N,0,Dt,Dx)
        u1 = vcat([u0[k] + Dt*ut0[k] + 0.5*Dt^2*((1 + eps*cd_u0[k])*cdd_u0[k] + dif[k]) for k in 1:(N-1)],[0.0])
    else
        u1 = vcat([u0[k] + Dt*ut0[k] + 0.5*Dt^2*(1 + eps*cd_u0[k])*cdd_u0[k] for k in 1:(N-1)],[0.0])
    end
    push!(ulist,u1)

    u_prev = u0; u_now = u1
    for m in 2:M
        ini0 = zeros(N - 1)
        if test
            u_next = nls(h_DVDM2_test, u_now, u_prev, eps, m-1, Dt, Dx, ini = ini0)
        else
            u_next = nls(h_DVDM2, u_now, u_prev, eps, Dt, Dx, ini = ini0)
        end
        u_next = vcat(u_next[1],[0.0])
        println("DVDM2, ", M, " ", m, " ", G_DVDM2(u_next, u_now, eps, Dt, Dx))
        push!(ulist,u_next)
        u_prev = u_now; u_now = u_next
    end
    t_end = time()
    println("DVDN2_time ", t_end-t_start)
    return ulist
end

#2段RSC
function FPU_RSC(u0, ut0, eps, M, Dt, Dx, test = false)
    t_start = time()
    ulist = [u0]
    N = length(u0)
    
    # u1 を計算
    cd_u0 = diff1(u0,Dx)
    cdd_u0 = diff2(u0,Dx)
    if test
        dif = com_test(N,0,Dt,Dx)
        u1 = vcat([u0[k] + Dt*ut0[k] + 0.5*Dt^2*((1+eps*cd_u0[k])*cdd_u0[k] + dif[k]) for k in 1:(N-1)],[0.0])
    else
        u1 = vcat([u0[k] + Dt*ut0[k] + 0.5*Dt^2*(1+eps*cd_u0[k])*cdd_u0[k] for k in 1:(N-1)],[0.0])
    end
    push!(ulist,u1)

    u_prev = u0; u_now = u1
    for m in 2:M
        ini0 = zeros(N - 1)
        if test
            u_next = nls(h_RSC_test, u_now, u_prev, eps, m-1, Dt, Dx, ini = ini0)
        else
            u_next = nls(h_RSC, u_now, u_prev, eps, Dt, Dx, ini = ini0)
        end
        u_next = vcat(u_next[1],[0.0])
        println("RSC, ", M, " ", m, " ", G_RSC(u_next, u_now, eps, Dt, Dx))
        push!(ulist,u_next)
        u_prev = u_now; u_now = u_next
    end
    t_end = time()
    println("RSC_time ",t_end-t_start)
    return ulist
end

function L2error(u,v,Dx)
    N = length(u)
    e = 0.0
    for k in 1:N
        e += (u[k] - v[k])^2*Dx
    end
    return e^0.5
end

function testing1(M, Dt, Dx)
    #L = 1
    N = Int64(ceil(1/Dx))
    u0 = vcat([sin(π*k*Dx) for k in 1:(N-1)],[0.0])
    ut0 = vcat([-sin(π*k*Dx) for k in 1:(N-1)],[0.0])
    U0 = vcat(u0,ut0)
    u_DVDM1 = FPU_DVDM1(U0, 0.25, M, Dt, Dx, true)
    u_DVDM2 = FPU_DVDM2(u0, ut0, 0.25, M, Dt, Dx, true)
    u_RSC = FPU_RSC(u0, ut0, 0.25, M, Dt, Dx, true)
    u_true = [vcat([sin(π*k*Dx)*exp(-m*Dt) for k in 1:(N-1)],[0.0]) for m in 0:M]
    error_DVDM1, error_DVDM2, error_RSC = 0.0, 0.0, 0.0
    for m in 1:(length(u_true))
        error_DVDM1 = max(error_DVDM1, L2error(u_DVDM1[m][1:N], u_true[m], Dx))
        error_DVDM2 = max(error_DVDM2, L2error(u_DVDM2[m], u_true[m], Dx))
        error_RSC = max(error_RSC, L2error(u_RSC[m], u_true[m], Dx))
    end
    println("M=",M,", Dt=", Dt, ", Dx=",Dx)
    println(error_DVDM1, " ", error_DVDM2, " ", error_RSC)
end

function testing2(t,M)
    L = 1; Dt = 0.1; Dx = 0.1
    N = Int64(ceil(L/Dx))
    ulist_DVDM1 = []; ulist_DVDM2 = []; ulist_RSC = []
    for i in 1:t
        u0 = vcat([sin((1/L)*π*k*Dx*0.5^(i-1)) for k in 1:(N*2^(i-1)-1)],[0.0])
        ut0 = zeros(Float64,N*2^(i-1))
        U0 = vcat(u0,ut0)
        push!(ulist_DVDM1, FPU_DVDM1(U0, 0.25, M*2^(i-1), Dt*0.5^(i-1), Dx*0.5^(i-1)))
        push!(ulist_DVDM2, FPU_DVDM2(u0, ut0, 0.25, M*2^(i-1), Dt*0.5^(i-1), Dx*0.5^(i-1)))
        push!(ulist_RSC, FPU_RSC(u0, ut0, 0.25, M*2^(i-1), Dt*0.5^(i-1), Dx*0.5^(i-1)))
    end
    for i in 1:(t-1)
        error_DVDM1, error_DVDM2, error_RSC = 0.0, 0.0, 0.0
        for m in 1:(length(ulist_DVDM1[i]))
            u1_DVDM1 = ulist_DVDM1[i][m][1:N*2^(i-1)]
            u2_DVDM1 = [ulist_DVDM1[t][1+(m-1)*2^(t-i)][k*2^(t-i)] for k in 1:(N*2^(i-1))]
            error_DVDM1 = max(error_DVDM1, L2error(u1_DVDM1, u2_DVDM1, Dx*0.5^(i-1)))
            u1_DVDM2 = ulist_DVDM2[i][m][1:N*2^(i-1)]
            u2_DVDM2 = [ulist_DVDM2[t][1+(m-1)*2^(t-i)][k*2^(t-i)] for k in 1:(N*2^(i-1))]
            error_DVDM2 = max(error_DVDM2, L2error(u1_DVDM2, u2_DVDM2, Dx*0.5^(i-1)))
            u1_RSC = ulist_RSC[i][m][1:N*2^(i-1)]
            u2_RSC = [ulist_RSC[t][1+(m-1)*2^(t-i)][k*2^(t-i)] for k in 1:(N*2^(i-1))]
            error_RSC = max(error_RSC, L2error(u1_RSC, u2_RSC, Dx*0.5^(i-1)))
        end
        println(error_DVDM1, " ", error_DVDM2, " ", error_RSC)
    end
end

# checking u1
# function testing0(Dx)
#     N = Int64(ceil(1/Dx))
#     u0 = vcat([sin(π*k*Dx) for k in 1:(N-1)],[0.0])
#     ut0 = vcat([-sin(π*k*Dx) for k in 1:(N-1)],[0.0])
#     cd_u0 = diff1(u0,Dx)
#     cdd_u0 = diff2(u0,Dx)
#     dif = com_test(N,0,Dx,Dx)
#     u1 = vcat([u0[k] + Dx*ut0[k] + 0.5*Dx^2*((1 + 0.25*cd_u0[k])*cdd_u0[k] + dif[k]) for k in 1:(N-1)],[0.0])
#     u_true = vcat([sin(π*k*Dx)*exp(-Dx) for k in 1:(N-1)],[0.0])
#     println(L2error(u1,u_true,Dx))
# end

#testing1(80, 0.0125, 0.0125)
testing2(4,5)
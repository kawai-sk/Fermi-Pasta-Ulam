# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Julia 1.11.2
#     language: julia
#     name: julia-1.11
# ---

# %%
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

# %%
#空間前進差分
function diff_for(u, Dx)
    N = length(u)
    return vcat([(u[i+1]-u[i])/Dx  for i in 1:(N-1)],[(u[1]-u[N])/Dx])
end
#空間後退差分
function diff_back(u, Dx)
    N = length(u)
    return vcat([(u[1]-u[N])/Dx],[(u[i]-u[i-1])/Dx  for i in 2:N],)
end
#空間中心1次差分
function diff1(u, Dx)
    N = length(u)
    return vcat([(u[2]-u[N])/(2*Dx)],[(u[i+1] - u[i-1])/(2*Dx) for i in 2:(N-1)],[(u[1]-u[N-1])/(2*Dx)])
end
#空間中心2次差分
function diff2(u, Dx)
    N = length(u)
    return vcat([(u[2] - 2*u[1] + u[N])/(Dx^2)],[(u[i+1] - 2*u[i] + u[i-1])/(Dx^2) for i in 2:(N-1)],[(u[1]-2*u[N] + u[N-1])/(Dx^2)])
end
#誤差
function L2error(u,v,Dx)
    N = length(u)
    e = 0.0
    for k in 1:N
        e += (u[k] - v[k])^2*Dx
    end
    return e^0.5
end

# %%
#1段DVDM(u,v)
function h_DVDM1(U, U0, eps, Dt, Dx) #U=(u,v)
    N = Int64(length(U0)/2)
    u = [U[k] for k in 1:N]
    v = [U[k] for k in (N+1):(2*N)]
    u0 = [U0[k] for k in 1:N]
    v0 = [U0[k] for k in (N+1):(2*N)]
    cd_u = diff1(u,Dx)
    cd_u0 = diff1(u0,Dx)
    cdd_u = diff2(u,Dx)
    cdd_u0 = diff2(u0,Dx)
    ret = zeros(Float64,2*N)
    for k in 1:N
        ret[k] = (u[k]-u0[k])/Dt - (v[k]+v0[k])/2
        ret[N+k] = (v[k]-v0[k])/Dt - (cdd_u[k]+cdd_u0[k])/2 - eps*( 2*cdd_u[k]* cd_u[k] + cdd_u[k]* cd_u0[k] + cdd_u0[k]*cd_u[k] + 2*cdd_u0[k]* cd_u0[k])/6
    end
    return ret
end
#1段DVDM保存量
function G_DVDM1(U,eps,Dx)
    N = Int(length(U)/2)
    u = [U[k] for k in 1:N]
    v = [U[k] for k in (N+1):(2*N)]
    bd_u = diff_back(u,Dx)
    return Dx*sum([v[k]^2/2 + (bd_u[k])^2/2 + eps*(bd_u[k])^3/6 for k in 1:N])
end

# %%
#2段DVDM(u)
function h_DVDM2(u, u_now, u_prev, eps, Dt, Dx) #U=(u,v)
    N = length(u_now)
    fa_u = (u + u_now)/2
    ba_u = (u_now + u_prev)/2
    cd_fa_u = diff1(fa_u,Dx)
    cd_ba_u = diff1(ba_u,Dx)
    cdd_fa_u = diff2(fa_u,Dx)
    cdd_ba_u = diff2(ba_u,Dx)
    faba_cdd_u = (cdd_fa_u + cdd_ba_u)/2
    left = (u - 2*u_now + u_prev)/(Dt^2) - faba_cdd_u
    rest = [2*cdd_fa_u[k]*cd_fa_u[k] + cdd_fa_u[k]*cd_ba_u[k] + cdd_ba_u[k]*cd_fa_u[k] + 2*cdd_ba_u[k]*cd_ba_u[k] for k in 1:N]
    return left - rest*eps/6
end
#2段DVDM保存量
function G_DVDM2(u,u_now, eps, Dt, Dx)
    N = length(u)
    fd_fa_u = diff_for((u + u_now)/2,Dx)
    return Dx*sum([(u[k] - u_now[k])^2/(2*Dt^2) + (fd_fa_u[k])^2/2 + eps*(fd_fa_u[k])^3/6 for k in 1:N])
end

# %%
#2段スキームRSC
function h_RSC(u, u_now, u_prev, eps, Dt, Dx) #U=(u,v)
    N = length(u_now)
    cd_u = diff1(u,Dx)
    cdd_u = diff2(u,Dx)
    au = (u + u_now + u_prev)/3
    cd_au = diff1(au,Dx)
    cdd_au = diff2(au,Dx)
    cdd_ca_u = diff2((u + u_prev)/2,Dx)
    left = (u - 2*u_now + u_prev)/(Dt^2) - cdd_ca_u
    rest = [cdd_u[k]*cd_au[k] + cd_u[k]*cdd_au[k] for k in 1:N]
    return left - eps*rest/6
end
function h_RSC_initial(u, u0, ut0, eps, Dt, Dx) #U=(u,v)
    N = length(u0)
    cd_u0 = diff1(u0,Dx)
    cdd_au = diff2((u+u0)/2,Dx)
    left = 2*((u - u0)/Dt - ut0) - cdd_au
    rest = [ eps*cd_u0[k]*cdd_au[k] for k in 1:N]
    return left - rest
end
#2段スキーム保存量
function G_RSC(u, u_now, eps, Dt, Dx)
    N = length(u)
    bd_u = diff_back(u,Dx)
    bd_u_now = diff_back(u_now,Dx)
    return Dx*sum([(u[k]-u_now[k])^2/(Dt^2) + 0.5*(bd_u[k])^2 + 0.5*(bd_u_now[k])^2 + eps*(bd_u[k]^2*bd_u_now[k] + bd_u_now[k]^2*bd_u[k])/6 for k in 1:N])
end

# %%
#テスト関数
function com_test(N, m, Dt, Dx)
    return [(1 + π^2 + 0.25*π^3*cos(π*k*Dx)*exp(-m*Dt))*sin(π*k*Dx)*exp(-m*Dt) for k in 1:N]
end
#1段DVDMテスト
function h_DVDM1_test(U, U0, eps, m, Dt, Dx)
    N = Int64(length(U0)/2)
    ret = h_DVDM1(U, U0, eps, Dt, Dx)
    dif = com_test(N, m+0.5, Dt, Dx)
    for k in 1:N
        ret[N+k] -= dif[k]
    end
    return ret
end
#2段DVDMテスト
function h_DVDM2_test(u, u_now, u_prev, eps, m, Dt, Dx)
    N = length(u_now)
    ret = h_DVDM2(u, u_now, u_prev, eps, Dt, Dx)
    dif = com_test(N, m, Dt, Dx)
    return ret - dif
end
#2段RSCテスト
function h_RSC_test(u, u_now, u_prev, eps, m, Dt, Dx)
    N = length(u_now)
    ret = h_RSC(u, u_now, u_prev, eps, Dt, Dx)
    dif = com_test(N, m, Dt, Dx)
    return ret - dif
end
function h_RSC_initial_test(u, u_now, u_prev, eps, Dt, Dx) #U=(u,v)
    N = length(u_now)
    ret = h_RSC_initial(u, u_now, u_prev, eps, Dt, Dx)
    dif = com_test(N, m, Dt, Dx)
    return ret - dif
end

# %%
#1段DVDM
function FPU_DVDM1(U0, eps, M, Dt, Dx, test = false) # U0 = (u0,ut0)
    t_start = time()
    ulist = [U0]
    U_now = U0
    ene0 = G_DVDM1(U_now, eps, Dx)
    max_err = 0
    for m in 1:M
        ini0 = 0.001*zeros(length(U0))
        if test
            U_next = nls(h_DVDM1_test, U_now, eps, m-1, Dt, Dx, ini = U_now+ini0)
        else
            U_next = nls(h_DVDM1, U_now, eps, Dt, Dx, ini = U_now+ini0)
        end
        U_next = U_next[1]
        u_next = U_next[1:(Int64(length(U0)/2))]
        v_next = U_next[(Int64(length(U0)/2)+1):length(U0)]
        U_next = vcat(u_next,v_next)
        ene_now = G_DVDM1(U_next, eps, Dx)
        ratio = abs((ene_now - ene0)/ene0)
        max_err = max(max_err,ratio)
        if m%10 == 0
            println("DVDVM1, ", M, " ", m, " ", ratio)
        end
        push!(ulist,U_next)
        U_now = U_next
    end
    t_end = time()
    println("DVDM1_time ", t_end-t_start, ",保存量の変化比率最大値: ",max_err)
    return ulist
end

# %%
#2段DVDM
function FPU_DVDM2(u0, ut0, eps, M, Dt, Dx, test = false)
    println(length(u0), " ", M," ",Dt," ",Dx)
    t_start = time()
    ulist = [u0]
    N = length(u0)
    # u1 を計算
    cd_u0 = diff1(u0,Dx)
    cdd_u0 = diff2(u0,Dx)
    if test
        dif = com_test(N,0,Dt,Dx)
        u1 = [u0[k] + Dt*ut0[k] + 0.5*Dt^2*((1 + eps*cd_u0[k])*cdd_u0[k] + dif[k]) for k in 1:N]
    else
        u1 = [u0[k] + Dt*ut0[k] + 0.5*Dt^2*(1 + eps*cd_u0[k])*cdd_u0[k] for k in 1:N]
    end
    push!(ulist,u1)

    u_prev = u0; u_now = u1
    ene0 = G_DVDM2(u_now, u_prev, eps, Dt, Dx)
    max_err = 0
    for m in 2:M
        ini0 = 0.001*ones(N)
        if test
            u_next = nls(h_DVDM2_test, u_now, u_prev, eps, m-1, Dt, Dx, ini = u_now+ini0)
        else
            u_next = nls(h_DVDM2, u_now, u_prev, eps, Dt, Dx, ini = u_now+ini0)
        end
        u_next = u_next[1]
        ene_now = G_DVDM2(u_next, u_now, eps, Dt, Dx)
        ratio = abs((ene_now - ene0)/ene0)
        max_err = max(max_err,ratio)
        if m%10 == 0
            println("DVDM2, ", M, " ", m, " ", ratio)
        end
        push!(ulist,u_next)
        u_prev = u_now; u_now = u_next
    end
    t_end = time()
    println("DVDM2_time ", t_end-t_start, ",保存量の変化比率最大値: ",max_err)
    return ulist
end

# %%
#2段RSC
function FPU_RSC(u0, ut0, eps, M, Dt, Dx, test = false, original = false)
    println(length(u0), " ", M," ",Dt," ",Dx)
    t_start = time()
    ulist = [u0]
    N = length(u0)
    
    # # u1 を計算
    # Taylor
    # 
    if original
        ini0 = 0.001*ones(N)
        if test
            u_next = nls(h_RSC_initial_test, u0, ut0, eps, Dt, Dx, ini = u0+ini0)
        else
            u_next = nls(h_RSC_initial, u0, ut0, eps, Dt, Dx, ini = u0+ini0)
        end
        u1 = u_next[1]
    else
        cd_u0 = diff1(u0,Dx)
        cdd_u0 = diff2(u0,Dx)
        if test
            dif = com_test(N,0,Dt,Dx)
            u1 = [u0[k] + Dt*ut0[k] + 0.5*Dt^2*((1+eps*cd_u0[k])*cdd_u0[k] + dif[k]) for k in 1:N]
        else
            u1 = [u0[k] + Dt*ut0[k] + 0.5*Dt^2*(1+eps*cd_u0[k])*cdd_u0[k] for k in 1:N]
        end
    end
    
    push!(ulist,u1)

    u_prev = u0; u_now = u1
    ene0 = G_RSC(u_now, u_prev, eps, Dt, Dx)
    max_err = 0
    for m in 2:M
        ini0 = 0.001*ones(N)
        if test
            u_next = nls(h_RSC_test, u_now, u_prev, eps, m-1, Dt, Dx, ini = u_now+ini0)
        else
            u_next = nls(h_RSC, u_now, u_prev, eps, Dt, Dx, ini = u_now+ini0)
        end
        u_next = u_next[1]
        ene_now = G_RSC(u_next, u_now, eps, Dt, Dx)
        ratio = abs((ene_now - ene0)/ene0)
        max_err = max(max_err,ratio)
        if m%10 == 0
            println("RSC, ", M, " ", m, " ", ratio)
        end
        push!(ulist,u_next)
        u_prev = u_now; u_now = u_next
    end
    t_end = time()
    println("RSC_time ",t_end-t_start, ",保存量の変化比率最大値: ",max_err)
    return ulist
end

# %%
function testing1(T, Dt, Dx)
    #L = 1
    N = Int64(ceil(1/Dx)); M = Int64(ceil(T/Dt))
    u0 = [sin(π*k*Dx) for k in 1:N]
    ut0 = [-sin(π*k*Dx) for k in 1:N]
    U0 = vcat(u0,ut0)
    u_DVDM1 = FPU_DVDM1(U0, 0.25, M, Dt, Dx, true)
    u_DVDM2 = FPU_DVDM2(u0, ut0, 0.25, M, Dt, Dx, true)
    u_RSC = FPU_RSC(u0, ut0, 0.25, M, Dt, Dx, true)
    u_true = [[sin(π*k*Dx)*exp(-m*Dt) for k in 1:N] for m in 0:M]
    error_DVDM1, error_DVDM2, error_RSC = 0.0, 0.0, 0.0
    for m in 1:(length(u_true))
        error_DVDM1 = max(error_DVDM1, L2error(u_DVDM1[m][1:N], u_true[m], Dx))
        error_DVDM2 = max(error_DVDM2, L2error(u_DVDM2[m], u_true[m], Dx))
        error_RSC = max(error_RSC, L2error(u_RSC[m], u_true[m], Dx))
    end
    println("M=",M,", Dt=", Dt, ", Dx=",Dx)
    println(error_DVDM1, " ", error_DVDM2, " ", error_RSC)
end

# %%
testing2(1,6,1,false)

# %%
import Pkg; Pkg.add("Plots")

# %%
function testing2(L,t,T)
    Dt = 0.1; Dx = 0.1
    N = Int64(ceil(L/Dx)); M = Int64(ceil(T/Dx))
    ulist_DVDM1 = []; ulist_DVDM2 = []; ulist_RSC = []
    println(N,M)
    for i in 1:t
        u0 = [sin((1/L)*2*π*k*Dx*0.5^(i-1)) for k in 1:(N*2^(i-1))]
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

# %%
using Plots
gr()
function graphing2_comparing(Dt,L,T)
    Dx = Dt
    N = Int64(ceil(L/Dx))
    M = Int64(ceil(T/Dt))
    u0 = [sin((1/L)*2*π*k*Dx) for k in 1:N]
    ut0 = zeros(Float64,N)
    U0 = vcat(u0,ut0)
    u1 = FPU_DVDM1(U0, 0.25, M, Dt, Dx)
    u2 = FPU_DVDM2(u0, ut0, 0.25, M, Dt, Dx)
    uR = FPU_RSC(u0, ut0, 0.25, M, Dt, Dx)
    anim = @animate for i in 1:length(u1)
        plt = plot(u1[i][1:N],label="DVDM1")
        plot!(plt,u2[i],label="DVDM2")
        plot!(plt,uR[i],label="RSC")
        plot(plt,title = "(L,Dt,t)=("*string(L)*","*string(Dt)*","*string(round((i-1)*Dt,digits=3))*")")
    end
    text = string("FPU_Dt=")*string(Dt)*"_L="*string(L)*"_T="*string(T)*(".gif")
    gif(anim, text, fps=20)
end

# %%
graphing2_comparing(0.025,1,1)

# %%
testing2(1,4,1)

# %%

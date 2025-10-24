using RCall 
R"""
library(igraph)
"""

# 
function blockstr(p, diag, offdiag, bs)
    x, y = zeros(p, p), zeros(bs, bs)

    for i in 1 : bs
        for j in (i + 1) : bs
            y[i, j] = y[j, i] = offdiag ^ abs(i - j)# 0.9 - 0.05 * abs(i - j)
        end
    end

    for i = 1 : Int(p / bs); x[(bs * (i - 1) + 1): (bs * i), (bs * (i - 1) + 1): (bs * i)] = y; end
    for i = 1 : p; x[i, i] = diag; end
    return x
end

# Normal distribution
const constant = 0.5 * log(2.0 * pi)
normpdf(x) = exp(-0.5 * x * x - constant)
normpdf(x, mu, sigma) = normpdf((x - mu) / sigma) / sigma
log_normpdf(x, mu, sigma) = (z = (x - mu) / sigma;  -0.5 * z * z - log(sigma) - constant)

function index_j(j, ind)
    x, i = zeros(Int32, length(ind) - 1), 1
    @inbounds for s in 1 : length(ind); if s != j; x[i] = s; i += 1; end; end
    x
end

function f_add_diagonal_(x, c)
    xx = copy(x)
    @inbounds for s = 1 : size(x)[1]; xx[s, s] += 1 / c[s]; end
    xx
end

function update_precision!(ind, Omega, cov, V, p, lambda, S, n)

    for j = 1 : p

        ind_j = index_j(j, ind)

        C11 = cov[ind_j, ind_j]
        C12 = cov[ind_j, j] 
        O11_ = C11 - C12 * C12' / cov[j, j]

        C_ = (S[j, j] + lambda) * O11_        
        C_ = f_add_diagonal_(C_, V[ind_j, j])
        C_ = 0.5 * (C_ + C_')  
        C_chol = cholesky(C_)
        mu = -(C_chol \ S[ind_j, j])

        rn = randn(p - 1)
        w12 = mu + C_chol.U \ rn   
        Omega[ind_j, j] = Omega[j, ind_j] = w12

        nu = rand(Gamma(0.5 * n + 1.0, 2 / (S[j, j] + lambda)))
        Omega[j, j] = nu + dot(w12, O11_, w12)
        
        ##### Updating covariance matrix 
        O11_w12 = O11_ * w12
        cjj = 1 / nu
        cov[ind_j, ind_j] = O11_ + O11_w12 * O11_w12' * cjj
        cov[j, ind_j] = cov[ind_j, j] = - O11_w12 * cjj
        cov[j, j] = cjj     
    end
    
end

function update_G!(G, Omega, p_g, nu1, nu2, nu12, nu22, V, p)
    
    for j = 1 : p
        for l = (j + 1) : p
            w = Omega[l, j]
            logO = log_normpdf(w, 0.0, nu1) + log(1 - p_g) - (log_normpdf(w, 0.0, nu2) + log(p_g))
            success_p = 1 / (1 + exp(logO))
            g = rand(Bernoulli(success_p))
            if g != G[l, j]
                G[l, j] = G[j, l] = ifelse(g == true, 1, 0)
                if g == 1; V[l, j] = V[j, l] = nu22; else V[l, j] = V[j, l] = nu12; end;
            end
        end
    end
    
end

function update_G!(G, Rstar, Omega, nu1, nu2, nu12, nu22, V, p, ag, bg, Ltri)
    Gstar = G[Ltri]
    
    k = 1
    for j = 1 : p
        for l = (j + 1) : p
            w = Omega[l, j]
            cc = ftn3(Gstar, Rstar, k)
            logO = log_normpdf(w, 0.0, nu1) - (log_normpdf(w, 0.0, nu2) + ag + 2 * bg * cc)
            success_p = 1 / (1 + exp(logO))
            g = rand(Bernoulli(success_p))
            
            if g != Gstar[k]
                Gstar[k] = ifelse(g == true, 1, 0)
                G[l, j] = G[j, l] = ifelse(g == true, 1, 0)
                if g == 1; V[l, j] = V[j, l] = nu22; else V[l, j] = V[j, l] = nu12; end;
            end
            k += 1
        end
    end
end

function sum_Mb_j(v, A, j)
    x = zeros(size(A, 1))
    @inbounds for s in 1 : size(A, 2)
        for i = 1 : size(A, 1); if s != j; x[i] += A[i, s] * v[s]; end; end
    end
    x
end

function sum_j(v, A, j)
    x = 0.0
    @inbounds for s in 1 : size(A, 2); if s != j; x += A[s, j] * v[s]; end; end
    x
end

function sum_at_j(v, A, j)
    x = 0.0
    @inbounds for i = 1 : size(A, 1); x += A[i, j] * v[i]; end
    x
end

function ftn3(v1, A, k)
    x = 0.0
    @inbounds for s = 1 : length(v1)
        if s != k
            x += v1[s]  * A[s, k] 
        end
    end
    x
end 


function update_bmae!(bm, ae, b, a, r1, r21, r22, Ebe, Omega, sigma2, sigma2_, pi1, pi2, y, M, M2, sum_M2, E, sum_E2, p, sb, sa, sb2, sa2)
    
    heart = M * Omega
    
    for j in  1 : p
        
        Mbm_j = sum_Mb_j(bm, M, j)
        star = y - (Mbm_j + Ebe) 
        sumMjstar = dot(M[:, j], star)

        bm1 = r21[j] * b[j] 
        ae1 = r22[j] * a[j]
        
        bm2sumM2j = bm1^2 * sum_M2[j]
        c1 = bm2sumM2j - 2.0 * bm1 * sumMjstar

        sumaeOmega_j = sum_j(ae, Omega, j)
        sumEheart = sum_at_j(E, heart, j)
        c2 = Omega[j, j] * ae1^2 * sum_E2 - 2.0 * ae1 * sumEheart + 2.0 * ae1 * sumaeOmega_j * sum_E2        
        
        # update r1
        logO = log(1 - pi1) - (log(pi1) - 0.5 * sigma2_ * c1 - 0.5 * c2)
        success_p = 1 / (1 + exp(logO))
        r1[j] = rand(Bernoulli(success_p))

        bm[j] = r1[j] * r21[j] * b[j]
        ae[j] = r1[j] * r22[j] * a[j]

        # update r21
        if r1[j] == 0; c1 = 0 else bm2sumM2j = b[j]^2 * sum_M2[j]; c1 = bm2sumM2j - 2.0 * b[j] * sumMjstar; end
        logO = log(1 - pi2) - (log(pi2) - 0.5 * sigma2_ * c1)
        success_p = 1 / (1 + exp(logO))
        r21[j] = rand(Bernoulli(success_p))
        bm[j] = r1[j] * r21[j] * b[j]
        
        # update r22
        if r1[j] == 0; c2 = 0; else c2 = Omega[j, j] * a[j]^2 * sum_E2 - 2.0 * a[j] * sumEheart + 2.0 * a[j] * sumaeOmega_j * sum_E2; end
        logO = log(1 - pi2) - (log(pi2) - 0.5 * c2)
        success_p = 1 / (1 + exp(logO))
        r22[j] = rand(Bernoulli(success_p))
        ae[j] = r1[j] * r22[j] * a[j]
        
        # update b
        if (r1[j] * r21[j]) == 1
            c3_ = 1 / (sb2 * sum_M2[j] + sigma2)
            s2_b = sigma2 * sb2 * c3_
            mu_b = sb2 * sumMjstar * c3_
            b[j] = rand(Normal(mu_b, sqrt(s2_b)))
        else 
            b[j] = rand(Normal(0, sb))
        end
        bm[j] = r1[j] * r21[j] * b[j]

        #update a
        if (r1[j] * r22[j]) == 1 
            c4_ = 1 / (sa2 * Omega[j, j] * sum_E2 + 1)
            s2_a = sa2 * c4_
            mu_a = sa2 * (sumEheart - sumaeOmega_j * sum_E2) * c4_
            a[j] = rand(Normal(mu_a, sqrt(s2_a)))
        else
            a[j] = rand(Normal(0, sa))
        end
        ae[j] = r1[j] * r22[j] * a[j]
    end
    
end

mutable struct Hyperparameters
    n::Int64 # sample size
    p::Int64 # dimension of the mediators 
    
    ag::Float64 # MRF for G
    bg::Float64 
    
    nu1::Float64 # spike
    nu2::Float64 # slab
    nu12::Float64 # nu1 * nu1
    nu22::Float64 # nu2 * nu2
    lambda::Float64 
    
    a1pi::Float64 # pi1 ~ Beta(a1pi, b1pi)
    b1pi::Float64
    a2pi::Float64 # pi2 ~ Beta(a2pi, b2pi)
    b2pi::Float64
    
    ind::Array{Int32, 1}
    n_gstar::Int32
    p_g::Float64 
    
    sigma2e::Float64
    
    a_sigma2::Float64 # sigma2 ~ IG(a_sigma2, b_sigma2)
    b_sigma2::Float64
    
    sb::Float64 # b ~ Normal(0, sb^2)
    sa::Float64 # a ~ Normal(0, sa^2)
    sb2::Float64 
    sa2::Float64 
end

function construct_H(M)
    n = size(M)[1] # sample size
    p = size(M)[2] # dimension of the mediators   
    ag, bg = -2.75, 0.01 # MRF for G
    
    nu1, nu2 = 0.02, 1.0
    nu12, nu22 = nu1 * nu1, nu2 * nu2
    lambda = 1.0

    a1pi, b1pi = 1.0, 1.0 # pi2 ~ beta(a2pi, b2pi) 
    a2pi, b2pi = 1.0, 1.0 # pi2 ~ beta(a2pi, b2pi)
    
    ind = collect(1:p)
    n_gstar = Int(p * (p - 1) / 2)
    p_g = 2 / (p - 1)

    sigma2e = 10.0
    
    a_sigma2, b_sigma2, sb, sa = 1, 0.01, sqrt(10.0), sqrt(10.0)
    sb2, sa2 = sb * sb, sa * sa
    return Hyperparameters(n, p, ag, bg, nu1, nu2, nu12, nu22, lambda, a1pi, b1pi, a2pi, b2pi, ind, n_gstar, p_g, sigma2e, a_sigma2, b_sigma2, sb, sa, sb2, sa2)
end

function run_sampler(y, M, E, H; n_total = 5000, mrfprior = false, R = false)
    n, p = H.n, H.p
    ag, bg = H.ag, H.bg
    nu1, nu2, nu12, nu22 = H.nu1, H.nu2, H.nu12, H.nu22
    lambda = H.lambda
    a1pi, b1pi, a2pi, b2pi = H.a1pi, H.b1pi, H.a2pi, H.b2pi
    ind, p_g = H.ind, H.p_g
    sigma2e = H.sigma2e
    a_sigma2, b_sigma2, sb, sa, sb2, sa2 = H.a_sigma2, H.b_sigma2, H.sb, H.sa, H.sb2, H.sa2

    ##### Data
    E2 = E .^ 2
    M2 = M .^ 2
    sum_M2 = vec(sum(M2, dims = 1))
    sum_E2 = sum(E2)

    @rput R
    R"""
    g <- graph_from_adjacency_matrix(as.matrix(R))
    m = components(g) $ membership
    """
    @rget m
    
    if mrfprior; 
        ncombi = binomial(p, 2)
        list_id = zeros(Int16, ncombi, 2); 
        k = 0; for i = 1 : p; for j = (i + 1) : p; k += 1; list_id[k, 1] = i; list_id[k, 2] = j; end; end
        
        component_matrix = zeros(Int8, ncombi, ncombi)
        for k = 1 : ncombi
            for kk = (k + 1) : ncombi
                if length(unique([m[list_id[k, 1]], m[list_id[k, 2]], m[list_id[kk, 1]], m[list_id[kk, 2]]])) == 1
                    component_matrix[k, kk] = component_matrix[kk, k] = 1
                else
                    component_matrix[k, kk] = component_matrix[kk, k] = 0
                end
            end
        end

        Ltri = tril!(trues(size(R)), -1)
        Rstar = R[Ltri] * R[Ltri]'
        Rstar = Rstar .* component_matrix
    end
    
    ##### Initial values
    r1, r21, r22 = zeros(Int8, p), zeros(Int8, p), zeros(Int8, p)
    bm, ae = zeros(p), zeros(p)
    b, a = rand(Normal(0, sb), p), rand(Normal(0, sb), p)
    be = rand(Normal(0, sqrt(sigma2e)), 1)[1]

    pi1 = rand(Beta(a1pi, b1pi))
    pi2 = rand(Beta(a2pi, b2pi))
    sigma2 = rand(InverseGamma(a_sigma2, b_sigma2))
    sigma2_ = 1 / sigma2

    Omega = blockstr(p, 2, 0.2, 5) 
    G = zeros(Int8, p, p); for i in 1 : p; for j in 1 : p; if Omega[i, j] != 0; G[i, j] = 1; end; end; end
    V = zeros(p, p); for i in 1 : p; for j = 1 : p; if Omega[i, j] != 0; V[i, j] = nu22; else V[i, j] = nu12; end; end; end
    
    for i = 1 : p; for j = 1 : p; if Omega[i, j] == 0; Omega[i, j] = 0.001; end; end; end
    eadj = minimum([eigen(Omega).values .- 0.1; 0])
    for jj = 1 : p; Omega[jj, jj] = Omega[jj, jj] - Omega[jj, jj] * eadj; end
    cov = inv(Omega)
    
    Ebe = E * be

    r1_r, r21_r, r22_r = zeros(Int8, n_total, p), zeros(Int8, n_total, p), zeros(Int8, n_total, p)
    bm_r, ae_r = zeros(n_total, p), zeros(n_total, p)
    Omega_r, G_r = zeros(n_total, p, p), zeros(Int8, n_total, p, p)
    be_r = zeros(n_total)
    pi_r, sigma2_r = zeros(n_total, 2), zeros(n_total)

    println("n = $n, n_total = $n_total")
    print("Running... ")    
    
    elapsed_time = (
        @elapsed for iter = 1 : n_total
            update_bmae!(bm, ae, b, a, r1, r21, r22, Ebe, Omega, sigma2, sigma2_, pi1, pi2, y, M, M2, sum_M2, E, sum_E2, p, sb, sa, sb2, sa2)
         
            Eae = E * ae'
            Mbm = M * bm

            ##### update precision
            S = (M - Eae)' * (M - Eae) 
            update_precision!(ind, Omega, cov, V, p, lambda, S, n)
                             
            if mrfprior; 
                update_G!(G, Rstar, Omega, nu1, nu2, nu12, nu22, V, p, ag, bg, Ltri)
            else 
                update_G!(G, Omega, p_g, nu1, nu2, nu12, nu22, V, p)
            end
                
            ##### update be
            denom = 1 / (sigma2e * sum_E2 + sigma2)
            mube = sigma2e * dot(E, (y - Mbm)) * denom
            sigma2be = sigma2 * sigma2e * denom
            be = rand(Normal(mube, sqrt(sigma2be)))
            
            Ebe = E * be
            
            ##### update sigma2
            ei = y - Mbm - Ebe
            sigma2 = rand(InverseGamma(0.5 * n + a_sigma2, 0.5 * dot(ei, ei) + b_sigma2))
            sigma2_ = 1 / sigma2
            
            ##### update pi1 and pi2
            sum_r1, sum_r21 = sum(r1), sum(r21) + sum(r22)
            pi1, pi2 = rand(Beta(a1pi + sum_r1, b1pi + p - sum_r1)), rand(Beta(a2pi + sum_r21, b2pi + 2 * p - sum_r21))

            r1_r[iter, :], r21_r[iter, :], r22_r[iter, :] = r1, r21, r22
            bm_r[iter, :], ae_r[iter, :] = bm, ae
            Omega_r[iter, :, :], G_r[iter, :, :] = Omega, G

            pi_r[iter, 1], pi_r[iter, 2] = pi1, pi2
            be_r[iter] = be
            sigma2_r[iter] = sigma2
        end
        )
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    
    return Result(bm_r, ae_r, r1_r, r21_r, r22_r, be_r, Omega_r, G_r, sigma2_r, pi_r, elapsed_time)
end

struct Result
    bm::Array{Float64, 2}
    ae::Array{Float64, 2}
    r1::Array{Int8, 2}
    r21::Array{Int8, 2}
    r22::Array{Int8, 2}
    be::Array{Float64, 1}
    Omega::Array{Float64, 3}
    G::Array{Int8, 3}
    sigma2::Array{Float64, 1}
    pi::Array{Float64, 2}
    elapsed_time::Float64
end    
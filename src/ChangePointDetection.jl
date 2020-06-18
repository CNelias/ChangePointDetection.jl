module ChangePointDetection
using Random
using LinearAlgebra

"""
    squared_distance(X::Array{Float64,1},C::Array{Float64,1})
    computes the square distance between elements of
    X and C. returns squared_dist.
    squared_dist[ij] = ||X[:, i] - C[:, j]||^2
"""
function squared_distance(X::Array{Float64,1},C::Array{Float64,1})
    sqd = zeros(length(X),length(C))
    for i in 1:length(X)
        for j in 1:length(C)
            sqd[i,j] = X[i]^2 + C[j]^2 - 2*X[i]*C[j]
        end
    end
    return sqd
end

"""
    lsdd(x::Array{Float64,1}, y::Array{Float64,1}; folds = 5, sigma_list = nothing, lambda_list = nothing)
    Computes the least-squares density-difference (lsdd) for arrays `x` and `y`.
    The lsdd value characterizes how different the probability densities the generated 'x' and 'y' are.
    The closer the lsdd is to 0, the more similar the probability densities are.
    Input :
        'x', 'y' : the arrays of data upon which to perform the lsdd computation.
        folds : the number of cross-validation tests. higher is more precise but more expensive.
        sigma_list, lambda_list : points defining the grid search during the optimization of gaussian kernels.
    Returns :
        L2 : lsdd value.
"""
function lsdd(x::Array{Float64,1}, y::Array{Float64,1}; folds = 5, sigma_list = nothing, lambda_list = nothing)
    lx, ly = length(x), length(y)
    b = min(lx+ly,300)
    C = shuffle(vcat(x,y))[1:b]
    CC_dist2 = squared_distance(C,C)
    xC_dist2, yC_dist2 = squared_distance(x,C), squared_distance(y,C)
    Tx, Ty = length(x) - div(lx,folds), length(y) - div(ly,folds)
    #Define the training and testing data sets
    #cv stands for cross-validation
    cv_split1, cv_split2 = floor.(collect(1:lx)*folds/lx), floor.(collect(1:ly)*folds/ly)
    cv_index1, cv_index2 = shuffle(cv_split1), shuffle(cv_split2)
    tr_idx1,tr_idx2 = [findall(x->x!=i,cv_index1) for i in 1:folds], [findall(x->x!=i,cv_index2) for i in 1:folds]
    te_idx1,te_idx2 = [findall(x->x==i,cv_index1) for i in 1:folds], [findall(x->x==i,cv_index2) for i in 1:folds]
    xTr_dist, yTr_dist = [xC_dist2[i,:] for i in tr_idx1], [yC_dist2[i,:] for i in tr_idx2]
    xTe_dist, yTe_dist = [xC_dist2[i,:] for i in te_idx1], [yC_dist2[i,:] for i in te_idx2]
    #grid search is less expensive than optimization
    if sigma_list == nothing
        sigma_list = [0.25, 0.5, 0.75, 1, 1.2, 1.5, 2, 2.5, 2.2, 3, 5]
    end
    if lambda_list == nothing
        lambda_list = [1.00000000e-03, 3.16227766e-03, 1.00000000e-02, 3.16227766e-02,
       1.00000000e-01, 3.16227766e-01, 1.00000000e+00, 3.16227766e+00,
       1.00000000e+01]
    end
    score_cv = zeros(length(sigma_list),length(lambda_list))
    H = zeros(b,b)
    hx_tr, hy_tr = [zeros(b,1) for i in 1:folds], [zeros(b,1) for i in 1:folds]
    hx_te, hy_te = [zeros(1,b) for i in 1:folds], [zeros(1,b) for i in 1:folds]
    theta = zeros(b)

    for (sigma_idx,sigma) in enumerate(sigma_list)
        set_H(H,CC_dist2,sigma,b)
        set_htr(hx_tr,xTr_dist,sigma,Tx), set_htr(hy_tr,yTr_dist,sigma,Ty)
        set_hte(hx_te,xTe_dist,sigma,lx-Tx), set_hte(hy_te,yTe_dist,sigma,ly-Ty)

        for i in 1:folds
            h_tr = hx_tr[i] - hy_tr[i]
            h_te = hx_te[i] - hy_te[i]
            for (lambda_idx,lambda) in enumerate(lambda_list)
                set_theta(theta, H, lambda, h_tr, b)
                score_cv[sigma_idx, lambda_idx] += dot(theta, H*theta) - 2*dot(theta, h_te)
            end
        end
    end
    #retrieve the value of the optimal parameters
    sigma_chosen = sigma_list[findmin(score_cv)[2][2]]
    lambda_chosen = lambda_list[findmin(score_cv)[2][2]]
    #calculating the new optimal solution
    H = sqrt((sigma_chosen^2)*pi)*exp.(-CC_dist2/(4*sigma_chosen^2))
    H_lambda = H + lambda_chosen*Matrix{Float64}(I, b, b)
    h = (1/lx)*sum(exp.(-xC_dist2/(2*sigma_chosen^2)),dims = 1) - (1/ly)*sum(exp.(-yC_dist2/(2*sigma_chosen^2)),dims = 1)
    theta_final =  H_lambda\transpose(h)
    f = transpose(theta_final).*sum(exp.(-vcat(xC_dist2,yC_dist2)/(2*sigma_chosen^2)),dims = 1)
    L2 = 2*dot(theta_final,h) - dot(theta_final,H*theta_final)
    return L2
end

function set_H(H::Array{Float64,2},dist::Array{Float64,2},sigma::Float64,b::Int64)
    for i in 1:b
        for j in 1:b
            H[i,j] = sqrt((sigma^2)*pi)*exp(-dist[i,j]/(4*sigma^2))
        end
    end
end

function set_theta(theta::Array{Float64,1},H::Array{Float64,2},lambda::Float64,h::Array{Float64,2},b::Int64)
    Hl = (H + lambda*Matrix{Float64}(I, b, b))
    LAPACK.posv!('L', Hl, h)
    theta = h
end


function set_htr(h::Array{Array{Float64,2},1},dists::Array{Array{Float64,2},1},sigma::Float64,T::Int64)
        for (CVidx,dist) in enumerate(dists)
            for (idx,value) in enumerate((1/T)*sum(exp.(-dist/(2*sigma^2)),dims = 1))
                h[CVidx][idx] = value
            end
        end
end

function set_hte(h::Array{Array{Float64,2},1},dists::Array{Array{Float64,2},1},sigma::Float64,T::Int64)
    for (CVidx,dist) in enumerate(dists)
        for (idx,value) in enumerate((1/T)*sum(exp.(-dist/(2*sigma^2)),dims = 1))
            h[CVidx][idx] = value
        end
    end
end

"""
    lsdd_profile(ts; window = 150)
    returns for each point of given time-series 'ts' the lsdd value.
    Allows to estimate changes in the underlying probability density : a peak in lsdd value indicate a change point.
"""
function lsdd_profile(ts; window = 150)
    diff = []
    for i in 1:(length(ts)-2*window)
        pd1 = [ts[i+k] for k in 1:window]
        pd2 = [ts[i+window+k] for k in 1:window]
        push!(diff, lsdd(pd1,pd2)[1])
    end
    return diff
end

"""
    changepoints(ts; threshold = 0.5, window = 150)
Estimates change points in the underlying probability density of a time series via lsdd.
Every time the lsdd value exeeds the given threshold, a change point is detected.
returns the list of detected change points.
"""
function changepoints(ts; threshold = 0.5, window = 150)
    profile = lsdd_profile(ts; window = window)
    points = []
    exceeded = false
    for (index, value) in enumerate(profile)
        if (value > threshold) && exceeded == false
            push!(points, index)
            exceeded = true
        elseif (value < threshold) && exceeded == true
            exceeded = false
        end
    end
    return points
end

function getpoints(profile; threshold = 0.9)
    points = Int[]
    exceeded = false
    for (index, value) in enumerate(profile)
        if (value > threshold) && exceeded == false
            push!(points, index)
            exceeded = true
        elseif (value < threshold) && exceeded == true
            exceeded = false
        end
    end
    return points
end

using Plots
ts1 = rand(500)
ts2 = 2*rand(500) .+ 1
ts = vcat(ts1, ts2)
profile = lsdd_profile(ts; window = 30)
points = getpoints(profile)
a = plot(ts)
plot!(a, profile)
vline!(points)

display(a)

export squared_distance, lsdd
end

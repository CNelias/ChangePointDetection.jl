# ChangePointDetection.jl
[![Build Status](https://travis-ci.com/johncwok/ChangePointDetection.jl.svg?branch=master)](https://travis-ci.com/johncwok/ChangePointDetection.jl)

A fast Julia implementation of the least square density difference (LSDD) method. It is used to detect changepoints in time-series or to infer wether or not two time-series come from the same underlying probability distribution. The LSDD method was developped in the article *Density-difference estimation* from *M. Sugiyama, T. Suzuki, T. Kanamori, M. C. du Plessis, S. Liu, and I. Takeuchi.* in 2013.

Given two time-series <img src="https://render.githubusercontent.com/render/math?math=x_{1}(t)"> and <img src="https://render.githubusercontent.com/render/math?math=x_{2}(t)"> produced by underlying probability densities <img src="https://render.githubusercontent.com/render/math?math=p_{1}(x)"> and <img src="https://render.githubusercontent.com/render/math?math=p_{2}(x)">, the LSDD method directly modelizes the difference <img src="https://render.githubusercontent.com/render/math?math=g(x) = p_{1}(x) - p_{2}(x)"> to compute its L2 norm. This approach has good convergence properties and is more accurate than computing the KL-divergence of <img src="https://render.githubusercontent.com/render/math?math=p_{1}(x)"> and <img src="https://render.githubusercontent.com/render/math?math=p_{2}(x)"> after estimating them from scratch.

In practice, the closer the LSDD value is to 0, the more similar <img src="https://render.githubusercontent.com/render/math?math=p_{1}(x)"> and <img src="https://render.githubusercontent.com/render/math?math=p_{2}(x)"> are. To estimate changepoints in a time-series, two sliding windows are used. The LSDD value of these two sliding windows is computed along the whole time-series. Spikes in the obtained LSDD "profile" indicate potential changepoints. The procedure is best understood graphically :

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/34754896/85131911-1052dc80-b238-11ea-9e36-31d33a2fbd48.png">
</p>

One ends up with a time-dependant LSDD "profile", and threshold can be set to detect changepoints when exceeded.

## Usage :

To infer if two time-series come from the same underlying probability density, use the ```lsdd(ts1, ts2)``` function :
```
lsdd(x::Array{Float64,1}, y::Array{Float64,1}; folds = 5, sigma_list = nothing, lambda_list = nothing)
    Input :
        'x', 'y' : the time-series of data upon which to perform the lsdd computation.
        folds : the number of cross-validation tests. higher is more precise but more expensive.
        sigma_list, lambda_list : points defining the grid search during the optimization of gaussian kernels.
    Returns :
        L2 : lsdd value.
```

To compute the LSDD profile of a time-series, use the ```lsdd_profile(ts; window = 70)``` function :
```
lsdd_profile(ts; window = 70)
  Input :
      `ts` : the time-series to analyze
       window : the size of the sliding windows used for the lsdd computation. Bigger windows will be more accurate but can rapidly become very cumputationally expensive.
  Returns : lsdd profile of the input time-series. Note that since we are using 2 sliding windows, only time steps [1:end-2*window] are actually taken in account.  
```
Finally, to run an automatized changepoint detection, you can use the ```(ts; threshold = 0.5, window = 150)``` function.
```
Input
    ts : time-series to analyse
    threshold : when exceeded, a changepoint is detected
    window : size of the sliding windows;
Returns : a 1-D array of all detected change points.
```
## Example :
The following code 
```
using Plots

ts1 = rand(500)
ts2 = 1.5 * rand(500) .+ 1
ts = vcat(ts1, ts2)
profile = lsdd_profile(ts; window = 50)
points = getpoints(profile)
a = plot(ts,xlabel = "Time steps",ylabel = "Value",label = "",title = "Random time-series with change point",color = "black")
b = plot(profile,xlabel = "Time steps",ylabel = "LSDD value",label = "lsdd",title = "Corresponding LSDD profile",color = "black")
vline!(b,points,label = "detected changepoint \n (threshold exceeded)",color = "red",lw = 3,linestyle = :dash)
p = plot(a, b, layout = (2, 1), legend = true)
display(p)
```
Produced the following output :
<p align="center">
  <img width="800" height="500" src="https://user-images.githubusercontent.com/34754896/85133006-2bbee700-b23a-11ea-8478-203e90fdf1a4.PNG">
</p>


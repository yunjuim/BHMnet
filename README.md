## Publication
Yunju Im and Yuan Huang (2025). BHMnet: Bayesian High-Dimensional Mediation Analysis with Network Information Integration for Correlated Mediators. Manuscript.

## Description
1) BHMnet.jl contains the main function that produces MCMC samples for the BHMnet in the manuscript. 
2) simdata.jld2: the pre-saved data file containing simulated data called "mydat". 

## Examples
Below are examples of how to run the model with and without the network information (MRF prior).

### 1. Load Dependencies and Data
```julia
using StatsBase, LinearAlgebra, Distributions, Random, CSV, DataFrames, JLD2, RCall

# Load the main script
include("BHMnet.jl") 

# Load pre-saved data
@load "simdat.jld2" mydat R

# Format data
y = mydat[:, "y"]
E = mydat[:, "E"]
M = Matrix(select(mydat, r"x"))
p = size(M)[2]


H = construct_H(M) # Set hyperparameters
n_total, n_burn = 2000, 1000

# This runs the model without leveraging the external network R.
s1 = run_sampler(y, M, E, H, n_total = n_total, mrfprior = false);

r = s1.r1[(n_burn + 1) : end, :] .* (s1.r21[(n_burn + 1) : end, :] .* s1.r22[(n_burn + 1) : end, :])
findall(x -> x > 0.5, vec(mean(r, dims = 1)))

# This runs the full BHMnet model, integrating the external network R.
s2 = run_sampler(y, M, E, H, n_total = n_total, mrfprior = true, R = R)

r = s2.r1[(n_burn + 1) : end, :] .* (s1.r21[(n_burn + 1) : end, :] .* s1.r22[(n_burn + 1) : end, :])
findall(x -> x > 0.5, vec(mean(r, dims = 1)))
```

For more information, please contact Yunju Im at yim@unmc.edu. 

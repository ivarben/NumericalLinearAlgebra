include("arnoldi.jl")
using MAT, LinearAlgebra, Arpack, Random, Plots

B = matread("Bwedge.mat")["B"];
B_eigvals = matread("Bwedge.mat")["B_eigvals"];

## a) + b)
outside = B_eigvals[[1,2,3]]; # find outer eigenvalues
# Plots
#plt = plot(B_eigvals, legend = :topleft, label = "Eigenvalues", seriestype = :scatter, color = :black, marker = :x, markersize = 2)
plt = plot(xlims = [-50, 20], ylims = [-20, 20], B_eigvals, legend = :topleft, label = "Eigenvalues", seriestype = :scatter, color = :black, marker = :x, markersize = 2)
plot!(plt, outside, label = "Outer eigenvalues", seriestype = :scatter, color = :red, marker = :circle, markersize = 4)

## c)
n = size(B)[2];
Random.seed!(0)
b = randn(n); # Random starting vector
plt = scatter(xlims = [-48, 3], ylims = [-13, 13])

for m = [2 4 8 10 20 30 40]
    Q, H = arnoldi(B,b,m-1); # Generate Qm and Hm
    AM_vals, AM_vecs = eigen(Q'*B*Q); # Calculate Ritz values
    scatter!(plt, AM_vals, label = string("m = ", m), legend = :topleft, marker = :o, markersize = 3)
    display(plt)
    savefig(plt, string("plot_ex6c_m=", m, ".png"))
    error_AM = abs.(AM_vals[[1 m m-1]] - transpose(outside)) # Evaluate approximation error
    display(string(round(error_AM[1], sigdigits=4), " & ",round(error_AM[2], sigdigits=4), " & ", round(error_AM[3], sigdigits=4)))
end

## d) Shift and invert
target_eig = B_eigvals[findmin(abs.(B_eigvals .- (-9.8 + 1.5im)))[2]] # The eigenvalue in question

for σ = [-10, -7 + 2im, -9.8 + 1.5im]
    A = inv(B - σ * I); # Construct A
    display(string("σ = ", σ))
    for m = [10 20 30]
        display(string("m = ", m))
        Q, H = arnoldi(A,b,m-1); # Generate Qm and Hm using AM
        AM_vals, AM_vecs = eigen(Q'*A*Q); # Calculate Ritz values
        eig_approx = σ .+ 1 ./ AM_vals # Get eigenvalue approximations for B from Ritz values of A
        p = plot(eig_approx, legend = :topleft, label = "Eigenvalue approx", seriestype = :scatter, color = :black, marker = :o, markersize = 3, xlim = [-50,5], ylim = [-13,13])
        display(p)
        sleep(0.5)
        closest = eig_approx[findmin(abs.(eig_approx .- target_eig))[2]] # Find approximation closest to target
        display(round(abs(closest - target_eig), sigdigits = 4))
    end
end

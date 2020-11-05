include("arnoldi.jl")
using MAT, LinearAlgebra, Arpack, Plots

B = matread("Bwedge.mat")["B"];
B_eigvals = matread("Bwedge.mat")["B_eigvals"];

## a) + b) Plot eigenvalues and mark those that will converge fastest with AM.
# Indicate with circles and give estimations for convergence factors
outside = [-47.0161 + 0.1659im, 0.9856 - 11.8979im, 1.3137 + 12.6637im];
plot(B_eigvals, legend = :topleft, label = "Eigenvalues", seriestype = :scatter, color = :black, marker = :x, markersize = 2)
plot!(outside, label = "Outside Eigenvalues", seriestype = :scatter, color = :red, marker = :circle, markersize = 4)

## c)

n = size(B)[2];
b = randn(n);

for m = [2 4 8 10 20 30 40]
    Q, H = arnoldi(B,b,m);
    AM_vals, AM_vecs = eigen(Q'*B*Q);
    p = plot(AM_vals, legend = :topleft, label = "Ritz values", seriestype = :scatter, color = :black, marker = :o, markersize = 3)
    display(p)
    sleep(0.5)
    error_AM = AM_vals[[1 m m+1]] - transpose(outside)
    display(norm(error_AM))
end

## d) Shift and invert
for m = [10 20 30]
    σ = -9.8 + 1.5im;
    A = inv(B - σ * I);
    Q, H = arnoldi(A,b,m);
    AM_vals, AM_vecs = eigen(Q'*A*Q);
    display(σ .+ 1 ./ AM_vals)
end

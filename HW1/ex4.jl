using LinearAlgebra, Arpack, MatrixDepot, Random, Plots

nn=500;
m = 20;
Random.seed!(0)
A=matrixdepot("wathen",nn,nn)
n = size(A)[2];
b = randn(n)

## "True" values for comparison
vals, vecs = eigs(A)

## AM and Galerkin in same loop:
Q = zeros(n,m+1); # Initiate Q and H matrices
H = zeros(m+1,m);
Q[:,1] = b/norm(b);

K = zeros(n,m); # Initiate Km
K[:,1] = b;

plt = scatter(xlabel = "m", ylabel = "Real part of eigenval. approx.", xlims = [0, m], ylims = [0, 500]) # Set axes for plot

for k=1:m
    # Build Km for Galerkin method:
    if k >= 2
        Akb = A*K[:,k-1];
        K[:,k] = Akb/norm(Akb)
    end
    # Steps of Arnoldi Method:
    w=A*Q[:,k]; # Matrix-vector product with last element
    h,β,z = my_hw1_gs(Q[:,1:k],w); # Orthogonalize w against columns of Q
    H[1:(k+1),k] = [h;β]; #Put Gram-Schmidt coefficients into H
    Q[:,k+1] = z/β; # normalize

    # Obtain eigenvalue approximations from the respective methods
    galerkin_vals, galerkin_vecs = eigen(K[:,1:k]'*A*K[:,1:k], K[:,1:k]'*K[:,1:k])
    AM_vals, AM_vecs = eigen(Q[:,1:k]'*A*Q[:,1:k])

    # Plot the results
    if k == 1
        scatter!(plt, k*ones(length(AM_vals)), AM_vals, color = :black, marker = :circle, label = "Eigenvalue approx from Arnoldi method", markersize = 2, legend = :topleft)
        scatter!(plt, k*ones(length(galerkin_vals)), galerkin_vals, color = :red, marker = :x, label = "Eigenvalue approx from (2)", markersize = 2)
        display(plt)
    else
        scatter!(plt, k*ones(length(AM_vals)), AM_vals, color = :black, marker = :circle, label = nothing, markersize = 2)
        scatter!(plt, k*ones(length(galerkin_vals)), real.(galerkin_vals), color = :red, marker = :x, label = nothing, markersize = 2)
        display(plt)
    end
end

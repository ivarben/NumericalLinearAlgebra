include("arnoldi.jl")
using LinearAlgebra, Arpack, MatrixDepot, Random

nn=10;
m = 15;
Random.seed!(0)
A=matrixdepot("wathen",nn,nn)
#A = Diagonal([1, 2, 30, 3, 8, 9, 4]);
n = size(A)[2];
b = randn(n)

## "True" values
vals, vecs = eigs(A)

## AM and Galerkin in same loop:
Q = zeros(n,m+1);
H = zeros(m+1,m);
Q[:,1] = b/norm(b);

K = zeros(n,m);
K[:,1] = b;

plot()

for k=1:m
    # Galerkin:
    if k >= 2
        Akb = A*K[:,k-1];
        K[:,k] = Akb/norm(Akb)
    end

    # Arnoldi:
    w=A*Q[:,k]; # Matrix-vector product with last element
    # Orthogonalize w against columns of Q.
    # Implement this function or replace call with code for orthogonalizatio
    h,β,z = my_hw1_gs(Q[:,1:k],w);
    #Put Gram-Schmidt coefficients into H
    H[1:(k+1),k] = [h;β];
    # normalize
    Q[:,k+1] = z/β;

    galerkin_vals, galerkin_vecs = eigen(K[:,1:k]'*A*K[:,1:k], K[:,1:k]'*K[:,1:k])
    AM_vals, AM_vecs = eigen(Q[:,1:k+1]'*A*Q[:,1:k+1])
    display(galerkin_vals)
    display(AM_vals)

    plot!(AM_vals, label = "Eigenvalue approx from AM", seriestype = :scatter, color = :black, marker = :circle, markersize = 4)
    plot!(galerkin_vals, label = "Eigenvalue approx from (2)", seriestype = :scatter, color = :red, marker = :x, markersize = 4)
end

## Arnoldi Method
Q, H = arnoldi(A,b,m)
AM_vals, AM_vecs = eigen(Q'*A*Q)
display(norm(Q'*Q - I))

include("arnoldi.jl")
using LinearAlgebra, Arpack, MatrixDepot, Random

nn=10;
m = 6;
Random.seed!(0)
A=matrixdepot("wathen",nn,nn)
#A = Diagonal([1, 2, 30, 3, 8, 9, 4]);
n = size(A)[2];
b = randn(n)

## "True" values
vals, vecs = eigs(A)

## Galerkin Method
K = zeros(n,m);
K[:,1] = b;
for k = 2:m
    K[:,k] = (A^(k-1) * b)/norm(A^(k-1) * b);
end

galerkin_vals, galerkin_vecs = eigen(K'*A*K, K'*K)

## Arnoldi Method
Q, H = arnoldi(A,b,m)
AM_vals, AM_vecs = eigen(Q'*A*Q)
display(norm(Q'*Q - I))

## Plots

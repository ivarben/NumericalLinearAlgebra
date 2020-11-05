using Random, LinearAlgebra
using Arpack, SparseArrays # for sparse matrices

A=rand(4,4)
ee=eigvals(A) # compute the eigenvalues
V=eigvecs(A) # compute the eigenvectors

err=norm(A*V-V*diagm(ee))
println("The residual of the eigendecomposition is ",err)

# or compute the whole eigendecomposition
F = eigen(A) # compute the eigen-values and vectors together
err=norm(A*F.vectors-F.vectors*diagm(F.values))
println("The residual of the eigendecomposition is ",err)

# create a sparse matrix
n=1000
dm=ones(n-1)
dp=ones(n-1)
d0=-2*ones(n)
A=spdiagm(-1=>dm,0=>d0,1=>dp)

λ, ϕ = eigs(A, nev = 2, maxiter=6000)

res1=norm(A*ϕ[:,1]-λ[1]*ϕ[:,1])
res2=norm(A*ϕ[:,2]-λ[2]*ϕ[:,2])
println("The residual of the first eig pair ",res1)
println("The residual of the second eig pair ",res2)

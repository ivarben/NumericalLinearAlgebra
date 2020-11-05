# if not intalled
# using Pkg; Pkg.add("LinearAlgebra")
using LinearAlgebra
n=5;
A=rand(n,n)
C=one(A)
# compute norm of I (should be one)
println("Norm of C=",norm(C))
# obs: norm() computes the Frobenious norm
println("Operator norm of C=",opnorm(C))
# getting help: type "? norm" and "? opnorm"

# solve a linear system
b=rand(n)
# I is always the identity
M=A+I # add the identity
x=M\b
println("residual lin syst=",norm(M*x-b))
# operations with matrices
AT=transpose(A) # create a (lazy) transpose, no memory allocated
AT2=copy(transpose(A)) # materialize the transpose of A. Meoru allocation

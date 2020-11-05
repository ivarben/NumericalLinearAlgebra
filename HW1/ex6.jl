include("arnoldi.jl")
using MAT, LinearAlgebra, Arpack, Random

B=matread("Bwedge.mat")["B"];

## a) + b) Plot eigenvalues and mark those that will converge fastest with AM.
# Indicate with circles and give estimations for convergence factors

##

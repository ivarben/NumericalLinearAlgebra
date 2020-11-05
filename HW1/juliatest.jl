include("my_hw1_gs.jl")
include("arnoldi.jl")

using Random
A=randn(100,100); b=randn(100);
m=10;
Q,H=arnoldi(A,b,m);
println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
println("2:nd Should be zero = ", norm(Q'*Q-I));

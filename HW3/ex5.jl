include("schur_parlett.jl")

using LinearAlgebra, Plots, Random

## a)
f=z->sin(z);
A = [1 4 4; 3 -1 3; -1 4 4];
F = schur_parlett(A, f)
display(norm(F-sin(A)))

## b)
Random.seed!(4)
A = rand(100,100);
A=A/norm(A);

plt = scatter([0], [0], label="Naive", marker=:x, color=:black, xlabel="N", ylabel="CPU-time")
scatter!(plt, [0], [0], label="Schur-Parlett", marker=:o, color=:red)

trials=10;
for N = 5:5:400
#for N=1:400
    #Naive:
    t_naive = 0;
    for k = 1:trials
    t = @timed begin
    B=A;
    for i=1:N-1
        B=B*A;
    end
    end
    t_naive = t_naive + t[2];
    end
    t_naive = t_naive/trials;
    #SP:
    t_SP = 0
    for k = 1:trials
    t = @timed begin
    g=z->z^N;
    F = schur_parlett(A, g)
    end
    t_SP = t_SP + t[2];
    end
    t_SP = t_SP/trials;

    scatter!(plt, [N], [t_naive], marker=:x, color=:black, label=nothing)
    scatter!(plt, [N], [t_SP], marker=:o, color=:red, label=nothing)
    display(plt)
end

## c)
# Naive: every multiplication is O(n^3)
# we do N iterations => O(n^3*N)
# p=3, q=1
# SP:
# diagonal: O(n)
# then: triple nested for loop with each
# loop of complexity O(n) => O(n^3)
# Total complexity not dependent on N
# p=3, q = 0

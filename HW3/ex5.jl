include("schur_parlett.jl")

using LinearAlgebra, Plots

## a)
f=z->sin(z);
A = [1 4 4; 3 -1 3; -1 4 4];
F = schur_parlett(A, f)
display(norm(F-sin(A)))

## b)
A = rand(100,100);
A=A/norm(A);

plt = scatter([0], [0], label="Naive", marker=:x, color=:black, xlabel="N", ylabel="CPU-time")
scatter!(plt, [0], [0], label="Schur-Parlett", marker=:o, color=:red)
for N=1:400
    #Naive:
    t_naive = @timed begin
    B=A;
    for i=1:N-1
        B=B*A;
    end
    end
    #SP:
    t_SP = @timed begin
    f=z->z^N;
    F = schur_parlett(A, f)
    end
    display(norm(B-F))

    scatter!(plt, [N], [t_naive[2]], marker=:x, color=:black, label=nothing)
    scatter!(plt, [N], [t_SP[2]], marker=:o, color=:red, label=nothing)
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

using LinearAlgebra, Plots

plt = scatter(xscale=:log, yscale=:log, xlabel="ε", ylabel="||f(A)-F||")
for power = -10:-1

    ε = 10.0^power;
    A = [π 1; 0 π+ε];

    ## find exp(A) using JCF definition according to example in 4.1.2
    E = eigen(A)
    FJ = E.vectors*Diagonal(exp.(E.values))*inv(E.vectors)

    ## find exp(A) using the formula in b)
    β = (exp(π + ε) - exp(π))/ε;
    α = exp(π) - β*π;
    F = α*I + β*A;

    error = norm(F-FJ);
    scatter!(plt, [ε], [error], label=nothing, marker=:x, color = :red)
end
display(plt)

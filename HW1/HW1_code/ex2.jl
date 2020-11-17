using LinearAlgebra
using Plots

function eig_power(A, x0, n_iterations)
    
    x = x0 / norm(x0)
    lambda = zeros(n_iterations)
    for i = 1:n_iterations
        x = A * x
        x = x / norm(x)
        lambda[i] = x' * A * x
    end
    
    return lambda, x
    
end

function eig_rayleigh(A, x0, n_iterations)
    
    x = x0 / norm(x0)
    lambda = zeros(n_iterations)
    for i = 1:n_iterations
        lambda_current = i == 1 ? x0' * A * x0 : lambda[i - 1]
        x = (A - UniformScaling(lambda_current)) \ x
        x = x / norm(x)
        lambda[i] = x' * A * x
    end
    
    return lambda, x
    
end

# Matrix for 2a, 2b and if true for 2c
A = [1. 2. 3.; 2. 2. 2.; 3. 2. 9.]
if true
    A[1, 3] = 4
end

true_lambdas = eigvals(A)

lambdas_power, __ = eig_power(A, [1; 1; 1], 10)
lambdas_rayleigh, __ = eig_rayleigh(A, [1; 1; 1], 10)

true_lambda = true_lambdas[3]

# 2a
l = abs.(lambdas_power .- true_lambda)
conv = (true_lambdas[2] / true_lambdas[3]) .^ (2 * (1:length(lambdas_power)))
plot(l, yaxis=:log, label="Power method")
plot!(conv, yaxis=:log, label="Predicted")
xlabel!("Iteration")
ylabel!("Error")
savefig("2a.png")

# 2b/2c
plot(abs.(lambdas_rayleigh .- true_lambda .+ 1e-15), yaxis=:log, label="Rayleigh method")
xlabel!("Iteration")
ylabel!("Error")
savefig("2b.png")
#savefig("2c.png")
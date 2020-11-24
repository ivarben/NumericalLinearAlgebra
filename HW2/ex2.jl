include("GMRES.jl")

using LinearAlgebra, SparseArrays, Random, Plots, Arpack

# Create the matrix A for the respective values of alpha
n=100;
A_list = [];
b_list = [];
for alpha = [1 5 10 100]
    Random.seed!(1)
    A = sprand(n,n,0.5);
    A = A + alpha*sparse(1.0*I, n, n); A=A/norm(A,1);
    b = rand(n,1);
    push!(A_list, A);
    push!(b_list, b);
end


## b) Plot eigenvalues of A for alpha = 1,5,10,100. Provide bounds on convergence factors.
#Plot the convergence factors in the semilog plots from a)
scatter(eigvals(Matrix(A_list[1])), label = "alpha = 1", aspect_ratio = 1)
scatter!(eigvals(Matrix(A_list[2])), label = "alpha = 5")
scatter!(eigvals(Matrix(A_list[3])), label = "alpha = 10")
scatter!(eigvals(Matrix(A_list[4])), label = "alpha = 100")

# Estimated convergence factors:
convergence_factors = [0.00125/0.0005, 0.0011/0.001875, 0.0009/0.003, 0.001/0.009]

## a) Plot residual and error norm in semilog plot. Do it for alpha = 1,5,10,100.
alpha_index = 1;
A = A_list[alpha_index]; # index depending on alpha
b = b_list[alpha_index]; # index depending on alpha
x_true = A\b;
m = 20
res_norms = [];
err_norms = [];
# GMRES implementation based on AM
Q = zeros(n,m+1);
H = zeros(m+1,m);
Q[:,1] = b/norm(b);

for k=1:m
    w=A*Q[:,k]; # Matrix-vector product with last element
    # Orthogonalize w against columns of Q.
    # Implement this function or replace call with code for orthogonalizatio
    h,β,z = my_hw1_gs(Q[:,1:k],w);
    #Put Gram-Schmidt coefficients into H
    H[1:(k+1),k] = [h;β];
    # normalize
    Q[:,k+1] = z/β;

    z_star = H[1:(k+1),1:k]\([1; zeros(k)]*norm(b));
    x_tilde = Q[:,1:k]*z_star;

    res_norm = norm(A*x_tilde - b);
    err_norm = norm(x_tilde - x_true);
    append!(res_norms, res_norm)
    append!(err_norms, err_norm)
end
plot(1:m, res_norms, legend = :topright, label = "residual error norm", yaxis=:log, ylim = [1e-16, 1e6])
plot!(1:m, err_norms, label = "error norm")
plot!(1:m, convergence_factors[alpha_index].^(1:m), label = "expected convergence")

## c) Generate tables with computation times for alpha = 1, 100.
#For GMRES and "backslash", m = 5, 10, 20, 50, 100, n = 100, 200, 500.
alpha = 1; # 1, 100

# Create and collect A for the 3 values of n and 1 value of alpha
A_list = []
b_list = []
for n = [100 200 500]
    Random.seed!(1);
    A = sprand(n,n,0.5);
    A = A + alpha*sparse(1.0*I, n, n); A=A/norm(A,1);
    b = rand(n,1);
    push!(A_list, A)
    push!(b_list, b)
end

backslash_times = [];
backslash_res_norms = [];
for i = 1:3
    A = A_list[i];
    b = b_list[i];
    x_true = @timed A\b; # index 1 is solution, index 2 is time
    push!(backslash_res_norms, norm(A*x_true[1] - b));
    push!(backslash_times, x_true[2]);
end

backslash_tablerow = string(round(backslash_res_norms[1], sigdigits=4), " & ", round(backslash_times[1], sigdigits=4))
for i = 2:3
    backslash_tablerow = string(backslash_tablerow, " & ", round(backslash_res_norms[i], sigdigits=4), " & ", round(backslash_times[i], sigdigits=4))
end
display(backslash_tablerow)

for m = [5 10 20 50 100] # inefficient double for loop for printing purposes!!
    times = []
    res_norms = []
    for i = 1:3
        A = A_list[i];
        b = b_list[i];
        x_tilde = @timed GMRES(A,b,m);
        push!(res_norms, norm(A*x_tilde[1] - b));
        push!(times, x_tilde[2]);
    end
    tablerow = string("m = ", m)
    for i = 1:3
        tablerow = string(tablerow, " & ", round(res_norms[i], sigdigits=4), " & ", round(times[i], sigdigits=4))
    end
    display(tablerow)
end

## d) The tables show that under some circumstances we achieve acceptable error from GMRES with low m, which is faster than \.

include("GMRES.jl")

using LinearAlgebra, SparseArrays, Random

alpha=100; n=100; Random.seed!(1)
A = sprand(n,n,0.5);
A = A + alpha*sparse(1.0*I, n, n); A=A/norm(A,1);
b = rand(n,1);

x = GMRES(A, b, 100)
r = norm(A*x - b)

## a) Plot residual and error norm in semilog plot. Do it for alpha = 1,5,10,100.
x_true = A\b;
for m = 1:110
    x_tilde = GMRES(A, b, m);
    res_norm = norm(A*x_tilde - b);
    err_norm = norm(x_tilde - x_true);
    display(res_norm) # Change to plot
    display(err_norm) # Change to plot
end

## b) Plot eigenvalues of A for alpha = 1,5,10,100. Provide bounds on convergence factors.
#Plot the convergence factors in the semilog plots from a)

## c) Generate tables with computation times for alpha = 1, 100. For GMRES and "backslash", m = 5, 10, 20, 50, 100, n = 100, 200, 500.
alpha = 100; # 1, 100

A_list = []
b_list = []
for n = [100 200 500]
    Random.seed!(1); # seed?
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

include("my_hw1_gs.jl")

function GMRES(A,b,m)

    n = length(b);
    Q = complex(zeros(n,m+1));
    H = complex(zeros(m+1,m));
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

    end
    z_star = H\([1; zeros(m)]*norm(b));
    x_tilde = Q[:,1:m]*z_star;

    return x_tilde
end

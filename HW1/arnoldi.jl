include("my_hw1_gs.jl")
"""
    Q,H=arnoldi(A,b,m)

A simple implementation of the Arnoldi method.
The algorithm will return an Arnoldi "factorization":
Q*H[1:m+1,1:m]-A*Q[:,1:m]=0
where Q is an orthogonal basis of the Krylov subspace
and H a Hessenberg matrix.



The function `my_hw1_gs(Q,w,k)` needs to be available.


"""
function arnoldi(A,b,m)

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
    return Q,H
end

# gram schmidt implementation

function my_hw1_gs(Q,b)
    h = Q'*b;
    z = b-Q*h;
    g = Q'*z;
    z = z-Q*g;
    h = h+g;
    β = norm(z);

    return h, β, z
end

using LinearAlgebra
using Plots
using Base
using ImageView, Images, ImageIO, ImageMagick, Printf, FileIO, LinearAlgebra, Arpack

function load_india()
    
    basefilename="india_driving1_frames\\india_driving_frame"
    # Load the image into an image object, here the first frame
    img=load(abspath(basefilename*"0001.png")); 
    # Visualize it
    #imshow(img)
    # Determine the image size
    sz=size(img); szv=sz[1]*sz[2]; 
    # Reshape the image to a vector:
    R=float(red.(img));
    G=float(green.(img));
    B=float(blue.(img));
    v=vcat(vec(R), vec(G), vec(B))
    
    m=4  # Number of frames to load
    A=zeros(size(v,1),m) # Matrix to store all the frames in
    for k=1:m
        fname=@sprintf("%s%04d.png",basefilename,k);
        img=load(abspath(fname));
        #println(fname)
        R=float(red.(img));
        G=float(green.(img));
        B=float(blue.(img));
        v=vcat(vec(R), vec(G), vec(B));
        A[:,k]=v;
    end
    
    return A
    
end

function load_market()
    
    basefilename="market_snapshots\\market_snapshots"
    # Load the image into an image object, here the first frame
    img=load(abspath(basefilename*"_0001.jpg")); 
    # Visualize it
    #imshow(img)
    # Determine the image size
    sz=size(img); szv=sz[1]*sz[2]; 
    # Reshape the image to a vector:
    R=float(red.(img));
    G=float(green.(img));
    B=float(blue.(img));
    v=vcat(vec(R), vec(G), vec(B))
    
    m=295  # Number of frames to load
    A=zeros(size(v,1),m) # Matrix to store all the frames in
    for k=1:m
        fname=@sprintf("%s_%04d.jpg",basefilename,k);
        img=load(abspath(fname));
        #println(fname)
        R=float(red.(img));
        G=float(green.(img));
        B=float(blue.(img));
        v=vcat(vec(R), vec(G), vec(B));
        A[:,k]=v;
    end
    
    return A
    
end

function visualize_india(v)
    
    basefilename="india_driving1_frames\\india_driving_frame"
    # Load the image into an image object, here the first frame
    img=load(abspath(basefilename*"0001.png")); 
    # Visualize it
    #imshow(img)
    # Determine the image size
    sz=size(img); szv=sz[1]*sz[2];  
    vv=reshape(v,szv,3);
    R=reshape(vv[:,1],sz[1],sz[2]);
    G=reshape(vv[:,2],sz[1],sz[2]);
    B=reshape(vv[:,3],sz[1],sz[2]);
    newimg=RGB.(R,G,B);
    imshow(newimg);

end

function visualize_market(v)
    
    basefilename="market_snapshots\\market_snapshots"
    # Load the image into an image object, here the first frame
    img=load(abspath(basefilename*"_0001.jpg")); 
    # Visualize it
    #imshow(img)
    # Determine the image size
    sz=size(img); szv=sz[1]*sz[2]; 
    vv=reshape(v,szv,3);
    R=reshape(vv[:,1],sz[1],sz[2]);
    G=reshape(vv[:,2],sz[1],sz[2]);
    B=reshape(vv[:,3],sz[1],sz[2]);
    newimg=RGB.(R,G,B);
    imshow(newimg);

end

function lanczos(A, b, M)
    
    Q = zeros(size(A, 1), M)
    Q[:, 1] = b ./ norm(b)
    
    alpha = zeros(M)
    beta = zeros(M)
    for m = 1:M
        v = A * Q[:, m]
        alpha[m] = Q[:, m]' * v
        v = v - alpha[m] * Q[:, m]
        if m > 1
            v = v - beta[m - 1] * Q[:, m - 1]
        end
        beta[m] = norm(v)
        if m < M
            Q[:, m + 1] = v ./ beta[m]
        end
    end
    H = SymTridiagonal(alpha, beta)
    
    return H, Q
    
end

function svd_lanczos_rk1(A)
    
    @assert size(A, 1) >= size(A, 2)
    
    H, Q = lanczos(A' * A, ones(size(A, 2)), size(A, 2))
    lambda, xi = eigs(H, nev=1)
    v1 = Q * xi
    v1 = v1 / norm(v1)
    sigma1 = sqrt(lambda[1])
    u1 = A * v1 / sigma1
    
    return u1, sigma1, v1
    
end

# Indua
A = load_india()
U, s, V = svd(A)
A_tilde = s[1] * U[:, 1] * V[:, 1]'
visualize_india(A_tilde[:, 1])

# Market
A = load_market()
u1, sigma1, v1 = svd_lanczos_rk1(A)
A_tilde = sigma1 * u1 * v1'
visualize_market(A_tilde[:, 1])
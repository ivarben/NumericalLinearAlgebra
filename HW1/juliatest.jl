#tests
include("my_hw1_gs.jl")

b = [1 2 3]'
A = [1 2 3; 0 1 0; 0 0 1]
B = [1 0 3; 0 1 0; 4 0 1]

x = A\b
display(x)

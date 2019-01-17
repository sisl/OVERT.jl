using NeuralVerification

solver = ReluVal() # got errors with BaB, Sherlock

my_nnet = read_nnet("/Users/Chelsea/Dropbox/AAHAA/src/nnet_files/file_test_1")

inputSet  = Hyperrectangle(low = [-0.001, -0.001], high = [0.001, 0.001])
outputSet = Hyperrectangle(low = [-1.57, -10.], high = [1.57, 10.])
problem = Problem(my_nnet, inputSet, outputSet)

@time result = solve(solver, problem)

println(result.status)

# Notes:
# What is the exact meaning of '1D interval?'
# vector like this: high =[1,2,3], low=[0,0,0] ? or this high=[1] low=[0] ?
# BaB error: ERROR: LoadError: Invalid coefficient NaN on variable __anon__[1]
# Sherlock error: ERROR: LoadError: Incompatible sizes
# This page needs updating: https://github.com/sisl/NeuralVerification.jl/blob/master/docs/src/solvers.md
# -> I think these two can only work with one output node

# Hyperrectangle vs hpolytope?
# hpolytope: Ax<=b: takes in Hpolytope(A,b)
#(I ) = (UB )
#(-I)   (-LB)

# note that 

# halfspace constraints: A matric cna only have 1 row
# you can have relations bnetween the variables using HPolytopes like x1>=x2

# one constraint == halfspace, according to 

# planet and AI2 are broken?

# reluplex, planet, and ai2 have issues
# but thhey never say that unsafe stuff is safe

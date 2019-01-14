using NeuralVerification

solver = Sherlock() # BaB() # got errors with BaB

small_nnet = read_nnet("/Users/Chelsea/.julia/packages/NeuralVerification/YzObS/examples/networks/small_nnet.txt")

my_nnet = read_nnet("/Users/Chelsea/Dropbox/AAHAA/src/nnet_files/file_test_1")


inputSet  = Hyperrectangle(low = [-0.001, -0.001], high = [0.001, 0.001])
outputSet = Hyperrectangle(low = [-1.57, -10.], high = [1.57, 10.])
problem = Problem(my_nnet, inputSet, outputSet)

result = solve(solver, problem)

result.status
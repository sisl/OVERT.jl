using NeuralVerification
using LinearAlgebra
using LazySets
include("utils.jl")

# Hyperrectangle vs hpolytope?
# hpolytope: Ax<=b: takes in Hpolytope(A,b)
#(I ) = (UB )
#(-I)   (-LB)

# read network and setup inputs and outputs
my_nnet = 
try
	read_nnet("/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files/correct_overrapprox_const_dyn_2867.nnet")
catch;
	read_nnet("nnet_files/correct_overrapprox_const_dyn_2867.nnet")
end
# inputs: (5)
# import/assign_init_vals/theta_dot_hat_1,
# import/assign_init_vals/theta_dot_hat_2,
# import/initial_values/theta_dot_0,
# import/assign_init_vals/theta_dot_hat_3,
# import/initial_values/theta_0,
input_low = [-100., -100., -50., -100., -1*π/180]
input_high = [100.,  100.,  50.,  100.,  1*π/180]
A = vcat(Matrix{Float64}(I, 5, 5), -Matrix{Float64}(I,5,5))
b = vcat(input_high, -input_low)
inputSet  = HPolytope(A, b)
# outputs (12)
# thetas, tdhs, tdlbs, tdubs (3x each)
A_th_ub = hcat(eye(3),zeros(3,9))
b_th_ub = 45*π/180*ones(3)
A_th_lb = hcat(-1*eye(3),zeros(3,9))
b_th_lb = -45*π/180*ones(3)
A_thd_ub = hcat(zeros(3,3), 
				Matrix{Float64}(I,3,3), 
				zeros(3,3), 
				-Matrix{Float64}(I,3,3))
b_thd_ub = zeros(3)
A_thd_lb = hcat(zeros(3,3),
				-Matrix{Float64}(I,3,3),
				Matrix{Float64}(I,3,3),
				zeros(3,3)
			)
b_thd_lb = zeros(3)

A_ub_arbitrary = hcat(zeros(9,3),
						Matrix{Float64}(I,9,9)
					)
b_ub_arb = ones(9)*100
A_lb_arbitrary = hcat(zeros(9,3),
						-Matrix{Float64}(I,9,9)
					)
b_lb_arb = ones(9)*(-100)

A = vcat(A_th_ub, A_th_lb, A_thd_ub, A_thd_lb, A_ub_arbitrary, A_lb_arbitrary)
b = vcat(b_th_ub, b_th_lb, b_thd_ub, b_thd_lb, b_ub_arb, b_lb_arb)

outputSet = HPolytope(A,b)

function test(inputSet, outputSet, nnet)
	solver = Ai2() # got errors with BaB, Sherlock

	problem = Problem(nnet, inputSet, outputSet)

	@time result = solve(solver, problem)

	println(result.status)
end

test(inputSet, outputSet, my_nnet)




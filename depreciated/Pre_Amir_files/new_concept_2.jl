using NeuralVerification
using LinearAlgebra

# Hyperrectangle vs hpolytope?
# hpolytope: Ax<=b: takes in Hpolytope(A,b)
#(I ) = (UB )
#(-I)   (-LB)

# read network and setup inputs and outputs
my_nnet = read_nnet("/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files/correct_overrapprox_const_dyn_1_step1641.nnet")
println("1 step of dynamics")
# inputs
# theta, theta_dot_hat_1, theta_dot

input_low = [-45*π/180, -100., -50.]
input_high = [45*π/180, 100., 50.]
A = vcat(Matrix{Float64}(I, 3, 3), -Matrix{Float64}(I,3,3))
b = vcat(input_high, -input_low)
inputSet  = HPolytope(A, b)
# outputs
# theta_1, tdh1, tdlb1, tdub1
A_th_ub = hcat([1.],zeros(3)')
b_th_ub = [45*π/180]

A_th_lb = hcat([-1.],zeros(3)')
b_th_lb = [-45*π/180]

A_thd_ub = hcat(zeros(1), 
				Matrix{Float64}(I,1,1), 
				zeros(1,1), 
				-Matrix{Float64}(I,1,1))
b_thd_ub = zeros(1)
A_thd_lb = hcat(zeros(1),
				-Matrix{Float64}(I,1,1),
				-Matrix{Float64}(I,1,1),
				zeros(1,1)
			)
b_thd_lb = zeros(1)
A_ub_arbitrary = hcat(zeros(3),
						Matrix{Float64}(I,3,3)
					)
b_ub_arb = ones(3)*100
A_lb_arbitrary = hcat(zeros(3),
						-Matrix{Float64}(I,3,3)
					)
b_lb_arb = ones(3)*(-100)

A = vcat(A_th_ub, A_th_lb, A_thd_ub, A_thd_lb, A_ub_arbitrary, A_lb_arbitrary)
b = vcat(b_th_ub, b_th_lb, b_thd_ub, b_thd_lb, 
	b_ub_arb, b_lb_arb)

outputSet = HPolytope(A,b)

function test(inputSet, outputSet, nnet)
	solver = ExactReach() # got errors with BaB, Sherlock

	problem = Problem(nnet, inputSet, outputSet)

	@time result = solve(solver, problem)

	println(result.status)
end

test(inputSet, outputSet, my_nnet)




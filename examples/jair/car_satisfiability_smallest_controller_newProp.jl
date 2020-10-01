include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/car/simple_car.jl")
using JLD2

function run_query(query_number, avoid_set, controller_name)
	controller = "nnet_files/jair/car_"*controller_name*"_controller.nnet"

	query = OvertQuery(
		SimpleCar,  # problem
		controller,    # network file
		Id(),      	# last layer activation layer Id()=linear, or ReLU()=relu
		"MIP",     	# query solver, "MIP" or "ReluPlex"
		25,        	# ntime
		0.2,       	# dt
		-1,        	# N_overt
		)

	# In this example, our property is the following:
	# We want the car to reach the box [-.6, .6] [-.2,.2]
	# at SOME point in the time history
	# we will perform 4 separate queries and "AND" them together to look
	# for a point where all properties hold
	input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])
	target_set = InfiniteHyperrectangle([-Inf, -Inf, -Inf, -Inf], [5.0, Inf, Inf, Inf])
	t1 = Dates.time()
	SATii, valii, statii = symbolic_satisfiability(query, input_set, target_set; return_all=true)
	t2 = Dates.time()
	dt = (t2-t1)

	JLD2.@save "examples/jair/data/new/car_satisfiability_"*string(controller_name)*"_controller_data_q"*string(query_number)*".jld2" query input_set target_set SATii valii statii dt

	return SATii
end

function run_car_satisfiability(; controller_name="smallest")
	# query 1
	avoid_set1 = MyHyperrect(low=[-Inf, -Inf, -Inf, -Inf], high=[-0.6, Inf, Inf, Inf]) 
	SAT1 = run_query(1, avoid_set1, controller_name)

	# query 2
	avoid_set2 = MyHyperrect(low=[0.6, -Inf, -Inf, -Inf], high=[Inf, Inf, Inf, Inf]) 
	SAT2 = run_query(2, avoid_set2, controller_name)

	# query 3
	avoid_set3 = MyHyperrect(low=[-Inf, -Inf, -Inf, -Inf], high=[Inf, -0.2, Inf, Inf]) 
	SAT3 = run_query(3, avoid_set3, controller_name)

	# query 4
	avoid_set4 = MyHyperrect(low=[-Inf, 0.2, -Inf, -Inf], high=[Inf, Inf, Inf, Inf]) 
	SAT4 = run_query(4, avoid_set4, controller_name)

	# now we want to know when all properties hold
	result1 = SAT1 .== "unsat"
	result2 = SAT2 .== "unsat"
	result3 = SAT3 .== "unsat"
	result4 = SAT4 .== "unsat"

	all_hold = ((result1 .& result2) .& result3) .& result4
	timesteps_where_peroperties_hold = findall(all_hold)
	print("The property holds at timestep: ", timesteps_where_peroperties_hold)

	JLD2.@save "examples/jair/data/new/car_satisfiability_"*string(controller_name)*"_controller_data_final_result.jld2" timesteps_where_peroperties_hold
end

run_car_satisfiability(controller_name="smallest")
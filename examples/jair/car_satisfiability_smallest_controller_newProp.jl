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
		2,        	# ntime
		0.2,       	# dt
		-1,        	# N_overt
		)

	input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])
	t1 = Dates.time()
	SATii, valii, statii = symbolic_satisfiability(query, input_set, avoid_set; return_all=true)
	println("satii is: ", SATii)
	t2 = Dates.time()
	dt = (t2-t1)

	JLD2.@save "examples/jair/data/new/car_satisfiability_"*string(controller_name)*"_controller_data_q"*string(query_number)*".jld2" query input_set avoid_set SATii valii statii dt

	println("satii after save is: ", SATii)
	return SATii
end

function run_car_satisfiability(; controller_name="smallest")
	# In this example, our property is the following:
	# We want the car to reach the box [-.6, .6] [-.2,.2]
	# at SOME point in the time history
	# we will perform 4 separate queries and "AND" them together to look
	# for a point where all properties hold

	# query 1
	avoid_set1 = MyHyperrect(low=[-Inf, -Inf, -Inf, -Inf], high=[-0.6, Inf, Inf, Inf]) 
	# query 2
	avoid_set2 = MyHyperrect(low=[0.6, -Inf, -Inf, -Inf], high=[Inf, Inf, Inf, Inf]) 
	# query 3
	avoid_set3 = MyHyperrect(low=[-Inf, -Inf, -Inf, -Inf], high=[Inf, -0.2, Inf, Inf]) 
	# query 4
	avoid_set4 = MyHyperrect(low=[-Inf, 0.2, -Inf, -Inf], high=[Inf, Inf, Inf, Inf])
	avoid_sets = [avoid_set1, avoid_set2, avoid_set3, avoid_set4]

	SAT = []

	s = ["unsat", "unsat"]
	for enum = enumerate(avoid_sets)
		i, avoid_set = enum
		if ~all(s .== "sat") # possibly quit early if all of s = "sat"
			s = run_query(i, avoid_set, controller_name)
			push!(SAT, s)
		else
			println("skipping property ", i, " because prior property does not hold any time.")
		end
	end

	# now we want to know when all properties hold
	all_hold = [true for _ in 1:length(SAT[1])]
	for i = 1:length(SAT)
		all_hold = all_hold .& (SAT[i] .== "unsat")
	end
	timesteps_where_properties_hold = findall(all_hold)
	if len(timesteps_where_properties_hold) > 0
		println("The property holds at timestep: ", timesteps_where_properties_hold)
	else
		println("The property does not hold.")
	end

	JLD2.@save "examples/jair/data/new/car_satisfiability_"*string(controller_name)*"_controller_data_final_result.jld2" SAT timesteps_where_properties_hold
end

run_car_satisfiability(controller_name="smallest")
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")
using JLD2

function run_query(query_number, avoid_set, controller_name)
	controller = "nnet_files/jair/tora_"*controller_name*"_controller.nnet"

	println("Controller: ", controller_name)

	query = OvertQuery(
		Tora,      # problem
		controller, # network file
		Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
		"MIP",     # query solver, "MIP" or "ReluPlex"
		2,        # ntime
		0.1,       # dt
		-1,        # N_overt
		)

	input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
	# the other Infs are just to denote that these constraints should be ignored. So essentially the only constraint being added is:
	# x1 <= -2
	t1 = Dates.time()
	SATus, vals, stats = symbolic_satisfiability(query, input_set, avoid_set)
	t2 = Dates.time()
	dt = (t2-t1)

	JLD2.@save "examples/jmlr/data/tora_satisfiability_"*string(controller_name)*"_controller_data_q"*string(query_number)*".jld2" query input_set avoid_set SATus vals stats dt query_number controller_name

	return SATus
end

function run_tora_satisfiability(;controller_name="smallest")

	# query 1
	avoid_set1 = HalfSpace([1., 0., 0., 0.], -2.) # checks if x1 <= -2 
	SATus1 = run_query(1, avoid_set1, controller_name)

	if SATus1 == "unsat" # early stopping, potentially, if first query is sat
		# query 2
		avoid_set2 =  HalfSpace([-1., 0., 0., 0.], -2.)# checks if x1 >= 2
		SATus2 = run_query(2, avoid_set2, controller_name)

		open("examples/jmlr/data/tora_satisfiability_"*string(controller_name)*".txt", "w") do io
			write(io, "SATus1 = $SATus1 \n SATus2 = $SATus2")
		end;
	end
end

run_tora_satisfiability(controller_name=ARGS[1])
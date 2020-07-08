include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/car/simple_car.jl")

network = "nnet_files/jair/car_smaller_controller.nnet"

query = OvertQuery(
	SimpleCar,  # problem
	network,    # network file
	Id(),      	# last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     	# query solver, "MIP" or "ReluPlex"
	22,        	# ntime
	0.2,       	# dt
	-1,        	# N_overt
	)

input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])
t1 = Dates.time()
all_sets, all_sets_symbolic = symbolic_reachability_with_concretization(query, input_set, [6, 6, 5, 5])
t2 = Dates.time()
dt = (t2-t1)

using JLD2
JLD2.@save "examples/jair/data/car_reachability_smaller_controller_data.jld2" query input_set all_sets all_sets_symbolic dt

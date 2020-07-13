include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/drone/drone.jl")

network = "nnet_files/jair/drone_small_controller2.nnet"

query = OvertQuery(
	Drone,  # problem
	network,    # network file
	Id(),      	# last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     	# query solver, "MIP" or "ReluPlex"
	10,        	# ntime
	0.1,       	# dt
	-1,        	# N_overt
	)

# the controller is trained on data initial data in the following range
# -2 < x1 < 2, -1 < x2 < 1, 4 < x3 < 6,
# 0 < x4,x5,x6 < 1, 0 < x7-x12 < 0.2


input_set = Hyperrectangle(low=[-1., -0.5, 4.5, 0.45, 0.45, 0.45, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                          high=[ 1.,  0.5, 5.5, 0.55, 0.55, 0.55, 0.1 , 0.1 , 0.1 , 0.1 , 0.1,  0.1])
t1 = Dates.time()
all_sets, all_sets_symbolic = many_timestep_concretization(query, input_set)
t2 = Dates.time()
dt = (t2-t1)

# using JLD2
# JLD2.@save "examples/jair/data/car_reachability_big_controller_data.jld2" query input_set all_sets all_sets_symbolic dt

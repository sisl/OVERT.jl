include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

#controller = "nnet_files/sherlock/tora_bigger_controller_offset_10_scale_1.nnet"
controller = "nnet_files/sherlock/tora_smaller_controller.nnet"

query = OvertQuery(
	Tora,      # problem
	controller, # network file
	Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     # query solver, "MIP" or "ReluPlex"
	25,        # ntime
	0.1,       # dt
	-1,        # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
target_set = Hyperrectangle([-1., 0.1, -0.1, -0.6], [0.1, 0.1, 0.1, 0.1])

t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set)
t2 = Dates.time()
print("elapsed time= $((t2-t1)/1000) seconds")

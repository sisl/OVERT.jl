include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

controller = "nnet_files/jair/tora_smallest_controller.nnet"

query = OvertQuery(
	Tora,       # problem
	controller, # network file
	Id(),       # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",      # query solver, "MIP" or "ReluPlex"
	10,         # ntime
	0.1,        # dt
	-1,         # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
t1 = Dates.time()
all_sets, all_sets_symbolic = symbolic_reachability(query, input_set)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

using JLD2
JLD2.@save "examples/jair/data/tora_nnv_comparison_data.jld2" query input_set all_sets all_sets_symbolic dt

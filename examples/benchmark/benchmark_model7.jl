include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/sherlock-model-7/model7.jl")

query = OvertQuery(
	Model7,                                   # problem
	"nnet_files/controller_model7.nnet",      # network file
	ReLU(),                                   # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                    # query solver, "MIP" or "ReluPlex"
	5,                                        # ntime
	0.5,                                      # dt
	-1,                                       # N_overt
	)

input_set = Hyperrectangle(low=[0.35, 0.45, 0.25], high=[0.45, 0.55, 0.35])

output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
#all_sets, all_sets_symbolic = symbolic_bound(query, input_set)

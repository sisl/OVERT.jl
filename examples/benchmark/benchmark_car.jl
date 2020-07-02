include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/car/simple_car.jl")

query = OvertQuery(
	SimpleCar,                                   # problem
	"nnet_files/ARCH_COMP/controller_car.nnet",            # network file
	ReLU(),                                      # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                       # query solver, "MIP" or "ReluPlex"
	5,                                           # ntime
	0.2,                                         # dt
	-1,                                          # N_overt
	)

input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])

all_sets, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets2, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets3, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets4, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets5, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets6, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

all_sets7, all_sets_symbolic = symbolic_reachability(query, input_set)
input_set = all_sets_symbolic

#
# query.ntime= 35
# input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])
# output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
#
# all_sets= vcat(all_sets, all_sets_2, all_sets_3, all_sets_4, all_sets_5, all_sets_6, all_sets_7)
#
# fig = plot_output_sets(all_sets)
# fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
# fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

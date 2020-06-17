include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

query = OvertQuery(
	Tora,                                     # problem
	"nnet_files/bigger_controller_tora.nnet", # network file
	Id(),                                     # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                    # query solver, "MIP" or "ReluPlex"
	20,                                       # ntime
	1.0,                                      # dt
	-1,                                       # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4., 0.5], high=[0.7, -0.6, -0.3., 0.6])


all_sets, all_sets_symbolic = symbolic_bound(query, input_set)
output_sets, xvec = monte_carlo_simulate(query, input_set)

fig = plot_output_sets(all_sets)
fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

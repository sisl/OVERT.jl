include("models/problems.jl")
include("OverApprox/src/overapprox_nd_relational.jl")
include("OverApprox/src/overt_parser.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("models/models/single_pendulum/single_pend.jl")

query = OvertQuery(
	SinglePendulum,                                   # problem
	"nnet_files/controller_simple_single_pend.nnet",  # network file
	Id(),                                             # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                            # query solver, "MIP" or "ReluPlex"
	20,                                               # ntime
	0.1,                                              # dt
	-1,                                                # N_overt
	)

input_set = Hyperrectangle(low=[1., 1.], high=[2., 2.])


satus, values, stats = symbolic_satisfiability(query, input_set)
# output_sets, xvec = monte_carlo_simulate(query, input_set)

# fig = plot_output_sets(all_sets)
# fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
# fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

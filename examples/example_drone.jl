include("models/problems.jl")
include("OverApprox/src/overapprox_nd_relational.jl")
include("OverApprox/src/overt_parser.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("models/drone/drone.jl")

query = OvertQuery(
	Drone,                                         # problem
	"nnet_files/controller_medium_drone.nnet", # network file
	Id(),                                          # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                         # query solver, "MIP" or "ReluPlex"
	10,                                            # ntime
	0.1,                                           # dt
	-1,                                             # N_overt
	)

input_set = Hyperrectangle(low=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                           high=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])


all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set)
#all_sets, all_sets_symbolic = symbolic_bound(query, input_set)
output_sets, xvec = monte_carlo_simulate(query, input_set; n_sim=10000000)

fig = plot_output_sets(all_sets)
fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

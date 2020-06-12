include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../overt_to_mip.jl")
include("../problems.jl")
include("../utils.jl")
include("../models/ball_beam/ball_beam.jl")

query = OvertQuery(
	BallnBeam,                                          # problem
	"../nnet_files/controller_spread_ball_beam.nnet",   # network file
	Id(),                                               # last layer activation layer Id()=linear, or ReLU()=relu
	"blah",                                             # query type, "concrete" or "symoblic"
	5,                                                  # ntime
	0.1,                                                # dt
	-1,                                                 # N_overt
	)

input_set = Hyperrectangle(low=[0.5, 0.5, 0.5, 0.5], high=[1., 1., 1., 1.])


#all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set)
all_sets, all_sets_symbolic = symbolic_bound(query, input_set)
output_sets, xvec = monte_carlo_simulate(query, input_set)

fig = plot_output_sets(all_sets)
fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/single_pendulum/single_pend.jl")

query = OvertQuery(
	SinglePendulum,                                   # problem
	"nnet_files/ARCH_COMP/controller_single_pendulum.nnet",     # network file
	Id(),                                             # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                            # query solver, "MIP" or "ReluPlex"
	50,                                               # ntime
	0.1,                                              # dt
	-1,                                                # N_overt
	)

input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])

all_sets, all_sets_symbolic = symbolic_reachability(query, input_set)

all_sets2, all_sets_symbolic2 = symbolic_reachability_with_concretization(query, input_set, 10)

output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)

idx = [1,2]
fig = plot_output_sets(all_sets; idx=idx)
fig = plot_output_sets([all_sets_symbolic]; idx=idx, linecolor=:red, fig=fig)

fig = plot_output_sets(vcat(all_sets2...); idx=idx, linecolor=:blue, fig=fig)
fig = plot_output_sets(vcat(all_sets_symbolic2...); idx=idx, linecolor=:green, fig=fig)

fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

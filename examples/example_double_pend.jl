include("../models/problems.jl")
include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("../MIP/src/overt_to_mip.jl")
include("../MIP/src/mip_utils.jl")
include("../models/double_pendulum/double_pend.jl")

query = OvertQuery(
	DoublePendulum,                                      # problem
	"nnet_files/controller_complex_double_pend.nnet",    # network file
	Id(),                                                # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                               # query solver, "MIP" or "ReluPlex"
	5,                                                   # ntime
	0.1,                                                 # dt
	-1,                                                  # N_overt
	)

input_set = Hyperrectangle(low=[1., 1., 1., 1.], high=[1.5, 1.5, 1.5, 1.5])
all_sets, all_sets_symbolic = symbolic_bound(query, input_set)
output_sets, xvec = monte_carlo_simulate(query, input_set)

fig = plot_output_sets(all_sets)
fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100)

fig = plot_output_sets_pgfplot(all_sets; labels=[L"$x$", L"$\dot{x}$"])
fig = plot_output_sets_pgfplot([all_sets_symbolic]; linecolor=:red, fig=fig)
fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, inner_points=true)

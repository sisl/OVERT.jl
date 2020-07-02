include("../models/problems.jl")
include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("../MIP/src/overt_to_mip.jl")
include("../MIP/src/mip_utils.jl")
include("../models/single_pendulum/single_pend.jl")

query = OvertQuery(
	SinglePendulum,                                      # problem
	"nnet_files/ARCH_COMP/controller_single_pendulum.nnet",        # network file
	Id(),                                                # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",                                               # query solver, "MIP" or "ReluPlex"
	10,                                                  # ntime
	0.1,                                                 # dt
	-1,                                                  # N_overt
	)

input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])

all_sets, all_sets_symbolic = symbolic_reachability(query, input_set)
output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)

# target_set =Hyperrectangle(low=[1., -0.65], high=[1.1, -0.55])
# # SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set; unsat_problem=false)
# #

fig = plot_output_sets(all_sets)
fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig)
#fig = plot_output_hist(xvec, query.ntime; fig=fig)
fig = plot_mc_trajectories(x0, xvec; fig=fig, markersize=1)

#
# fig = plot_output_sets_pgfplot(all_sets)
# fig = plot_output_sets_pgfplot([all_sets_symbolic]; linecolor=:red, fig=fig)
# fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig)

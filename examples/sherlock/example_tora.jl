include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

#controller = "nnet_files/sherlock/tora_bigger_controller_offset_10_scale_1.nnet"
controller = "/home/amaleki/Downloads/ARCH-COMP2020/benchmarks/Benchmark9-Tora/controllerTora_smallest.nnet"

query = OvertQuery(
Tora,                # problem
controller,          # controller file (nnet).
Id(),                # last layer activation layer Id()=linear, or ReLU()=relu
"MIP",               # query solver, "MIP" or "ReluPlex"
5,                   # ntime
1,                   # dt
-1,                  # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])

#all_sets = many_timestep_concretization(query, input_set)[1]

all_sets, all_sets_symbolic = symbolic_reachability(query, input_set)
output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)

fig = plot_output_sets(all_sets; idx=[1,3])
# fig = plot_output_sets([all_sets_symbolic]; linecolor=:red, fig=fig, idx=[1,3])
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100, idx=[1,3])

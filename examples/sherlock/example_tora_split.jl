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
8,                   # ntime
1,                   # dt
-1,                  # N_overt
	)

input_set1 = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.65, -0.6, -0.35, 0.6])
all_sets1, all_sets_symbolic1 = symbolic_reachability(query, input_set1)

input_set2 = Hyperrectangle(low=[0.6, -0.7, -0.35, 0.5], high=[0.65, -0.6, -0.3, 0.6])
all_sets2, all_sets_symbolic2 = symbolic_reachability(query, input_set2)

input_set3 = Hyperrectangle(low=[0.65, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.35, 0.6])
all_sets3, all_sets_symbolic3 = symbolic_reachability(query, input_set3)

input_set4 = Hyperrectangle(low=[0.65, -0.7, -0.35, 0.5], high=[0.7, -0.6, -0.3, 0.6])
all_sets4, all_sets_symbolic4 = symbolic_reachability(query, input_set4)

input_set_tot = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
output_sets, xvec, x0 = monte_carlo_simulate(query, input_set_tot)

idx = [1,3]
fig = plot_output_sets(all_sets1; idx=idx)
fig = plot_output_sets(all_sets2; idx=idx, fig=fig)
fig = plot_output_sets(all_sets3; idx=idx, fig=fig)
fig = plot_output_sets(all_sets4; idx=idx, fig=fig)
fig = plot_output_sets([all_sets_symbolic1]; linecolor=:red, fig=fig, idx=idx)
fig = plot_output_sets([all_sets_symbolic2]; linecolor=:red, fig=fig, idx=idx)
fig = plot_output_sets([all_sets_symbolic3]; linecolor=:red, fig=fig, idx=idx)
fig = plot_output_sets([all_sets_symbolic4]; linecolor=:red, fig=fig, idx=idx)
fig = plot_output_hist(xvec, query.ntime; fig=fig, nbins=100, idx=idx)

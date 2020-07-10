using JLD2
using PGFPlots
using QHull

JLD2.@load "examples/jair/data/drone_reachability_small_controller_data.jld2"

idx = [1,3]
all_sets = vcat(all_sets...)
fig = plot_output_sets_pgfplot(all_sets; idx=idx, fig=nothing, linewidth="thin",
    linecolor="black", fillalpha=0, linestyle="dashed")


# output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
# fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=idx, inner_points=false)
#
# PGFPlots.save("examples/jair/plots/tora_reachability_smallest_controller_data.tex", fig)

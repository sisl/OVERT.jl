using PGFPlots
using QHull
using JLD2

JLD2.@load "examples/jair/data/tora_reachability_smallest_controller_splitting_data.jld2"

idx = [2,3]
# all_sets = vcat(vcat(vcat(vcat(all_sets...)...)...)...)
#


all_sets_symbolic = vcat(vcat(vcat(all_sets_symbolic...)...)...)
fig = plot_output_sets_pgfplot(all_sets_symbolic[end-63:end]; idx=idx, linewidth="very thick",
    linecolor="green", fillalpha=0)

JLD2.@load "examples/jair/data/tora_reachability_smallest_controller_nosplitting_data.jld2"

all_sets = vcat(all_sets...)

fig = plot_output_sets_pgfplot(all_sets_symbolic; idx=idx, fig=fig, linewidth="very thick",
    linecolor="red", fillalpha=0)

fig = plot_output_sets_pgfplot(all_sets; idx=idx, fig=fig, linewidth="thin",
    linecolor="black", fillalpha=0, linestyle="dashed")

#output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=idx, inner_points=false)

#PGFPlots.save("examples/jair/plots/tora_reachability_smallest_controller_splitting.tex", fig)

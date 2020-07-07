using PGFPlots
using QHull
using JLD2

JLD2.@load "examples/jair/data/tora_reachability_smallest_controller_l1reg_data.jld2"

idx = [1,4]
all_sets = vcat(all_sets...)
fig = plot_output_sets_pgfplot(all_sets; idx=idx, fig=nothing, linewidth="thin",
    linecolor="black", fillalpha=0, linestyle="dashed")


fig = plot_output_sets_pgfplot(all_sets[[8, 15, 22, 28]]; idx=idx, fig=fig, linewidth="thick",
    linecolor="black", fillalpha=0)

fig = plot_output_sets_pgfplot(all_sets_symbolic; idx=idx, fig=fig, linewidth="very thick",
    linecolor="red", fillalpha=0)


#output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=idx, inner_points=false)

PGFPlots.save("examples/jair/plots/tora_reachability_smallest_controller_l1reg.tex", fig)

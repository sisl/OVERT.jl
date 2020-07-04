using PGFPlots
using QHull
using JLD2

JLD2.@load "examples/jair/data/car_reachability_smaller_controller_data.jld2"

idx = [1,4]
all_sets = vcat(all_sets...)
fig = plot_output_sets_pgfplot(all_sets; idx=idx, fig=nothing, linewidth="thin",
    linecolor="black", fillalpha=0, linestyle="dashed")

concret_idx = [7, 14, 20, 26]
fig = plot_output_sets_pgfplot(all_sets[concret_idx]; idx=idx, fig=fig, linewidth="thick",
    linecolor="black", fillalpha=0)

fig = plot_output_sets_pgfplot(all_sets_symbolic; idx=idx, fig=fig, linewidth="very thick",
    linecolor="red", fillalpha=0)


output_sets, xvec, x0 = monte_carlo_simulate(query, input_set)
fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=idx, inner_points=false)

PGFPlots.save("examples/jair/plots/car_reachability_smaller_controller.tex", fig)

using PGFPlots
using QHull
using JLD2

JLD2.@load "examples/benchmark/single_pendulum_satisfiability_small_controller_data.jld2"

idx = [1,2]
fig = plot_satisfiability_pgfplot(stats, vals, query; idx=idx)

PGFPlots.save("examples/benchmark/single_pendulum_satisfiability_small_controller.tex", fig)

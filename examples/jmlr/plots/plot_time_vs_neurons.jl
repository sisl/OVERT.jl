include("examples/jmlr/plots/plot_time_vs_neurons_functions.jl")
# defines functions, loads data, defines styles

# x_dim = 1 is vs weights as indep var
plot_time("Reachable Set Computation", 1)
plot_time("Feasibility Computation", 1)
# x_dim = 2 is activations as indep var
plot_time("Reachable Set Computation", 2)
plot_time("Feasibility Computation", 2)

# Plot 5: Sat vs Reach time for each problem (bar graph)
keys = ["S1", "S2", "C1", "C2", "C3", "ACC"]
keys = repeat(keys, outer=2)
sat_row = [single_pendulum_data[3]..., car_data[3]..., acc_data[3]...]
reach_row = [single_pendulum_data[4]..., car_data[4]..., acc_data[4]...]
comp_data = hcat(sat_row, reach_row)
ctg = repeat(["Feasibility", "Reachability"], inner = 6)
g = groupedbar(keys, comp_data, group=ctg, title="Computation Time", ylabel="Time (sec)")
StatPlots.savefig(g, "examples/jmlr/plots/reach_vs_sat_barchart.html")

# separately for TORA:
keys = ["T1", "T2"]
keys = repeat(keys, outer=2)
sat_row = tora_data[3][1:2]
reach_row = tora_data[4][1:2]
comp_data = hcat(sat_row, reach_row)
ctg = repeat(["Feasibility", "Reachability"], inner = 2)
g_tora = groupedbar(keys, comp_data, group=ctg, title="Computation Time", ylabel="Time (sec)")
StatPlots.savefig(g_tora, "examples/jmlr/plots/reach_vs_sat_barchart_tora12.html")

# separately for TORA:
keys = ["T3 Feasibility", "T3 Reachability"]
sat_row = tora_data[3][3]
reach_row = tora_data[4][3]
comp_data = reshape([sat_row, reach_row], 2)
g_tora3 = bar(keys, comp_data, title="Computation Time", ylabel="Time (sec)", yaxis=:log10)
StatPlots.savefig(g_tora3, "examples/jmlr/plots/reach_vs_sat_barchart_tora3.html")

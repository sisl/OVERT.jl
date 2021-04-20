using PGFPlots
using QHull
using JLD2
include("../../../models/problems.jl")
include("../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../OverApprox/src/overt_parser.jl")
include("../../../MIP/src/overt_to_mip.jl")
include("../../../MIP/src/mip_utils.jl")
include("../../../models/acc/acc.jl")
#include("../../../MIP/src/logic.jl")

JLD2.@load "/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jmlr/data/acc_reachability_data_check_2.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_check_2.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_check_1.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_23_longer.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_22_longer.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_21_bound.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_19_bound.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_17_medEps_extra_logging_and_plotting_numericFocus3_MIPGap1e-9.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_16_medEps_extra_logging_and_plotting_numericFocus3.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_18_medEps_extra_logging_and_plotting_numericFocus3_optimalityGap1e-9.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_17_medEps_extra_logging_and_plotting_numericFocus3_MIPGap1e-9.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_7_ADDED_1d_case_back.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_16_medEps_extra_logging_and_plotting_numericFocus3.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_14_moderateEps_extra_logging_and_plotting.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_13_bigEps_extra_logging_and_plotting.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_12_bigEps_extra_logging.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_11_1D_in_plus_extra_logging.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_10_more_logging.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_9_shorter_tspan.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_7_ADDED_1d_case_back.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_4.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIX_e19e1_2.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_ATTEMPT_BUG_FIXe19e1.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_commit_e19e1.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_commit55946.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_nov3_commit_a4ec26486fe2e63ba6e029256a469b16d9928261.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_nov_6_commit_e49f17598c9f14696aaca4e62171b74fd708637d.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_oct29_2020_commit_216275b788d29f4bf26b46d7f81839f84e09e145.jld2"
#"/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_acc_controller_data.jld2" # file from amir
#"examples/jair/data/new2/acc_reachability_acc_controller_data_35.jld2"

println("query: ", query)

all_sets = vcat(all_sets...)

# fig = plot_output_sets_pgfplot(all_sets; idx=[1,2], fig=nothing, linewidth="thin",
#     linecolor="black", fillalpha=0, linestyle="dashed")

# # fig = plot_output_sets_pgfplot(all_sets; idx=[4,5], fig=nothing, linewidth="thin",
# #     linecolor="black", fillalpha=0, linestyle="dashed")

# # concrete_idx = [16, 32, 43]
# # fig = plot_output_sets_pgfplot(all_sets[concrete_idx]; idx=idx, fig=fig, linewidth="thick",
# #     linecolor="black", fillalpha=0)

# fig = plot_output_sets_pgfplot(all_sets_symbolic; idx=[1,2], fig=fig, linewidth="very thick",
#     linecolor="red", fillalpha=0)
# # fig = plot_output_sets_pgfplot(all_sets_symbolic; idx=[4,5], fig=fig, linewidth="very thick",
# #     linecolor="red", fillalpha=0)

#query.ntime=55
mc_state_sets, xvec, x0, mc_meas_sets, yvec, y0 = monte_carlo_simulate(query, input_set, n_sim=200000)
#fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=[1,2], inner_points=false)

#PGFPlots.save("examples/jair/plots/acc_reachability.tex", fig)

# real plotting
using Plots 
plotly()
dims = [4,5]
mc_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in mc_state_sets]
concrete_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in concrete_state_sets]
symbolic_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in symbolic_state_sets]
Plots.plot(mc_sets, color="blue")
Plots.plot!(concrete_sets, color="gray")
Plots.plot!(symbolic_sets, color="red")

#### plot overt measurement sets compared to simulations
conc_meas_sets_plottable = gen_1D_sets(concrete_meas_sets, 1:length(concrete_meas_sets);  width=0.2)
Plots.plot(conc_meas_sets_plottable, color="grey")
sym_meas_sets_plottable = gen_1D_sets(symbolic_meas_sets, cumsum(concretization_intervals), width=0.2)
Plots.plot!(sym_meas_sets_plottable, color="red")
mc_meas_sets_plottable  = gen_1D_sets(mc_meas_sets, 1:length(mc_meas_sets); width=0.2)
Plots.plot!(mc_meas_sets_plottable, color="blue")

#### plot collision plot: lead position, eho position, and D-safe
dims = [1]
init_state, clean_states = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)
lead_car_pos = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in clean_states]
lead_pos_pltble = gen_1D_sets(lead_car_pos, 1:length(lead_car_pos); width=0.2)
dims= [4]
ego_car_pos = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in clean_states]
ego_pos_pltble = gen_1D_sets(ego_car_pos, 1:length(ego_car_pos); width=0.2)
# now plot!
Plots.plot(ego_pos_pltble, color="yellow")
Plots.plot!(lead_pos_pltble, color="purple", xlabel="timesteps", ylabel="Distance(m)", title="Ego (yellow), Lead (purple), Minimum Dsafe (green)")


# calculate D-safe. D_safe = D-default + Tgap*v-ego . Where D-default =10,Tgap=1.4
dims=[5]
v_ego = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in clean_states]
v_ego_pltble = gen_1D_sets(v_ego, 1:length(v_ego); width=0.2)
d_safe =  v_ego*1.4 .+ Ref(Interval(10.,10.))
d_safe_pltble =  gen_1D_sets(d_safe,  1:length(d_safe); width=0.2)

#### collision plot version 2: plot measurement set and D-safe
# Measured distance should always be larger  than the safe distance
clean_meas = clean_up_meas_sets(concrete_meas_sets, symbolic_meas_sets, concretization_intervals)
clean_meas_pltble = gen_1D_sets(clean_meas, 1:length(clean_meas); width=0.2)
Plots.plot(clean_meas_pltble, width=0.2, color="green")
Plots.plot!(d_safe_pltble, color="pink", title="Measured distance (green) and Minimum Safe Distance (pink)", xlabel="timesteps", ylabel="Distance (m)")


#####
dims=[1,2]
mc_state_sets_12 = get_subsets(output_sets, dims)
# state sets 12
concrete_state_sets_12 = get_subsets(concrete_state_sets, dims)
symbolic_state_sets_12 = get_subsets(symbolic_state_sets, dims)
mc_state_sets_12 = get_subsets(output_sets, dims)
# state sets 45
dims=[4,5]
concrete_state_sets_45 = get_subsets(concrete_state_sets, dims)
symbolic_state_sets_45 = get_subsets(symbolic_state_sets, dims)
mc_state_sets_45 = get_subsets(output_sets, dims)

Plots.plot(concrete_state_sets_12, color="grey")
Plots.plot!(symbolic_state_sets_12, color="blue")
Plots.plot!(mc_state_sets_12, color="orange")
Plots.plot!(concrete_state_sets_45, color="yellow")
Plots.plot!(symbolic_state_sets_45, color="purple")
Plots.plot!(mc_state_sets_45, color="green")
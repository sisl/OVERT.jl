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

JLD2.@load "/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/new2/acc_reachability_data_check_2.jld2"
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
output_sets, xvec, x0 = monte_carlo_simulate(query, input_set, n_sim=200000)
#fig = plot_output_hist_pgfplot(xvec, query.ntime; fig=fig, idx=[1,2], inner_points=false)

#PGFPlots.save("examples/jair/plots/acc_reachability.tex", fig)

# real plotting
using Plots 
plotly()
dims = [4,5]
mc_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in output_sets]
concrete_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in all_sets]
symbolic_sets = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in all_sets_symbolic]
Plots.plot(mc_sets, color="blue")
Plots.plot!(concrete_sets, color="gray")
Plots.plot!(symbolic_sets, color="red")
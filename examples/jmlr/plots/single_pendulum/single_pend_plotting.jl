using PGFPlots
using QHull
using JLD2
using LazySets
include("models/problems.jl")
include("OverApprox/src/OA_relational_util.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("models/single_pendulum/single_pend.jl")

controller_type = "small" # change to "small" or "big" as needed
if controller_type == "small"
    JLD2.@load "examples/jmlr/data/single_pend_-0.2167/single_pendulum_reachability_small_controller_data_1step.jld2"
    one_step_state_sets = symbolic_state_sets 
    JLD2.@load "examples/jmlr/data/single_pend_-0.2167/single_pendulum_reachability_small_controller_data.jld2"
elseif controller_type == "big"
    JLD2.@load "examples/jmlr/data/single_pend_-0.2167/single_pendulum_reachability_big_controller_data_1step.jld2"
    one_step_state_sets = symbolic_state_sets 
    JLD2.@load "examples/jmlr/data/single_pend_-0.2167/single_pendulum_reachability_big_controller_data.jld2"
end

# monte_carlo_simulate
mc_state_sets, xvec, x0, mc_meas_sets, yvec, y0 = monte_carlo_simulate(query, input_set, n_sim=1000000)

# clean up sets 
init_state, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

# plot intervals using lines in PGF plots
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$x_1\$", xlabel="timesteps", title="Single Pendulum Reachable Sets, S1 Controller")

input_set_style = "dashed, blue, mark=none"
concrete_line_style = "solid, black, mark=none"
symbolic_line_style = "solid, red, very thick, mark=none"
mc_line_style = "solid, blue, ultra thick, mark=none"
dims = [1]
conc_subsets = get_interval_subsets(one_step_state_sets, dims)
sym_subsets = get_interval_subsets(reachable_state_sets, dims)
mc_subsets = get_interval_subsets(mc_state_sets, dims)
deleteat!(mc_subsets, 1) # pop init set off start 
# plot init set
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=input_set_style, legendentry="Input Set"))
# avoid set
push!(fig, PGFPlots.Plots.Linear([-1., 26], [-0.2617, -0.2617], style="red, dashed", mark="none", legendentry="Unsafe Set"))
# plot the rest of the sets
for t in 1:query.ntime
    if t == query.ntime # last timestep
        # include legend entry
        timestep = Float64[t,t]
        push!(fig, PGFPlots.Plots.Linear(timestep, conc_subsets[t], style=concrete_line_style, legendentry="Concrete Sets"))
        push!(fig, PGFPlots.Plots.Linear(timestep,sym_subsets[t], style=symbolic_line_style, markSize=1, legendentry="OVERT Hybrid Symbolic Sets"))
        push!(fig, PGFPlots.Plots.Linear(timestep,mc_subsets[t], style=mc_line_style, markSize=1, legendentry="Monte Carlo Simulations"))
    else
        timestep = Float64[t,t]
        push!(fig, PGFPlots.Plots.Linear(timestep, conc_subsets[t], style=concrete_line_style))
        push!(fig, PGFPlots.Plots.Linear(timestep,sym_subsets[t], style=symbolic_line_style, markSize=1))
        push!(fig, PGFPlots.Plots.Linear(timestep,mc_subsets[t], style=mc_line_style, markSize=1))
    end
end

fig.legendStyle = "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/single_pendulum_$(controller_type).tex", fig)
save("examples/jmlr/plots/single_pendulum_$(controller_type).pdf", fig)
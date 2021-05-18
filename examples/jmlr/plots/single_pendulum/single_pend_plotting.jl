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

define_color("concrete_color", 0x139EAB)
define_color("symbolic_color", 0x9BFF85)
define_color("mc_color", 0x38caff)
# input_set_style = "dashed, blue, mark=none"
concrete_line_style = "solid, concrete_color, mark=none"
symbolic_line_style = "solid, symbolic_color, very thick, mark=none"
mc_line_style = "solid, mc_color, ultra thick, mark=none"
conc_style = "solid, concrete_color, mark=none, fill=concrete_color"
sym_style = "solid, symbolic_color, mark=none, fill=symbolic_color"
mc_style = "solid, mc_color, mark=none, fill=mc_color"
conc_style_trans = conc_style*", fill opacity=0.5"
sym_style_trans = sym_style*", fill opacity=0.5"
mc_style_trans = mc_style*", fill opacity=0.5"

###################################################
# Plot 1: 1D plots
###################################################

# plot intervals using lines in PGF plots
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$x_1\$", xlabel="timesteps", title="Single Pendulum Reachable Sets, $controller_type Controller")

dims = [1]
conc_subsets = get_interval_subsets(one_step_state_sets, dims)
sym_subsets = get_interval_subsets(reachable_state_sets, dims)
mc_subsets = get_interval_subsets(mc_state_sets, dims)
deleteat!(mc_subsets, 1) # pop init set off start 
# plot init set
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=concrete_line_style, legendentry="Concrete Sets"))
# plot init set
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=symbolic_line_style, legendentry="OVERT Hybrid Symbolic Sets"))
# plot init set
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=mc_line_style, legendentry="Monte Carlo Simulations"))
# avoid set
push!(fig, PGFPlots.Plots.Linear([-1., 26], [-0.2617, -0.2617], style="red, dashed", mark="none", legendentry="Unsafe Set"))
# plot the rest of the sets
for t in 1:query.ntime
    # if t == query.ntime # last timestep
    #     # include legend entry
    #     timestep = Float64[t,t]
    #     push!(fig, PGFPlots.Plots.Linear(timestep, conc_subsets[t], style=concrete_line_style, legendentry="Concrete Sets"))
    #     push!(fig, PGFPlots.Plots.Linear(timestep,sym_subsets[t], style=symbolic_line_style, markSize=1, legendentry="OVERT Hybrid Symbolic Sets"))
    #     push!(fig, PGFPlots.Plots.Linear(timestep,mc_subsets[t], style=mc_line_style, markSize=1, legendentry="Monte Carlo Simulations"))
    # else
        timestep = Float64[t,t]
        push!(fig, PGFPlots.Plots.Linear(timestep, conc_subsets[t], style=concrete_line_style))
        push!(fig, PGFPlots.Plots.Linear(timestep,sym_subsets[t], style=symbolic_line_style, markSize=1))
        push!(fig, PGFPlots.Plots.Linear(timestep,mc_subsets[t], style=mc_line_style, markSize=1))
    # end
end

fig.legendStyle = "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/single_pendulum_$(controller_type).tex", fig)
PGFPlots.save("examples/jmlr/plots/single_pendulum_$(controller_type).pdf", fig)

###################################################
# Plot 2: 2D plots
###################################################
dims = [1,2]
fig = PGFPlots.Axis(style="width=10cm, height=10cm, axis equal image", xlabel="\$x_1\$", ylabel="\$x_2\$", title="Single Pendulum Reachable Sets, $controller_type Controller")

# plot init set in both concrete and hybrid colors
inputx, inputy = get_rectangle(input_set, dims)
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=conc_style, legendentry="Concrete Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=sym_style, legendentry="OVERT Hybrid Symbolic Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=mc_style, legendentry="Monte Carlo Simulations"))
# avoid set
push!(fig, PGFPlots.Plots.Linear([-0.2617, -0.2617], [-0.9, 0.5], style="red, dashed", mark="none", legendentry="Unsafe Set"))

for t in 1:query.ntime
    if t == query.ntime
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
        # plot mc_sim points 
        points = xvec[:, t, dims]
        #plot hull
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style))
    else
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style_trans))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style_trans))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
        # plot mc simulations hull
        points = xvec[:, t, dims]
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style))
    end
end

# plot avoid set again because it gets covered up
push!(fig, PGFPlots.Plots.Linear([-0.2617, -0.2617], [-0.9, 0.25], style="red, dashed", mark="none"))

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/single_pendulum/single_pendulum_$(controller_type)_x12.tex", fig)
PGFPlots.save("examples/jmlr/plots/single_pendulum/single_pendulum_$(controller_type)_x12.pdf", fig)

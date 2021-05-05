# acc plotting
using PGFPlots
using QHull
using JLD2
using LazySets
using LinearAlgebra

include("../../../../models/problems.jl")
include("../../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../../OverApprox/src/overt_parser.jl")
include("../../../../MIP/src/overt_to_mip.jl")
include("../../../../MIP/src/mip_utils.jl")
include("../../../../models/tora/tora.jl")

using JLD2

controller = ARGS[1] #"smallest"

if controller == "smallest"
    JLD2.@load "examples/jmlr/data/tora/tora_reachability_smallest_controller_data_1step.jld2"
    one_step_state_sets = symbolic_state_sets 
    one_step_meas_sets = symbolic_meas_sets

    JLD2.@load "examples/jmlr/data/tora/tora_reachability_smallest_controller_data.jld2"

elseif controller == "smaller"
    JLD2.@load "examples/jmlr/data/tora/tora_reachability_smaller_controller_data_1step.jld2"
    one_step_state_sets = symbolic_state_sets 
    one_step_meas_sets = symbolic_meas_sets

    JLD2.@load "examples/jmlr/data/tora/tora_reachability_smaller_controller_data.jld2"
elseif controller == "big"
    JLD2.@load "examples/jmlr/data/tora/tora_reachability_big_controller_data_1step.jld2"
    one_step_state_sets = symbolic_state_sets 
    one_step_meas_sets = symbolic_meas_sets

    JLD2.@load "examples/jmlr/data/tora/tora_reachability_big_controller_data.jld2"
end

if (!@isdefined concretization_intervals) || length(concretization_intervals) == query.ntime
    concretization_intervals = [length(s) for s in concrete_state_sets] .- 1
end

concrete_state_sets = vcat(concrete_state_sets...)
symbolic_state_sets = vcat(symbolic_state_sets...)

# monte carlo simulate
mc_state_sets, xvec, x0, mc_meas_sets, yvec, y0 = monte_carlo_simulate(query, input_set, n_sim=1000000)

# clean up sets
init, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

####################################
##### plot 1: intervals of x1 ######
####################################
dim=[1]
sym_sets = get_interval_subsets(reachable_state_sets, dim)
conc_sets = get_interval_subsets(one_step_state_sets, dim)
mc_sets = get_interval_subsets(mc_state_sets, dim)
deleteat!(mc_sets, 1) # pop init set off start

# plot styles
# define_color("concrete_color", 0xc9a8ff)
# define_color("symbolic_color", 0xff5c57)
# define_color("mc_color", 0x96eaff)
define_color("concrete_color", 0x139EAB)
define_color("symbolic_color", 0x9BFF85)
define_color("mc_color", 0x38caff) #96eaff)
# define_color("concrete_color", 0x648FFF)
# define_color("symbolic_color", 0xDC267F)
# define_color("mc_color", 0xFE6100)
conc_style = "solid, concrete_color, line width=1pt, mark=none"
sym_style = "solid, symbolic_color, line width = 2 pt, mark=none"
mc_style = "solid, mc_color, line width=3pt, mark=none"
input_style = "dashed, blue, mark=none"
############################################################
# Plot: x1 and property
############################################################
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$x_1\$", xlabel="timesteps", title="TORA, State Sets")

# AVOID SETS
push!(fig, PGFPlots.Plots.Linear([-1., 16], [-2., -2.], style="red, dashed", mark="none", legendentry="Unsafe Set"))

# plot init set
dims=[1] 
input = [low(input_set)[dims]..., high(input_set)[dims]...]
push!(fig, PGFPlots.Plots.Linear([0.,0.], input, style=conc_style))
push!(fig, PGFPlots.Plots.Linear([0.,0.], input, style=sym_style))
push!(fig, PGFPlots.Plots.Linear([0.,0.], input, style=mc_style))

for t in 1:query.ntime
    timestep = Float64[t,t]
    if t == query.ntime 
        push!(fig, PGFPlots.Plots.Linear( timestep, conc_sets[t], style=conc_style, legendentry="Concrete Sets"))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_sets[t], style=sym_style, legendentry="OVERT Hybrid Symbolic Sets"))
        push!(fig, PGFPlots.Plots.Linear( timestep, mc_sets[t], style=mc_style, legendentry="Monte Carlo Simulations"))
    else
        push!(fig, PGFPlots.Plots.Linear( timestep, conc_sets[t], style=conc_style))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_sets[t], style=sym_style))
        push!(fig, PGFPlots.Plots.Linear( timestep, mc_sets[t], style=mc_style))
    end
end

# second unsafe set
push!(fig, PGFPlots.Plots.Linear([-1., 16], [2., 2.], style="red, dashed", mark="none"))

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/tora/tora_$(controller)_x1.tex", fig)
save("examples/jmlr/plots/tora/tora_$(controller)_x1.pdf", fig)

####################################
##### plot 2: boxes ######
####################################
conc_style_solid = "solid, concrete_color, thick, mark=none, fill=concrete_color"
conc_style_transparent = conc_style_solid*", fill opacity=0.5"
sym_style_solid = "solid, symbolic_color, thick, mark=none, fill=symbolic_color"
sym_style_transparent =  sym_style_solid*", fill opacity=0.5"
mc_style_solid = "solid, mc_color, thick, mark=none, fill=mc_color"
mc_style_transparent = mc_style_solid*", fill opacity=0.5"

fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$x_2\$", xlabel="\$x_3\$", title="TORA, State Sets")

# hack for legend
push!(fig, PGFPlots.Plots.Linear([-0.7, -0.7], [0.2, 0.2], style=mc_style_solid, legendentry="Monte Carlo Simulations"))

# plot init set in both concrete and hybrid colors
dims=[2,3] 
inputx, inputy = get_rectangle(input_set, dims)
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=conc_style_transparent, legendentry="Concrete Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=sym_style_transparent, legendentry="OVERT Hybrid Symbolic Sets"))

for t in 1:query.ntime
    if t == query.ntime
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style_solid))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style_solid))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style, legendentry="Monte Carlo Simulations"))
        # plot mc_sim points 
        points = xvec[:, t, dims]
        #plot hull
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style_solid))
    else
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style_transparent))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style_transparent))
    end
end

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/tora/tora_$(controller)_x23.tex", fig)
save("examples/jmlr/plots/tora/tora_$(controller)_x23.pdf", fig)
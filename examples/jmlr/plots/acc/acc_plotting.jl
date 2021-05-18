# acc plotting
using PGFPlots
using QHull
using JLD2
using LazySets
using LinearAlgebra

include("models/problems.jl")
include("OverApprox/src/OA_relational_util.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("models/acc/acc.jl")

JLD2.@load "examples/jmlr/data/acc/acc_reachability_data_1step_55.jld2"
one_step_state_sets = symbolic_state_sets 
one_step_meas_sets = symbolic_meas_sets

JLD2.@load "examples/jmlr/data/acc/acc_reachability_data_55.jld2"

# monte carlo simulate
mc_state_sets, xvec, x0, mc_meas_sets, yvec, y0 = monte_carlo_simulate(query, input_set, n_sim=1000000)

# clean up sets
init, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

reachable_meas_sets = clean_up_meas_sets(concrete_meas_sets, symbolic_meas_sets, concretization_intervals)

# plot styles
concrete_line_style = "solid, black, mark=none"
symbolic_line_style = "solid, red, very thick, mark=none"
mc_line_style = "solid, blue, ultra thick, mark=none"
############################################################
# First plot: rigorous plot of meas w/ unsafe=10
############################################################
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$y\$", xlabel="timesteps", title="Adaptive Cruise Control, Measurement Sets")
conc_meas_sets = get_interval_subsets(one_step_meas_sets, [1])
sym_meas_sets = get_interval_subsets(reachable_meas_sets, [1])
mc_meas_sets = get_interval_subsets(mc_meas_sets, [1])
push!(fig, PGFPlots.Plots.Linear([-1., length(mc_meas_sets) + 1.], [10., 10.], style="red, dashed", mark="none", legendentry="Unsafe Set"))
for t in 1:query.ntime
    timestep = Float64[t,t]
    if t == query.ntime 
        push!(fig, PGFPlots.Plots.Linear(timestep, conc_meas_sets[t], style=concrete_line_style, legendentry="Concrete Sets"))
        push!(fig, PGFPlots.Plots.Linear(timestep, sym_meas_sets[t], style=symbolic_line_style, legendentry="OVERT Hybrid Symbolic Sets"))
        push!(fig, PGFPlots.Plots.Linear(timestep, mc_meas_sets[t], style=mc_line_style, legendentry="Monte Carlo Simulations"))
    else
        push!(fig, PGFPlots.Plots.Linear(timestep, conc_meas_sets[t], style=concrete_line_style))
        push!(fig, PGFPlots.Plots.Linear(timestep, sym_meas_sets[t], style=symbolic_line_style))
        push!(fig, PGFPlots.Plots.Linear(timestep, mc_meas_sets[t], style=mc_line_style))
    end
end
fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/acc/acc_meas.tex", fig)
save("examples/jmlr/plots/acc/acc_meas.pdf", fig)

############################################################
### Second plot: car positions SYMBOLIC ONLY
############################################################
# in tex, replace with: #ab0000
define_color("lead_color", 0xffc2ca)
input_lead_style = "dashed, lead_color, mark=none"
sym_lead_style = "solid, lead_color, ultra thick, mark=none"
# ego in tex, replace with #d84a43
define_color("ego_color", 0x800000)
input_ego_style = "dashed, ego_color, mark=none"
sym_ego_style = "solid, ego_color, ultra thick, mark=none"
# d safe, in tex replace with #ff8080
define_color("dsafe_color", 0xc76666)
input_dsafe_style = "dashed, dsafe_color, mark=none"
sym_dsafe_style = "densely dotted, dsafe_color, very thick, mark=none"

fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="Distance Along Roadway (m)", xlabel="timesteps", title="Adaptive Cruise Control, State Sets")
dim = [1] # x_lead
sym_lead_sets = get_interval_subsets(reachable_state_sets, dim)
dim = [4] # x_ego
sym_ego_sets = get_interval_subsets(reachable_state_sets, dim)
dim = [5] # v_ego
sym_dsafe_sets = get_interval_subsets(reachable_state_sets, dim) .* 1.4 .+ Ref([10., 10.])
# construct dsafe sets extending from behind lead car position
sym_dsafe_behind_ego = [[minimum(sym_lead_sets[i]) - maximum(sym_dsafe_sets[i]), minimum(sym_lead_sets[i])] for i in 1:length(sym_dsafe_sets)]


# plot init sets
dims=[1] # lead
lead_input = [low(input_set)[dims]..., high(input_set)[dims]...]
push!(fig, PGFPlots.Plots.Linear([0.,0.], lead_input, style=input_lead_style, legendentry="Lead Position Input Set"))
##############
dims=[5] # v_ego
d_safe_i = [low(input_set)[dims]..., high(input_set)[dims]...].*1.4 .+ 10.
dsafe_input = [minimum(lead_input) - maximum(d_safe_i), minimum(lead_input)]
push!(fig, PGFPlots.Plots.Linear([0.,0.], dsafe_input, style=input_dsafe_style, legendentry="Minimum Safe Distance Input Set"))
##############
dims=[4] # ego
ego_input = [low(input_set)[dims]..., high(input_set)[dims]...]
push!(fig, PGFPlots.Plots.Linear([0.,0.], ego_input, style=input_ego_style, legendentry="Ego Position Input Set"))

for t in 1:query.ntime
    timestep = Float64[t,t]
    if t == query.ntime 
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_lead_sets[t], style=sym_lead_style, legendentry="Lead Position, OVERT Hybrid Symbolic"))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_dsafe_behind_ego[t], style=sym_dsafe_style, legendentry="Minimum Safe Distance, OVERT Hybrid Symbolic"))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_ego_sets[t], style=sym_ego_style, legendentry="Ego Position, OVERT Hybrid Symbolic"))
    else
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_lead_sets[t], style=sym_lead_style))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_dsafe_behind_ego[t], style=sym_dsafe_style))
        push!(fig, PGFPlots.Plots.Linear( timestep, sym_ego_sets[t], style=sym_ego_style))
    end
end
fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/acc/acc.tex", fig)
save("examples/jmlr/plots/acc/acc.pdf", fig)

############################################################
### Third plot: 2D plots of either velocity or accel 
############################################################
concrete_set_style = "solid, black, mark=none, fill=black"
symbolic_set_style = "solid, ego_color, mark=none, fill=ego_color"
mc_set_style = "solid, blue, mark=none, fill=blue"
concrete_set_trans = concrete_set_style*", fill opacity=0.5"
symbolic_set_trans = symbolic_set_style*", fill opacity=0.5"
mc_set_trans = mc_set_style*", fill opacity=0.5"

dims=[2,5] # velocity
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$v_{lead}~(m/s)\$", xlabel="\$v_{ego}~(m/s)\$", title="ACC Velocity Reachable Sets")

# plot init set in both concrete and hybrid colors
inputx, inputy = get_rectangle(input_set, dims)
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=concrete_set_style, legendentry="Concrete Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=symbolic_set_style, legendentry="OVERT Hybrid Symbolic Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=mc_set_style, legendentry="Monte Carlo Simulations"))

for t in 1:query.ntime
    if t == query.ntime
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=concrete_set_style))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=symbolic_set_style))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
        # plot mc_sim points 
        points = xvec[:, t, dims]
        #plot hull
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_set_style))
    else
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=concrete_set_trans))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=symbolic_set_trans))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
        # plot mc simulations hull
        points = xvec[:, t, dims]
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_set_style))
    end
end

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/acc/acc_x$(dims[1])$(dims[2]).tex", fig)
PGFPlots.save("examples/jmlr/plots/acc/acc_x$(dims[1])$(dims[2]).pdf", fig)

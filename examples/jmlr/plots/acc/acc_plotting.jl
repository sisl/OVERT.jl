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

JLD2.@load "examples/jmlr/data/acc/acc_reachability_data_1step.jld2"
one_step_state_sets = symbolic_state_sets 
one_step_meas_sets = symbolic_meas_sets

JLD2.@load "examples/jmlr/data/acc/acc_reachability_data.jld2"

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
push!(fig, PGFPlots.Plots.Linear([-1., 36.], [10., 10.], style="red, dashed", mark="none", legendentry="Unsafe Set"))
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

save("examples/jmlr/plots/acc_meas.tex", fig)
save("examples/jmlr/plots/acc_meas.pdf", fig)

############################################################
### Second plot: car positions SYMBOLIC ONLY
############################################################
# in tex, replace with: #139eab
input_lead_style = "dashed, black, mark=none"
conc_lead_style = "solid, black, mark=none"
sym_lead_style = "solid, black, very thick, mark=none"
mc_lead_style = "solid, black, ultra thick, mark=none"
# ego in tex, replace with #05d2ad
input_ego_style = "dashed, blue, mark=none"
conc_ego_style = "solid, blue, mark=none"
sym_ego_style = "solid, blue, very thick, mark=none"
mc_ego_style = "solid, blue, ultra thick, mark=none"
# d safe, in tex replace with #9bff85
input_dsafe_style = "dashed, red, mark=none"
conc_dsafe_style = "solid, red, mark=none"
sym_dsafe_style = "solid, red, very thick, mark=none"
mc_dsafe_style = "solid, red, ultra thick, mark=none"
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$y\$", xlabel="timesteps", title="Adaptive Cruise Control, State Sets")
dim = [1] # x_lead
conc_lead_sets = get_interval_subsets(one_step_state_sets, dim)
sym_lead_sets = get_interval_subsets(reachable_state_sets, dim)
mc_lead_sets = get_interval_subsets(mc_state_sets, dim)
deleteat!(mc_lead_sets, 1)
dim = [4] # x_ego
conc_ego_sets = get_interval_subsets(one_step_state_sets, dim)
sym_ego_sets = get_interval_subsets(reachable_state_sets, dim)
mc_ego_sets = get_interval_subsets(mc_state_sets, dim)
deleteat!(mc_ego_sets, 1)
# construct dsafe sets extending from behind lead car position
dim = [5] # v_ego
conc_dsafe_sets = get_interval_subsets(one_step_state_sets, dim) .* 1.4 .+ Ref([10., 10.])

sym_dsafe_sets = get_interval_subsets(reachable_state_sets, dim) .* 1.4 .+ Ref([10., 10.])
mc_dsafe_sets = get_interval_subsets(mc_state_sets, dim) .* 1.4 .+ Ref([10., 10.])
deleteat!(mc_dsafe_sets, 1)

function plot_triples(fig, x, conc, sym, mc, conc_style, sym_style, mc_style; name="", legendentry="")
    if legendentry == ""
        push!(fig, PGFPlots.Linear(x, conc, style=conc_style))
        push!(fig, PGFPlots.Linear(x, sym, style=sym_style))
        push!(fig, PGFPlots.Linear(x, mc, style=mc_style))
    else
        push!(fig, PGFPlots.Linear(x, conc, style=conc_style, legendentry="Concrete $name Sets"))
        push!(fig, PGFPlots.Linear(x, sym, style=sym_style, legendentry="OVERT Hybrid Symbolic $name Sets"))
        push!(fig, PGFPlots.Linear(x, mc, style=mc_style, legendentry="Monte Carlo Simulation $name Sets"))
    end
end

# plot init sets
dims=[1] # lead
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=input_lead_style, legendentry="Lead Position Input Set"))
dims=[4] # ego
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...], style=input_ego_style, legendentry="Ego Position Input Set"))
dims=[5] # v_ego
push!(fig, PGFPlots.Plots.Linear([0.,0.], [low(input_set)[dims]..., high(input_set)[dims]...].*1.4 .+ 10., style=input_dsafe_style, legendentry="Minimum Safe Distance Input Set"))

for t in 1:query.ntime
    timestep = Float64[t,t]
    if t == query.ntime 
        plot_triples(fig, timestep, conc_ego_sets[t], sym_ego_sets[t], sym_ego_sets[t],
                                    conc_ego_style, sym_ego_style, mc_ego_style, name="Ego Position")
        plot_triples(fig, timestep, conc_lead_sets[t], sym_lead_sets[t], sym_lead_sets[t],
                                    conc_lead_style, sym_lead_style, mc_lead_style, name="Lead Position")
        plot_triples(fig, timestep, conc_dsafe_sets[t], sym_dsafe_sets[t], sym_dsafe_sets[t],
                                    conc_dsafe_style, sym_dsafe_style, mc_dsafe_style, name="Minimum Safe Distance")
    else
        plot_triples(fig, timestep, conc_ego_sets[t], sym_ego_sets[t], sym_ego_sets[t],
                                    conc_ego_style, sym_ego_style, mc_ego_style)
        plot_triples(fig, timestep, conc_lead_sets[t], sym_lead_sets[t], sym_lead_sets[t],
                                    conc_lead_style, sym_lead_style, mc_lead_style)
        plot_triples(fig, timestep, conc_dsafe_sets[t], sym_dsafe_sets[t], sym_dsafe_sets[t],
                                    conc_dsafe_style, sym_dsafe_style, mc_dsafe_style)
    end
end
fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

save("examples/jmlr/plots/acc_meas.tex", fig)
save("examples/jmlr/plots/acc_meas.pdf", fig)

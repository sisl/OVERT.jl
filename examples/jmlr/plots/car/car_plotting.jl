# car plotting: 2 plots (replaces 4 old plots in paper)
# 2D sets of x1 vs x2 
# show how they are never within "goal set"
# ALSO: plot of SAT trajectory! (second plot) take vals from SAT and simulate forward in time

using PGFPlots
using QHull
using JLD2
using FileIO
using LazySets
using LinearAlgebra
include("../../../../models/problems.jl")
include("../../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../../OverApprox/src/overt_parser.jl")
include("../../../../MIP/src/overt_to_mip.jl")
include("../../../../MIP/src/mip_utils.jl")
include("../../../../models/car/simple_car.jl")

controller = ARGS[1] # "smallest"

one_step_data = load("examples/jmlr/data/car_10step/car_reachability_$(controller)_controller_data_1step.jld2")
one_step_state_sets = one_step_data["symbolic_state_sets"] 
one_step_meas_sets = one_step_data["symbolic_meas_sets"]

data = load("examples/jmlr/data/car_10step/car_reachability_$(controller)_controller_data.jld2")

concrete_state_sets = vcat(data["concrete_state_sets"]...)
symbolic_state_sets = vcat(data["symbolic_state_sets"]...)

# monte carlo simulate trajectories for first plot
mc_state_sets, xvec, x0, mc_meas_sets, yvec, y0 = monte_carlo_simulate(data["query"], data["input_set"], n_sim=1000000)

# clean up sets
init, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, data["concretization_intervals"])

#########################################
# Define Colors and Styles ##############
#########################################
define_color("concrete_color", 0x139EAB)
define_color("symbolic_color", 0x9BFF85)
define_color("mc_color", 0x38caff)
define_color("goal_color", 0xa866ff)
conc_style_solid = "solid, concrete_color, mark=none, fill=concrete_color"
conc_style_transparent = conc_style_solid*", fill opacity=0.5, opacity=0.5"
sym_style_solid = "solid, symbolic_color, mark=none, fill=symbolic_color"
sym_style_transparent =  sym_style_solid*", fill opacity=0.5, opacity=0.5"
mc_style_solid = "solid, mc_color, mark=none, fill=mc_color"
mc_style_transparent = "solid, red, mark=none, fill=red, fill opacity=0.5, opacity=0.5"#mc_style_solid*", fill opacity=0.5"

goal_style = "solid, goal_color, thick, mark=none, fill=goal_color"



#########################################
##### plot 1: 2D sets of x1 vs x2  ######
#########################################
dims=[1,2]
fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$y~(m)\$", xlabel="\$x~(m)\$", title="Car Position Reachable Sets")

# push!(fig, PGFPlots.Plots.Linear([9., 9.], [-4., -4.], style=mc_style_transparent, legendentry="Monte Carlo Simulations hyperrectangular hull"))

# plot init set in both concrete and hybrid colors
inputx, inputy = get_rectangle(data["input_set"], dims)
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=conc_style_solid, legendentry="Concrete Sets"))
push!(fig, PGFPlots.Plots.Linear(inputx, inputy, style=sym_style_solid, legendentry="OVERT Hybrid Symbolic Sets"))

# hack for legend
push!(fig, PGFPlots.Plots.Linear([9., 9.], [-4., -4.], style=mc_style_solid, legendentry="Monte Carlo Simulations convex hull"))

# plot goal set 
goal_set = Hyperrectangle(low=[-0.6, -0.2], high=[0.6, 0.2])
goal_x, goal_y = get_rectangle(goal_set, dims)
push!(fig, PGFPlots.Plots.Linear(goal_x, goal_y, style=goal_style, legendentry="Goal Set"))

### plot init set of mc simulations
points = x0[:, dims]
#plot hull
border_idx = chull(points).vertices
x = points[border_idx, 1]
y = points[border_idx, 2]
push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style_solid))

for t in 1:data["query"].ntime
    # if t == data["query"].ntime
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style_solid))
        push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style_solid))
        # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
        # plot mc_sim points 
        points = xvec[:, t, dims]
        #plot hull
        border_idx = chull(points).vertices
        x = points[border_idx, 1]
        y = points[border_idx, 2]
        push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style_solid))
    # else
    #     push!(fig, PGFPlots.Plots.Linear( get_rectangle(one_step_state_sets[t], dims)..., style=conc_style_transparent))
    #     push!(fig, PGFPlots.Plots.Linear( get_rectangle(reachable_state_sets[t], dims)..., style=sym_style_transparent))
    #     # push!(fig, PGFPlots.Plots.Linear( get_rectangle(mc_state_sets[t+1], dims)..., style=mc_style_transparent))
    #     # plot mc simulations hull
    #     points = xvec[:, t, dims]
    #     border_idx = chull(points).vertices
    #     x = points[border_idx, 1]
    #     y = points[border_idx, 2]
    #     push!(fig, PGFPlots.Plots.Linear([x..., x[1]], [y..., y[1]], style=mc_style_solid))
    # end
end

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/car/car_$(controller)_x12.tex", fig)
PGFPlots.save("examples/jmlr/plots/car/car_$(controller)_x12.pdf", fig)

#########################################
##### plot 2: SAT trajectory  ######
#########################################
# counter examples are "easy" here because no trajectory reaches the goal :'D
# load sat data 
q1_data = load("examples/jmlr/data/car_10step/car_satisfiability_$(controller)_controller_data_q1.jld2")
q2_data = load("examples/jmlr/data/car_10step/car_satisfiability_$(controller)_controller_data_q2.jld2")
final_data = load("examples/jmlr/data/car_10step/car_satisfiability_$(controller)_controller_data_final_result.jld2")
# final data contains the SATuses "sat" or "unsat" for each timestep. 
# if one of the queries has multiple timesteps that were sat, this produces multiple possible counterexample "hints"
# we take the first timestep values of any "sat" step, and simulate them forward to get a real counterexample. we could do this for every timestep that returns SAT
x0 = [q2_data["statii"][1][Symbol("$(v)_1")] for v in q2_data["query"].problem.input_vars] 
xvec_mc = monte_carlo_one_simulate(q2_data["query"], x0)

fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$y~(m)\$", xlabel="\$x~(m)\$", title="Car Counter Example")

dims=[1,2]
# plot goal set 
goal_set = Hyperrectangle(low=[-0.6, -0.2], high=[0.6, 0.2])
goal_x, goal_y = get_rectangle(goal_set, dims)
push!(fig, PGFPlots.Plots.Linear(goal_x, goal_y, style=goal_style, legendentry="Goal Set"))

counter_example_style = "concrete_color, solid, mark=*, mark options={concrete_color, fill=concrete_color}"
# plot the counter example 
push!(fig, PGFPlots.Plots.Linear(xvec_mc[:, dims[1]], xvec_mc[:, dims[2]], style=counter_example_style, legendentry="Counter Example"))

fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/car/car_$(controller)_counter_example.tex", fig)
PGFPlots.save("examples/jmlr/plots/car/car_$(controller)_counter_example.pdf", fig)
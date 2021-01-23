# script to simulate MDP policy with overt dynamics

using Random
using Distributions
using LinearAlgebra
using POMDPs
using StaticArrays
using Parameters
using GridInterpolations
using POMDPModelTools
using POMDPModels
using Plots
using LocalFunctionApproximation
using LocalApproximationValueIteration 
using JLD2
include("../../../../jan_demo/GoAround/src/GoAround.jl")

include("../../models/problems.jl")
# include("../../OverApprox/src/overapprox_nd_relational.jl")
# include("../../OverApprox/src/overt_parser.jl")
# include("../../MIP/src/overt_to_mip.jl")
# include("../../MIP/src/mip_utils.jl")
include("../../models/landing/landing.jl")

JLD2.@load "/Users/Chelsea/Dropbox/AAHAA/jan_demo/GoAround/policy.jld2"

ga = GoAroundMDP()

function compute_accel(x)
    xc, vc, yp, vp = x # unpack state from overt dynamics
    @debug("state from overt: xc, vc, yp, vp = ", xc, vc, yp, vp)
    timetoland = (vp - landing_v_thresh)/landing_accel
    @debug("timetoland=", timetoland)
    ac_sym = action(approx_policy, GAState(xc, vc, timetoland))
    @debug("Action is:", ac_sym)
    act_ind = actionindex(ga, ac_sym)
    act_ind -= 1 # shift to [0,1] from [1,2]
    @debug("act index: ", act_ind)
    return [act_ind]
end

n_sim = 1000
dynamics_func = landing_dynamics
input_set = input_set = Hyperrectangle(low=[700,-15,-5, 99],
                                        high=[750, 0, 5, 100])
n_time = 45
dt = 1
n_states = 4
min_x = [[Inf64  for n = 1:n_states] for m = 1:n_time]
max_x = [[-Inf64 for n = 1:n_states] for m = 1:n_time]
sim_vals = zeros(n_sim, n_time, n_states)
x0 = zeros(n_sim, n_states)
for i = 1:n_sim
    x = rand(n_states)
    x .*= input_set.radius * 2
    x .+= input_set.center - input_set.radius
    # deal with integer velocity
    x[end] = rand([99,100])
    #
    x0[i, :] = x
    for j = 1:n_time
        u = compute_accel(x)
        dx = dynamics_func(x, u)
        x = x + dx*dt
        min_x[j] = min.(x, min_x[j])
        max_x[j] = max.(x, max_x[j])
        sim_vals[i, j, :] = x
    end
end

# plot system traces first
plot(sim_vals[1, :, [4]])
for i =2:n_sim
    plot!(sim_vals[i, :, [4]])
end

plot(sim_vals[1, :, [1]])
for i =2:n_sim
    plot!(sim_vals[i, :, [1]])
end

# compute_sets
output_sets = [input_set]
for (m1, m2) in zip(min_x, max_x)
    println(m1, m2)
    push!(output_sets, Hyperrectangle(low=m1, high=m2))
end
dims = [1,4]
vp_xc_subset = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in output_sets]
plot(vp_xc_subset, color="blue")

# plot sim points
scatter!(sim_vals[:, end, 1], sim_vals[:, end, 4])

# avoid set 
# assume variables: 
# [xc vp] 
# [0   1] < v_thresh
# [1   0] > 400
# [1   0] < 600
h = HPolyhedron([[0. 1.];[-1. 0.]; [1. 0.]], [66.,-400.,600.])
plot!(h, color="red", xlim=(0, 1000), (50, 110))
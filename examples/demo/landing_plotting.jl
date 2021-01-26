using Plots
using QHull
using JLD2
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/landing/landing.jl")

#JLD2.@load "/Users/Chelsea/Desktop/AAHAA/src/OVERT/examples/landing/data/landing_reachability_data.jld2"

println("query: ", query)
n_sim=10000
input_set = Hyperrectangle(low=[700,-15,-5, 99],
                           high=[750, -14, 5, 100])

# led to a crash
# input_set = Hyperrectangle(low=[900,-5,-5, 99],
#                            high=[1000, -4, 5, 100])
#input_set = Hyperrectangle(low=[700,-2,-5, 99],
#                            high=[750, 0, 5, 100])
# input_set = Hyperrectangle(low=[500,-2,-5, 99],
#                             high=[600, 0, 5, 100])



# all_sets = vcat(all_sets...)
output_sets, sim_vals, x0 = monte_carlo_simulate(query, input_set, n_sim=n_sim);
car_pose = sim_vals[:, :, [1]];
car_vel = sim_vals[:, :, [2]];
plane_pose = sim_vals[:,:,[3]];
plane_vel = sim_vals[:,:,[4]];

# plot system traces first
# plane velocity over time
plot(sim_vals[1, :, [4]])
for i =2:100
    plot!(sim_vals[i, :, [4]])
end
plot!(sim_vals[end, :, [4]], title="Plane velocity over time")

# car position over time
plot(sim_vals[1, :, [1]])
for i =2:100
    plot!(sim_vals[i, :, [1]])
end
plot!(sim_vals[end, :, [1]], title="Car position over time")

# plot sets
dims = [1,4]
vp_xc_subset = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in output_sets]
plot(vp_xc_subset, color="blue")

# plot sim points at final timestep
scatter!(sim_vals[:, end, 1], sim_vals[:, end, 4])


# plot positions of plane and car
runway = Hyperrectangle(low=[2800, 500-30], high=[5000,500+30])
carway = Hyperrectangle(low=[3000-10,-500], high=[3000+10,1200])
i = rand(1:n_sim) # sim by time
global plane_color = "blue"
animation = @animate for t in 1:size(sim_vals)[2]
    # select color for plane
    if plane_vel[i,t] <= landing_v_thresh
        # landed 
        # is safe?
        if abs(car_pose[i,t] - 500) < 100
            # unsafe 
            global plane_color= "red"
        else
            global plane_color = "green"
        end
    else
        if plane_color ∉ ["red", "green"]
            global plane_color = "blue"
        end
    end
    # backdrop
    # plane Runway
    plot(runway, color="grey", alpha=0.5, label="")
    plot!(carway, color="grey", alpha=0.5, label="")
    # plane
    plot!(plane_pose[i, 1:t], ones(t)*500, color=plane_color, label="Plane",  legend=:topleft, title="Landing Problem Simulation") #, Plane Vel="*string(plane_vel[i,t]))
    scatter!([plane_pose[i, 1:t]], ones(t)*500, color=plane_color, markersize=1, label="")
    scatter!([plane_pose[i, t]], [500], color=plane_color, markersize=5, label="")
    # car
    plot!(ones(t)*3000, car_pose[i, 1:t], color="green", label="Car")
    scatter!(ones(t)*3000,car_pose[i, 1:t], color="green", markersize=1, label="")
    scatter!([3000],[car_pose[i, t]], color="green", markersize=3, xlim=(0,3500), ylim=(-500, 1000), label="")
end
gif(animation, "plane_landing_sim.gif", fps=4)

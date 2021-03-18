using Plots
plotly()
using JLD2
using QHull

include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/landing/landing.jl")

JLD2.@load "/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/demo/landing_reachability_1_data.jld2"

println("query: ", query)

all_sets = vcat(all_sets...)

n_sim=100000

output_sets, sim_vals, x0 = monte_carlo_simulate(query, input_set, n_sim=n_sim);

# cut sets down to appropriate dimensions
# monte carlo
dims = [1,4]
mc_vp_xc = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in output_sets]  

# verification sets
sym_vp_xc = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in all_sets_symbolic]
all_vp_xc = [Hyperrectangle(low=low(h)[dims], high=high(h)[dims]) for h in all_sets] 

# get indices of duplicate sets within all sets and all sets symbolic 
dup_idx = findall(all_vp_xc .∈ Ref(sym_vp_xc))
deleteat!(all_vp_xc, dup_idx)

# plot
p = plot(0,0)
j=1
animation = @animate for i = 1:query.ntime+1
    #if i <= query.ntime+1
        plot!(all_vp_xc[i], color="grey")
        plot!(mc_vp_xc[i], color="blue")
        if i ∈ [11, 21, 31, 39, 46]
            plot!(sym_vp_xc[j], color="red")
            global j += 1
        end
    #else # if we are at time n+2
        x = 400
        y = 50
        plot!(Shape(x .+ [0,200,200,0],y .+ [0,0,16,16]), color="black", label="", xlim=(-200,1000), ylim=(50, 100))
    #end
end
gif(animation, "landing_reachable_sets.gif", fps=4)


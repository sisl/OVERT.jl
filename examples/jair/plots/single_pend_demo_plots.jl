# single pendulum plotting 

# /Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/single_pendulum_reachability_small_controller_data.jld2

using Plots
using PlotlyBase
plotly()
using QHull
using JLD2

include("../../../models/problems.jl")
include("../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../OverApprox/src/overt_parser.jl")
include("../../../MIP/src/overt_to_mip.jl")
include("../../../MIP/src/mip_utils.jl")
include("../../../models/single_pendulum/single_pend.jl")

@load "/Users/Chelsea/Dropbox/AAHAA/src/OVERT/examples/jair/data/single_pendulum_reachability_small_controller_data.jld2"

print("query: ", query)

all_sets = vcat(all_sets...)

mc_sets, xvec, x0 = monte_carlo_simulate(query, input_set)

# see how 2D plots look?
# if hard to interpret, go 1D ?
plot!(mc_sets, color="blue")
plot!(all_sets, color="gray")
plot!(all_sets_symbolic, color="red")


# first clean up all_sets
all_sets_clean = all_sets # ugh figure out how to deepcopy
deleteat!(all_sets_clean, [17, 33]) # remove symbolic sets
# 17 is the symbolic set
# 33 is symbolic
# animation plotting
p = plot(0,0)
j = 1
for i = 1:query.ntime+1 # includes starting set as well
    # symbolic: 15, 30, 40 (not including starting sets)
    plot!(all_sets[i], color="grey")
    plot!(mc_sets[i], color="blue")
    fname = "pend1_demo/im"*string(i)*".pdf"
    Plots.savefig(p, fname)
    if i ∈ [16, 31, 41]
        plot!(all_sets_symbolic[j], color="red")
        global j += 1
        fname = "pend1_demo/im"*string(i)*".1.pdf"
        Plots.savefig(p, fname)
    end
end
# avoid set
#plot!(HalfSpace([1., 0.], -.5))
#plot!(0,0)
    

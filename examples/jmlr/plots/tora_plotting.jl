include("../../../models/problems.jl")
include("../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../OverApprox/src/overt_parser.jl")
include("../../../MIP/src/overt_to_mip.jl")
include("../../../MIP/src/mip_utils.jl")
include("../../../models/tora/tora.jl")

using JLD2

@load "examples/jmlr/data/tora_reachability_smallest_controller_data.jld2"

# plot any violations
target_sets = avoid_sets 
target_set_color="black"
target_set_label="Avoid Set"
dims=[1,2]
dirname="examples/jmlr/data/"
plotname="tora_reachability_smallest"

plotly()
rs = get_subsets(reachable_sets, dims)
ts = get_subsets(target_sets, dims)
plot(get_subset(input_set, dims), color="yellow")
idx = cumsum(concretization_intervals)
for (i, s) = enumerate(rs)
    if i ∈ idx
        plot!(s, color="red", xlim=xlims, ylim=ylims)
    else
        plot!(s, color="grey", xlim=xlims, ylim=ylims)
    end
end
plot!(ts, color=target_set_color, label=target_set_name)
Plots.savefig(p, dirname*plotname*".html")
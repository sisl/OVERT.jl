using PGFPlots
using QHull
using DelimitedFiles
using JLD2
using FileIO
using LazySets
include("models/problems.jl")
include("OverApprox/src/OA_relational_util.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("models/single_pendulum/single_pend.jl")

# plot comparison between nnv and overt
# decided to compare actually, the original experiments. not smaller, modified ones.
directory = ENV["HOME"]*"/Dropbox/AAHAA/src/OVERT/examples/jmlr/data/single_pend_-0.2167/nnv_pendulum_experiment_8148/set_dir/"

fname = "pendulum_reach_set_tstep_"
nnv_reach_set_vertices = []
for i=0:25 # init set too
    fullfname = directory*fname*"$i.txt"
    push!(nnv_reach_set_vertices, readdlm(fullfname, ','))
end

# now get the convex hull of the vertices
nnv_reach_sets = []
for points in nnv_reach_set_vertices
    if size(points,1) < 4
        push!(nnv_reach_sets, points)
    else # >= 4
        border_idx = chull(points).vertices
        ordered_vertices = vcat(points[border_idx,:], points[border_idx,:][1,:]')
        push!(nnv_reach_sets, ordered_vertices)
    end
end

# load OVERT data
overt_data = load("examples/jmlr/data/single_pend_-0.2167/single_pendulum_reachability_small_controller_data.jld2")
init_state, overt_reach_sets = clean_up_sets(overt_data["concrete_state_sets"], overt_data["symbolic_state_sets"], overt_data["concretization_intervals"])

# plot styles
define_color("symbolic_color", 0x9BFF85)
define_color("nnv_color", 0xbc86d1)
nnv_style = "solid, nnv_color, mark=none, fill=nnv_color"
nnv_style_trans = nnv_style*", fill opacity=0.5"
overt_style = "solid, symbolic_color, mark=none, fill=symbolic_color"
overt_style_trans = overt_style*", fill opacity=0.5"

# make PGFPlot
fig = PGFPlots.Axis(style="width=8cm, height=8cm", xlabel="\$x_1\$", ylabel="\$x_2\$", title="Single Pendulum Reachable Sets, Comparison")

# avoid set
push!(fig, PGFPlots.Plots.Linear([-0.2617, -0.2617], [-10., 10.], style="red, dashed", mark="none", legendentry="Unsafe Set"))

# plot init set
push!(fig, PGFPlots.Plots.Linear(nnv_reach_sets[1][:,1], nnv_reach_sets[1][:,2], style=overt_style, legendentry="OVERT"))

push!(fig, PGFPlots.Plots.Linear(nnv_reach_sets[1][:,1], nnv_reach_sets[1][:,2], style=nnv_style, legendentry="nnv"))

for t in 1:10
    push!(fig, PGFPlots.Plots.Linear(nnv_reach_sets[t+1][:,1], nnv_reach_sets[t+1][:,2], style=nnv_style_trans))
    #push!(fig, PGFPlots.Plots.Linear(get_rectangle(overt_reach_sets[t], [1,2])..., style=overt_style_trans))
end

for t in 1:10
    #push!(fig, PGFPlots.Plots.Linear(nnv_reach_sets[t+1][:,1], nnv_reach_sets[t+1][:,2], style=nnv_style_trans))
    push!(fig, PGFPlots.Plots.Linear(get_rectangle(overt_reach_sets[t], [1,2])..., style=overt_style_trans))
end

fig.legendStyle = "at={(1.05,1.0)}, anchor=north west"

PGFPlots.save("examples/jmlr/plots/single_pendulum/single_pendulum_nnv_comparison.tex", fig)
PGFPlots.save("examples/jmlr/plots/single_pendulum/single_pendulum_nnv_comparison.pdf", fig)
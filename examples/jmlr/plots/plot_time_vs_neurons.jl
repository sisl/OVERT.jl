# plotting time
using JLD2
using FileIO
using PGFPlots
using StatPlots

# number of weights in network
function get_number_of_weights(nn::Network)
    n = 0
    for l in nn.layers
        n += prod(size(l.weights))
    end
    return n
end

# get number of relus in a network
function get_activations(nn::Network)
    n = 0
    for l in nn.layers 
        if l.activation == ReLU()
            n += n_nodes(l)
        end
    end
    return n
end

function get_nnet_metadata(nnet)
    nnet = read_nnet(nnet)
    w = get_number_of_weights(nnet)
    a = get_activations(nnet)
    return w, a
end

function get_reach_time(fname)
    data = load(fname)
    return ceil(data["dt"] + data["dt_check"])
end

function get_sat_time(fname, n_queries)
    t = 0
    if !occursin("acc", fname) && !occursin("single_pendulum", fname)
        for q in n_queries
            data = load(fname*"_q$(q).jld2")
            t += data["dt"]
        end
    else 
        fname = fname*".jld2"
        data = load(fname)
        t += data["dt"]
    end
    return ceil(t)
end

function get_data(controllers, nnet, prob_dir, reach_s, sat_s)
    weights = Float64[]
    acts = Float64[]
    feas_time = Float64[]
    reach_time = Float64[]
    for c in controllers
        w, a = get_nnet_metadata(format(nnet, c))
        dir = data_dir*prob_dir
        rt = get_reach_time(dir*format(reach_s, c))
        println("reach_time: $rt")
        ft = get_sat_time(dir*format(sat_s, c), [1, 2])
        println("feas_time: $ft")
        push!(weights, w)
        push!(acts, a)
        push!(feas_time, ft)
        push!(reach_time, rt)
    end
    return [weights, acts, feas_time, reach_time]
end

# collect all data 
data_dir = "examples/jmlr/data/"
######### acc 
acc_w, acc_a = get_nnet_metadata("nnet_files/jair/acc_controller.nnet")
acc_dir = data_dir*"acc/acc"
acc_reach_time = get_reach_time(acc_dir*"_reachability_data_55.jld2")
println("acc_reach_time: $acc_reach_time")
acc_feas_time = get_sat_time(acc_dir*"_satisfiability_data_55", [])
acc_data = [[acc_w], [acc_a], [acc_feas_time], [acc_reach_time]]
println("acc_feas_time: $acc_feas_time")

######### car
car_controllers = ["smallest", "smaller", "big"]
car_nnet = "nnet_files/jair/car_{:s}_controller.nnet"
car_prob_dir = "car_10step/car"
car_reach = "_reachability_{:s}_controller_data.jld2"
car_sat = "_satisfiability_{:s}_controller_data"
car_data = get_data(car_controllers, car_nnet, car_prob_dir, car_reach, car_sat)
println("cardata: $car_data")

######### single_pend
single_pendulum_controllers = ["small", "big"]
single_pendulum_nnet = "nnet_files/jair/single_pendulum_{:s}_controller.nnet"
single_pendulum_prob_dir = "single_pend_-0.2167/single_pendulum"
single_pendulum_reach = "_reachability_{:s}_controller_data.jld2"
single_pendulum_sat = "_satisfiability_{:s}_controller_data"
single_pendulum_data = get_data(single_pendulum_controllers, single_pendulum_nnet, single_pendulum_prob_dir, single_pendulum_reach, single_pendulum_sat)
println("single_pendulumdata: $single_pendulum_data")

######### tora
tora_controllers = ["smallest", "smaller", "big"]
tora_nnet = "nnet_files/jair/tora_{:s}_controller.nnet"
tora_prob_dir = "tora/tora"
tora_reach = "_reachability_{:s}_controller_data.jld2"
tora_sat = "_satisfiability_{:s}_controller_data"
tora_data = get_data(tora_controllers, tora_nnet, tora_prob_dir, tora_reach, tora_sat)
println("toradata: $tora_data")

###############################
### Plots
###############################
define_color("symbolic_color", 0x139EAB)
car_style = "solid, thick, symbolic_color, mark=*, mark options={fill=white}"
tora_style = "solid, thick, red, mark=*, mark options={fill=white}"
pend_style = "solid, thick, blue, mark=*, mark options={fill=white}"
acc_style = "solid, thick, orange, mark=*, mark options={fill=white}"

# Plots 1-4: Reach Time vs. # Neurons (Line Graph)
function plot_time(problem_type, x_dim)
    if occursin("Reach", problem_type)
        y_dim = 4
    elseif occursin("Feas", problem_type) || occursin("Sat", problem_type)
        y_dim = 3
    end
    if x_dim == 1
        x_label = "Weights"
    elseif x_dim == 2
        x_label = "Activations"
    end 

    fig = PGFPlots.Axis(style="width=8cm, height=8cm, axis equal image", xlabel="Number of $x_label in NN Controller", ylabel="Time (sec)", title="$problem_type Time", xmode="log", ymode="log")
    push!(fig, PGFPlots.Plots.Linear(single_pendulum_data[x_dim], single_pendulum_data[y_dim], style=pend_style, legendentry="Single Pendulum"))
    push!(fig, PGFPlots.Plots.Linear(car_data[x_dim], car_data[y_dim], style=car_style, legendentry="Car"))
    push!(fig, PGFPlots.Plots.Linear(tora_data[x_dim], tora_data[y_dim], style=tora_style, legendentry="TORA"))
    push!(fig, PGFPlots.Plots.Linear(acc_data[x_dim], acc_data[y_dim], style=acc_style, legendentry="Adaptive Cruise Control"))
    fig.legendStyle = "at={(1.05,1.0)}, anchor=north west"
    PGFPlots.save("examples/jmlr/plots/$(problem_type)_vs_$(x_label).pdf", fig)
    PGFPlots.save("examples/jmlr/plots/$(problem_type)_vs_$(x_label).tex", fig)
end
# x_dim = 1 is vs weights as indep var
plot_time("Reachable Set Computation", 1)
plot_time("Feasibility Computation", 1)
# x_dim = 2 is activations as indep var
plot_time("Reachable Set Computation", 2)
plot_time("Feasibility Computation", 2)

# Plot 5: Sat vs Reach time for each problem (bar graph)
keys = ["S1", "S2", "C1", "C2", "C3", "ACC"]
keys = repeat(keys, outer=2)
sat_row = [single_pendulum_data[3]..., car_data[3]..., acc_data[3]...]
reach_row = [single_pendulum_data[4]..., car_data[4]..., acc_data[4]...]
comp_data = hcat(sat_row, reach_row)
ctg = repeat(["Feasibility", "Reachability"], inner = 6)
g = groupedbar(keys, comp_data, group=ctg, title="Computation Time", ylabel="Time (sec)")
StatPlots.savefig(g, "examples/jmlr/plots/reach_vs_sat_barchart.html")

# separately for TORA:
keys = ["T1", "T2"]
keys = repeat(keys, outer=2)
sat_row = tora_data[3][1:2]
reach_row = tora_data[4][1:2]
comp_data = hcat(sat_row, reach_row)
ctg = repeat(["Feasibility", "Reachability"], inner = 2)
g_tora = groupedbar(keys, comp_data, group=ctg, title="Computation Time", ylabel="Time (sec)")
StatPlots.savefig(g_tora, "examples/jmlr/plots/reach_vs_sat_barchart_tora12.html")

# separately for TORA:
keys = ["T3 Feasibility", "T3 Reachability"]
sat_row = tora_data[3][3]
reach_row = tora_data[4][3]
comp_data = reshape([sat_row, reach_row], 2)
g_tora3 = bar(keys, comp_data, title="Computation Time", ylabel="Time (sec)", yaxis=:log10)
StatPlots.savefig(g_tora3, "examples/jmlr/plots/reach_vs_sat_barchart_tora3.html")

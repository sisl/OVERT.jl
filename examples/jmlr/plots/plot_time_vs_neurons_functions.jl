# plotting time
using JLD2
using FileIO
using PGFPlots
using StatPlots
using NeuralVerification
using NeuralVerification: ReLU, Id, Network, Layer, n_nodes
using Format

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

function get_reach_data(fname)
    data = load(fname)
    if occursin("car", fname)
        holds = data["goal_reached"]
    else # acc, single pend
        holds = data["safe"]
    end
    return ceil(data["dt"] + data["dt_check"]), holds
end

function get_sat_data(fname, n_queries)
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
    holds = occursin("unsat", data["SATus"])
    return ceil(t), holds
end

function get_data(controllers, nnet, prob_dir, reach_s, sat_s)
    weights = Float64[]
    acts = Float64[]
    feas_time = Float64[]
    reach_time = Float64[]
    for c in controllers
        w, a = get_nnet_metadata(format(nnet, c))
        dir = data_dir*prob_dir
        rt, rholds = get_reach_data(dir*format(reach_s, c))
        println("reach_time: $rt, reach holds: $rholds")
        ft, fholds = get_sat_data(dir*format(sat_s, c), [1, 2])
        println("feas_time: $ft, feas holds: $fholds")
        push!(weights, w)
        push!(acts, a)
        push!(feas_time, ft)
        push!(reach_time, rt)
    end
    return [weights, acts, feas_time, reach_time, rholds, fholds]
end

function collect_data()
    # collect all data 
    data_dir = "examples/jmlr/data/"
    ######### acc 
    acc_w, acc_a = get_nnet_metadata("nnet_files/jair/acc_controller.nnet")
    acc_dir = data_dir*"acc/acc"
    acc_reach_time, acc_reach_holds = get_reach_data(acc_dir*"_reachability_data_55.jld2")
    println("acc_reach_time: $acc_reach_time, acc_reach_hold: $(acc_reach_holds)")
    acc_feas_time, acc_feas_holds = get_sat_data(acc_dir*"_satisfiability_data_55", [])
    acc_data = [[acc_w], [acc_a], [acc_feas_time], [acc_reach_time], [acc_reach_holds], [acc_feas_holds]]
    println("acc_feas_time: $acc_feas_time, acc_feas_holds: $(acc_feas_holds)")

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

    return acc_data, car_data, single_pendulum_data, tora_data 
end
#acc_data, car_data, single_pendulum_data, tora_data = collect_data()


###############################
### Plots
###############################
define_color("symbolic_color", 0x139EAB)
car_style = "solid, thick, symbolic_color, mark=*, mark options={fill=white}"
tora_style = "solid, thick, red, mark=*, mark options={fill=white}"
pend_style = "solid, thick, blue, mark=*, mark options={fill=white}"
acc_style = "solid, thick, orange, mark=*, mark options={fill=white}"

# Plots 1-4: Reach Time vs. # Neurons (Line Graph)
function plot_time(problem_type, x_dim; save_fig=false)
    if occursin("Reach", problem_type)
        y_dim = 4
    elseif occursin("Feas", problem_type) || occursin("Sat", problem_type)
        y_dim = 3
    end
    if x_dim == 1
        x_label = "Weights"
    elseif x_dim == 2
        x_label = "ReLU Neurons"
    end 
    #hold_style = Dict(true => )

    fig = PGFPlots.Axis(style="width=15cm, height=8cm, axis equal image", xlabel="Number of $x_label in NN Controller", ylabel="Time (sec)", title="$problem_type Time", xmode="log", ymode="log")
    push!(fig, PGFPlots.Plots.Linear(single_pendulum_data[x_dim], single_pendulum_data[y_dim], style=pend_style, legendentry="Single Pendulum"))
    push!(fig, PGFPlots.Plots.Linear(car_data[x_dim], car_data[y_dim], style=car_style, legendentry="Car"))
    push!(fig, PGFPlots.Plots.Linear(tora_data[x_dim], tora_data[y_dim], style=tora_style, legendentry="TORA"))
    push!(fig, PGFPlots.Plots.Linear(acc_data[x_dim], acc_data[y_dim], style=acc_style, legendentry="Adaptive Cruise Control"))
    fig.legendStyle = "at={(1.05,1.0)}, anchor=north west"
    if save_fig
        save_dried_fig(fig, problem_type, x_label)
    else
        return fig 
    end
end

function save_dried_fig(fig, problem_type, x_label)
    PGFPlots.save("examples/jmlr/plots/$(problem_type)_vs_$(x_label).pdf", fig)
    PGFPlots.save("examples/jmlr/plots/$(problem_type)_vs_$(x_label).tex", fig)
end
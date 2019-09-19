using Flux, LinearAlgebra
using BSON: @save, @load
include("utils.jl")
include("relubypass.jl")

### FUNCTION DEFINITIONS ###
# Create the FFNN from the given RNN model.
function rnn_to_ffnn(model)
    for i = 1:length(model)
        if model[i] isa Flux.Recur
            W, b, R = unroll_rnn_layer(model, i)
            # update model
            new = [[Dense(W[i], b[i], R[i]) for i=1:length(model)-1];  Dense(W[end], b[end], model[end].σ)]
            model = Chain(new...)
        end
    end
    return model
end


function unroll_rnn_layer(model, rnn_index)
    Wh = latent_weights(model[rnn_index])
    W = Vector{Array{Float64}}(undef, length(model))
    B = Vector{Array{Float64}}(undef, length(model))
    R = Vector{Any}(undef, length(model))
    size_latent = latent_size(model[rnn_index])
    for i = 1:length(model)
        if i == rnn_index
            WiWh = [weights(model[i]) latent_weights(model[i])]
            bh = bias(model[i])
            W[i] = [WiWh; WiWh]
            B[i] = [bh; bh]
        else
            W[i] = block_diagonal(weights(model[i]), Matrix(1I, size(Wh)))
            B[i] = [bias(model[i]); zeros(size(Wh)[1])]
        end
        if i < rnn_index
            R[i] = ReLUBypass(layer_size(model[i], 1) .+ (1:size_latent))
        elseif i == rnn_index
            R[i] = model[i].cell.σ
        else
            R[i] = model[i].σ
        end
    end
    return W, B, R
end


function shuffle_layer(in_len, out_vec)
    W = 1.0I(in_len)[out_vec, :]
    b = zeros(length(out_vec))
    return Dense(W, b, identity)
end

### IMPLEMENT ###
controller = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))
@load "controller_weights.bson" weights_c
Flux.loadparams!(controller, weights_c)

@load "dynamics_model.bson" dynamics

ffnn = rnn_to_ffnn(controller)
ffnn_bypassed = add_bypass_variables(ffnn, 2)
preshuffle = shuffle_layer(6, [1,2, 3,4,5,6, 1,2])
postshuffle = shuffle_layer(7, [6,6,7,1,2,3,4,5])
dynamics_bypassed = add_bypass_variables(dynamics, 4)

pre_total = Chain(preshuffle, ffnn_bypassed..., postshuffle, dynamics_bypassed...)
total_network = pre_total |> relu_bypass

@save "closed_loop_controller.bson" total_network

### TESTING ###
@show x0 = rand(2)
init = Float64.(Tracker.data(controller[2].init))
x0_ = [x0; init]
x0__ = [x0_; x0]

_pretty_print(name, val) = println(rpad(name, 30), val)
capture_layers(model, x) = [x = l(x) for l in model]

_pretty_print("RNN Out + latent", [controller(x0); controller[2].state] |> Tracker.data |> Vector{Float64})
_pretty_print("FFNN Equiv", ffnn(x0_))
_pretty_print("FFNN Bypass", ffnn_bypassed(x0__))
_pretty_print("prefinal network", pre_total(x0_))
_pretty_print("Final network", total_network(x0_))
_pretty_print("Known point: [0, 0, init]", total_network([0;0;init]))

let
    x0 = rand(2)
    init = Float64.(Tracker.data(controller[2].init))
    rnn_out = [controller(x0); controller[2].state] |> Tracker.data |> Vector{Float64} 
    ffnn_out = ffnn([x0; init])
    ffnn_bypassed_out = ffnn_bypassed([x0; init; x0])
    prefinal_net = pre_total([x0; init])
    final_net = total_network([x0; init])

    @assert rnn_out == ffnn_out
    @assert ffnn_bypassed_out[1:end-2] == ffnn_out
    @assert ffnn_bypassed_out[end-2:end] == x0
    @assert ffnn_bypassed_out[end-2:end] == prefinal_net[end-2:end]
    @assert prefinal_net == final_net
end

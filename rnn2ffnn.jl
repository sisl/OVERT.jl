using Flux, LinearAlgebra
include("utils.jl")
include("relubypass.jl")

################################################################################
# This is the RNN we want to convert to FFNN.
rnn = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))

# Create the FFNN from the given RNN model. Currently the list "new" is hardcoded
function rnn_to_ffnn(model)
    for i = 1:length(model)
        if model[i] isa Flux.Recur
            W, b = unroll_rnn_layer(model, i)
            # update model
            new = [[Dense(W[i], b[i], relu) for i=1:length(model)-1];  Dense(W[end], b[end], identity)]
            model = Chain(new...)
        end
    end
    return model
end



function unroll_rnn_layer(model, rnn_index)
    Wh  = latent_weights(model[rnn_index])
    W = Vector{Array{Float64}}(undef, length(model))
    B  = Vector{Array{Float64}}(undef, length(model))
    for i = 1:length(model)
        if i == rnn_index
            WiWh = [weights(model[i]) latent_weights(model[i])]
            bh = bias(model[i]) + latent_bias(model[i])
            W[i] = [WiWh; WiWh]
            B[i] = [bh; bh]
        else
            W[i] = block_diagonal(weights(model[i]), Matrix(1I, size(Wh)))
            B[i] = [bias(model[i]); zeros(size(Wh)[1])]
        end
    end
    return W, B
end


# Returns number of layers of a model, including input layer
num_layers(model) = length(model) + 1

# Return list of size of each layer in flux model. Input layer included.
layer_sizes(x) = [layer_size(x[1], 2); layer_size.(x, 1)]


weights(D::Dense) = Tracker.data(D.W)
weights(R::Flux.Recur) = Tracker.data(R.cell.Wi)
bias(D::Dense) = Tracker.data(D.b)
bias(R::Flux.Recur) = Tracker.data(R.cell.b)
latent_weights(l::Flux.Recur) = Tracker.data(l.cell.Wh)
latent_bias(R::Flux.Recur) = Tracker.data(R.cell.h)

layer_size(L, i = nothing) = i == nothing ? size(weights(L)) : size(weights(L), i)

# Return RNN Layer. (Input limited to one RNN layer)
rnn_layer(model) = findfirst(l->l isa Flux.Recur, model.layers)




# ## CHECK CORRECTNESS ##
input = 2rand(2) - 1.0
eval_point = input
state = rand(2)
eval_point2 = vcat(input, zeros(size_latent), state)
long_in = [eval_point; rand(2)]
bypassed = add_bypass_variables(rnn, 2)

ffnn = add_bypass_variables(rnn_to_ffnn(rnn), 2)
ans_old = Tracker.data(rnn(eval_point))
ans_new = Tracker.data(ffnn(eval_point2))
print("\nCheck for same control output and latent state:")
print("\nRNN Out:        ", ans_old[1])
print("\nFFNN Equiv Out: ", ans_new[1])

print("\n\nRNN Latent State:  ", Tracker.data(rnn.layers[2].state))
print("\nFFNN Latent State:        ", ans_new[2:end-2])

print("\n\nInput State:             ", state)
print("\nFFNN Pass Through State: ", ans_new[end-1:end], "\n")

print("\n\nLong In: ", long_in)
print("\nBypassed Check:  ", bypassed(long_in))










## EXTRANEOUS FUNCTIONS ##
# # Creates a network where the input is ReLUBypassed for a given number of layers
# function state_bypass(state_size, layer_count)
#     layer_list = fill(Dense(Matrix(1I, state_size, state_size), zeros(state_size), ReLUBypass(collect(1:state_size)...)), (layer_count))
#     out = relu_bypass(Chain(layer_list...))
# end

## EXTRANEOUS PRINTING OF INPUT RNN ##
# # print weights #
# for layer in model.layers
#     if layer isa Flux.Dense
#         print("\n\nDense Weights:\n")
#         print(Tracker.data(layer.W))
#     elseif layer isa Flux.Recur
#         print("\n\nRNN Weights:\n")
#         print("In:     ", Tracker.data(layer.cell.Wi), "\n")
#         print("Latent: ", Tracker.data(layer.cell.Wh))
#     end
# end
# # print biases #
# for layer in model.layers
#     if layer isa Flux.Dense
#         print("\n\nDense Biases:\n")
#         print(Tracker.data(layer.b))
#     elseif layer isa Flux.Recur
#         print("\n\nRNN Biases:\n")
#         print("In:     ", Tracker.data(layer.cell.b), "\n")
#         print("Latent: ", Tracker.data(layer.cell.h))
#     end
# end
# print("\n\n")

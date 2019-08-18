using Flux, LinearAlgebra
# This file is for prototyping a function for turning an RNN to FFNN #
# The hardcoded list "new" in new_net() restricts to 4 total layers right now.
# A TODO is to not hardcode that list, which is definitely doable, I just
# couldn't figure it out after a few tries.

## 60 Lines of Relu Bypass code. We should find a better way of using this
## rather than copy and pasting to each new file ##
struct ReLUBypass{T}
    protected::T
end
ReLUBypass() = ReLUBypass(nothing)
ReLUBypass(args...) = ReLUBypass(collect(args))
# so that a ReLUBypass can be used as an activation function.
(RB::ReLUBypass{Nothing})(x) = identity(x)
function (RB::ReLUBypass)(x)
    out = relu.(x)
    out[RB.protected] = x[RB.protected]
    return out
end

Base.show(io::IO, RB::ReLUBypass) = print(io, "ReLUBypass($(repr(RB.protected)))")

# Also type piracy:
# TODO to harmonize better with Flux, define a specialized broadcast behavior instead for ReLUBypass
# (D::Dense{<:ReLUBypass, A, B})(x) where {A,B} = D.σ(D.W*x + D.b)
(D::Dense)(x::Number) = D.σ.(D.W*x + D.b)
FluxArr = AbstractArray{<:Union{Float32, Float64}, N} where N
(D::Dense{<:ReLUBypass, <:FluxArr, B})(x::FluxArr) where B = D.σ(D.W*x + D.b)

# If it's anything other than a bypass do nothing
relu_bypass(L1::Dense, L2::Dense) where {A, B} = L1, L2
# if it's a ReLUBypass
function relu_bypass(L1::Dense{<:ReLUBypass, A, B}, L2::Dense) where {A, B}
    W, b, protected = L1.W, L1.b, L1.σ.protected
    if protected == nothing
        protected = collect(axes(b, 1))
    end
    n = size(W, 1)
    I = Matrix(LinearAlgebra.I, n, n)
    # `before` only protects the indices we want, and `after` undoes the transformation
    # `before` needs to be left-multiplied by the weights and `after` right-multiplied
    # I.e. the full thing:   W₂*B'*σ(B*(W₁*x + b₁)) + b₂
    before = [I; -I[protected, :]]
    after = before'

    L1_new = Dense(before*W, before*b, relu)
    L2_new = Dense(L2.W*after, L2.b, L2.σ)
    return L1_new, L2_new
end

function relu_bypass(C::Chain)
    C_bypassed = collect(C)
    for i in 1:length(C)-1
        C_bypassed[i:i+1] .= relu_bypass(C_bypassed[i], C_bypassed[i+1])
    end
    Chain(C_bypassed...)
end

Base.:-(x::AbstractArray, y::Number) = x .- y
Base.:+(x::AbstractArray, y::Number) = x .+ y
Base.:-(x::Number, y::AbstractArray) = x .- y
Base.:+(x::Number, y::AbstractArray) = x .+ y

(D::Dense{<:ReLUBypass, A, B})(x::AbstractArray) where {A, B} = D.σ(D.W*x + D.b)
(D::Dense{<:ReLUBypass, A, <:FluxArr})(x::AbstractArray) where A = D.σ(D.W*x + D.b)
(D::Dense{<:ReLUBypass, <:FluxArr, <:FluxArr})(x::AbstractArray) where A = D.σ(D.W*x + D.b)
################################################################################






################################################################################
# This is the RNN we want to convert to FFNN.
rnn = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))

# Create the FFNN from the given RNN model. Currently the list "new" is hardcoded
function new_net(model, state_size)
    W, b = new_params(model, state_size)
    bypass = bypass_nodes(model, state_size); #bypass = bypass .+ state_size
    # new = [Dense(W[i], b[i], ReLUBypass(bypass[i])) for i=1:num_layers(model)-2]
    # new = append!(new, [Dense(W[end], b[end], identity)])
    # print("\n\nNEW: \n", new, "\n")
    new = [Dense(W[1], b[1], ReLUBypass(bypass[1])), Dense(W[2], b[2], ReLUBypass(bypass[2])), Dense(W[3], b[3], identity)]
    out = relu_bypass(Chain(new...))
end

# Returns weights and biases of new FFNN based on input RNN
function new_params(model, state_size)
    W_s = Matrix(1I, state_size, state_size)
    b_s = zeros(state_size)
    sizes_layers = layer_sizes(model)
    rnn_index = rnn_layer(model)
    size_latent = size(model[rnn_layer(model)].init)[1]
    weights = Vector{Array{Float64}}(undef, num_layers(model)-1)
    biases  = Vector{Array{Float64}}(undef, num_layers(model)-1)
    for i = 1:num_layers(model)-1
        if i < rnn_index
            weights[i] = hcat(zeros(sizes_layers[i+1], size_latent), Tracker.data(model.layers[i].W))
            weights[i] = vcat(Matrix(1I, size_latent, sizes_layers[i]+size_latent), weights[i])
            weights[i] = hcat(zeros(size(weights[i])[1], state_size), weights[i])
            weights[i] = vcat(Matrix(1I, state_size, size(weights[i])[2]), weights[i])
            biases[i]  = vcat(zeros(size_latent+state_size), Tracker.data(model.layers[i].b))
        elseif i == rnn_index
            weights[i] = vcat(hcat(Tracker.data(model.layers[i].cell.Wh), Tracker.data(model.layers[i].cell.Wi)), hcat(Tracker.data(model.layers[i].cell.Wh), Tracker.data(model.layers[i].cell.Wi)))
            weights[i] = hcat(zeros(size(weights[i])[1], state_size), weights[i])
            weights[i] = vcat(Matrix(1I, state_size, size(weights[i])[2]), weights[i])
            biases[i]  = vcat(Tracker.data(model.layers[i].cell.h) + Tracker.data(model.layers[i].cell.b), Tracker.data(model.layers[i].cell.h) + Tracker.data(model.layers[i].cell.b))
            biases[i]  = vcat(zeros(state_size), biases[i])
        elseif i > rnn_index
            weights[i] = hcat(zeros(sizes_layers[i+1], size_latent), Tracker.data(model.layers[i].W))
            weights[i] = vcat(weights[i], Matrix(1I, size_latent, sizes_layers[i]+size_latent))
            weights[i] = hcat(zeros(size(weights[i])[1], state_size), weights[i])
            weights[i] = vcat(Matrix(1I, state_size, size(weights[i])[2]), weights[i])
            biases[i]  = vcat(zeros(state_size), Tracker.data(model.layers[i].b), zeros(size_latent))
        end
    end
    return weights, biases
end

# Returns which indices of each layer to ReLUBypass
function bypass_nodes(model, state_size)
    size_latent = size(model[rnn_layer(model)].init)[1]
    [i == rnn_layer(model) ? collect(1:state_size) : collect(1:size_latent+state_size) for i = 1:num_layers(model)-1]
end

# Returns number of layers of a model, including input layer
num_layers(model) = 1 + sum([1 for l in model.layers])

# Return list of size of each layer in flux model. Input layer included.
layer_sizes(x) = append!([size(x[1].W)[2]], [x.layers[i] isa Dense ? size(x[i].W)[1] : size(x[i].init)[1] for i=1:num_layers(x)-1])

# Return RNN Layer. (Input limited to one RNN layer)
rnn_layer(model) = findfirst(l->l isa Flux.Recur, model.layers)




## CHECK CORRECTNESS ##
size_latent = size(rnn[rnn_layer(rnn)].init)[1]
state = [1.1, -11.5]; state_size = size(state)[1]
input = 2rand(2) - 1.0
eval_point = input
eval_point2 = vcat(state, zeros(size_latent), input)

ffnn = new_net(rnn, state_size)
ans_old = Tracker.data(rnn(eval_point))
ans_new = Tracker.data(ffnn(eval_point2))
print("\nCheck for same control output and latent state:")
print("\nRNN: ", ans_old[1], Tracker.data(rnn.layers[rnn_layer(rnn)].state))
print("\nFFNN: ", ans_new)  # Outputs [theta, theta_dot, u, l1, l2, l3, l4] l1 is first value of latent state
# ans_old[1] == ans_new[1] ? print("\nMatching Control Output✅") : print("\nNot Matching Control Output ❌")
# model.layers[2].state == ans_new[2:end] ? print("\nMatching RNN State ✅") : print("\nNot Matching RNN State ❌")







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

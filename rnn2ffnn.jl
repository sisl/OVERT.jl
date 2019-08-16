using Flux

# This file is for prototyping a function for turning an RNN to FFNN #
model = Chain(Dense(2, 2, relu), RNN(2, 2, relu), Dense(2, 1, relu))

# print weights #
for layer in model.layers
    if layer isa Flux.Dense
        print("\n\nDense Weights:\n")
        print(Tracker.data(layer.W))
    elseif layer isa Flux.Recur
        print("\n\nRNN Weights:\n")
        print("In:     ", Tracker.data(layer.cell.Wi), "\n")
        print("Latent: ", Tracker.data(layer.cell.Wh))
    end
end

# print biases #
for layer in model.layers
    if layer isa Flux.Dense
        print("\n\nDense Biases:\n")
        print(Tracker.data(layer.b))
    elseif layer isa Flux.Recur
        print("\n\nRNN Biases:\n")
        print("In:     ", Tracker.data(layer.cell.b), "\n")
        print("Latent: ", Tracker.data(layer.cell.h))
    end
end
print("\n\n")

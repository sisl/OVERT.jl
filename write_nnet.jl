
"""
Prepend `//` to each line of a string.
"""
to_comment(txt) = "//"*replace(txt, "\n"=>"\n//")

"""
    print_layer(file::IOStream, layer)

Print to `file` an object implementing `weights(layer)` and `bias(layer)`
"""
function print_layer(file::IOStream, layer)
   print_row(W, i) = println(file, join(W[i,:], ", "), ",")
   W = weights(layer)
   b = bias(layer)
   [print_row(W, row) for row in axes(W, 1)]
   [println(file, b[row], ",") for row in axes(W, 1)]
end

"""
    print_header(file::IOStream, model[; header_text])

The NNet format has a particular header containing information about the network size and training data.
`print_header` does not take training-related information into account (subject to change).
"""
function print_header(file::IOStream, model; header_text="")
   println(file, to_comment(header_text))
   # num layers, num inputs, num outputs, max layer size
   layer_sizes = [layer_size(model[1], 2); layer_size.(model, 1)]
   num_layers = length(model)
   num_inputs = layer_sizes[1]
   num_outputs = layer_sizes[end]
   max_layer = maximum(layer_sizes)
   println(file, join([num_layers, num_inputs, num_outputs, max_layer], ", "), ",")
   #layer sizes input, ..., output
   println(file, join(layer_sizes, ", "), ",")
   # empty
   println(file, "This line extraneous")
   # minimum vals of inputs
   println(file, -1e5)
   # maximum vals of inputs
   println(file, 1e5)
   # mean vals of inputs
   println(file, 0)
   # range vals of inputs
   println(file, 1)
   return nothing
end

"""
    write_nnet(filename, model[; header_text])

Write `model` to \$filename.nnet. `model` should be an iterable object containing
layers of a feed-forward, fully connected neural network with ReLU activations.
Note: Does not perform safety checks on inputs, so use with caution.
"""
function write_nnet(outfile, model; header_text="Default header text.\nShould replace with the real deal.")
    name, ext = splitext(outfile, ".")
    outfile = name*".nnet"
    open(outfile, "w") do f
        print_header(f, model, header_text=header_text)
        for layer in model
            print_layer(f, layer)
        end
    end
    nothing
end

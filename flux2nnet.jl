using Flux
include("utils.jl")

# define / load model #
model = Chain(Dense(2, 4, relu), Dense(4, 20, relu))

# print single layer #
function print_layer(file::IOStream, L::Dense)
   print_row(W, i) = println(file, join(W[i,:], ", "), ",\r\n")
   W = weights(L)
   b = bias(L)
   [print_row(W, row) for row in axes(W, 1)]
   [println(file, b[row], ",\r\n") for row in axes(W, 1)]
end

to_comment(txt) = "//"*replace(txt, "\r\n"=>"\r\n//")

function print_header(file::IOStream, C; header_text="")
   println(file, to_comment(header_text), "\r\n")
   # num layers, num inputs, num outputs, max layer size
   layer_sizes = [layer_size(C[1], 2); layer_size.(C, 1)]
   num_layers = length(C)
   num_inputs = layer_sizes[1]
   num_outputs = layer_sizes[end]
   max_layer = maximum(layer_sizes)
   println(file, join([num_layers, num_inputs, num_outputs, max_layer], ", "), ",\r\n")
   #layer sizes input, ..., output
   println(file, join(layer_sizes, ", "), ",\r\n")
   # empty
   println(file, "This line extraneous\r\n")
   # minimum vals of inputs (?)
   # maximum vals of inputs (?)
   # mean vals of inputs (?)
   # range vals of inputs (?)
   [println(file, "0\r\n") for i in 1:4]
   return nothing
end

function write_nnet(outfile, model)
   open(outfile, "w") do f
      print_header(f, model, header_text="DuBois. \r\nMy name is Joe.")
      [print_layer(f, layer) for layer in model]
   end
end

using Flux
include("utils.jl")

const NEWLINE = Sys.iswindows() ? "\r\n" : "\n"

to_comment(txt) = "//"*replace(txt, NEWLINE=>NEWLINE*"//")

# define / load model #
model = Chain(Dense(2, 4, relu), Dense(4, 20, relu))

# print single layer #
function print_layer(file::IOStream, L::Dense)
   print_row(W, i) = println(file, join(W[i,:], ", "), ",$NEWLINE")
   W = weights(L)
   b = bias(L)
   [print_row(W, row) for row in axes(W, 1)]
   [println(file, b[row], ",$NEWLINE") for row in axes(W, 1)]
end


function print_header(file::IOStream, C; header_text="")
   println(file, to_comment(header_text), NEWLINE)
   # num layers, num inputs, num outputs, max layer size
   layer_sizes = [layer_size(C[1], 2); layer_size.(C, 1)]
   num_layers = length(C)
   num_inputs = layer_sizes[1]
   num_outputs = layer_sizes[end]
   max_layer = maximum(layer_sizes)
   println(file, join([num_layers, num_inputs, num_outputs, max_layer], ", "), ",$NEWLINE")
   #layer sizes input, ..., output
   println(file, join(layer_sizes, ", "), ",$NEWLINE")
   # empty
   println(file, "This line extraneous$NEWLINE")
   # minimum vals of inputs (?)
   # maximum vals of inputs (?)
   # mean vals of inputs (?)
   # range vals of inputs (?)
   [println(file, "0$NEWLINE") for i in 1:4]
   return nothing
end

function write_nnet(outfile, model; header_text = "Default header text.")
   open(outfile, "w") do f
      print_header(f, model, header_text=header_text)
      [print_layer(f, layer) for layer in model]
   end
end

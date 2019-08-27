using Flux
include("utils.jl")

"""
Used to add additional newlines that are necessary on Windows OS only.
On windows, represents the newline tag `\\r\\n`. On unix, `NEWLINE` is an empty string.
"""
const NEWLINE = Sys.iswindows() ? "\r\n" : ""

function to_comment(txt)
    if Sys.iswindows()
        return "//"*replace(txt, "\r\n"=>"\r\n//")
    else
        return "//"*replace(txt, "\n"=>"\n//")
    end
end

# define / load model #
model = Chain(Dense(2, 4, relu), Dense(4, 20, relu))

# print single layer #
function print_layer(file::IOStream, layer)
   print_row(W, i) = println(file, join(W[i,:], ", "), ",$NEWLINE")
   W = weights(layer)
   b = bias(layer)
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
   println(file, -1e10, NEWLINE)
   # maximum vals of inputs (?)
   println(file, 1e10, NEWLINE)
   # mean vals of inputs (?)
   println(file, 0, NEWLINE)
   # range vals of inputs (?)
   println(file, 1, NEWLINE)
   return nothing
end

function write_nnet(outfile, model; header_text = "Default header text.\nShould replace with the real deal.")
   open(outfile, "w") do f
      print_header(f, model, header_text=header_text)
      [print_layer(f, layer) for layer in model]
   end
   nothing
end

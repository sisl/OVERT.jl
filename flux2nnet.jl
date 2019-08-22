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

# ### UNTESTED FFNN VERSION ###
# list = [2, 4]
# weights = Tracker.data.(params(model))
# outfile = "NN_NNet.txt"
# open(outfile, "w") do f
#    layer = 1
#    i = 1
#    n = 1
#    while i <= size(weights)[1]
#       for row = 1:list[layer+1]
#          for col = 1:list[layer]
#             println(f, weights[i][row,:][col])
#             println(f, ",")
#          end
#          println(f, "\r\n")
#       end
#       i += 1
#
#       # prints biases
#       for row = 1:list[layer+1]
#          println(f, weights[i][row])
#          println(f, ",\r\n")
#       end
#       i += 1
#       layer += 1
#    end
# end




#############################################
# ############RNN VERSION ##################
# list = [2 4 4 1]  # size of each layer
# rnn_layer = 3  # what indices have rnn layers
# weights = Tracker.data.(params(model))
# outfile = "RNN_NNet.txt"
# open(outfile, "w") do f
#    layer = 1
#    i = 1
#    n = 1
#    while i <= size(weights)[1]
#       layer+1 == rnn_layer ? n=2 : n=1
#
#       # prints weights. Will need to change for general rnn dim
#       for _ = 1:n
#          for row = 1:list[layer+1]
#             for col = 1:list[layer]
#                println(f, weights[i][row,:][col])
#                println(f, ",")
#             end
#             println(f, "\r\n")
#          end
#          i += 1
#       end
#
#       # prints biases
#       for _ = 1:n
#          for row = 1:list[layer+1]
#             println(f, weights[i][row])
#             println(f, ",\r\n")
#          end
#          i += 1
#       end
#
#       layer += 1
#    end
# end

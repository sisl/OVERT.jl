using Flux

### UNTESTED FFNN VERSION ###
weights = Tracker.data.(params(model))
outfile = "NN_NNet.txt"
open(outfile, "w") do f
   layer = 1
   i = 1
   n = 1
   while i <= size(weights)[1]
      for row = 1:list[layer+1]
         for col = 1:list[layer]
            println(f, weights[i][row,:][col])
            println(f, ",")
         end
         println(f, "\r\n")
      end
      i += 1

      # prints biases
      for row = 1:list[layer+1]
         println(f, weights[i][row])
         println(f, ",\r\n")
      end
      i += 1
      layer += 1
   end
end




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

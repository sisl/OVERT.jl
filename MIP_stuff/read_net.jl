using JuMP
using MathProgBase.SolverInterface
using GLPK
using LazySets
using Parameters
using Interpolations
using Gurobi

NEURAL_VERIFICATION_PATH = "/home/amaleki/Downloads/NeuralVerification.jl"

include("$NEURAL_VERIFICATION_PATH/src/utils/activation.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/network.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/problem.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/util.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/constraints.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/objectives.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/variables.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/mipVerify.jl")
include("$NEURAL_VERIFICATION_PATH/src/reachability/maxSens.jl")


function add_controller_constraints(model, network_nnet_address, input_set, input_vars, output_vars; last_layer_activation=Id())

    # inputSet = Hyperrectangle([0.0], [.5])
    # optimizer = Gurobi.Optimizer
    # model = Model(with_optimizer(optimizer))
    network = read_nnet(network_nnet_address, last_layer_activation=last_layer_activation)
    neurons = init_neurons(model, network)
    deltas = init_deltas(model, network)
    bounds = get_bounds(network, input_set)
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    @constraint(model, input_vars .== neurons[1])  # set inputvars
    @constraint(model, output_vars .== neurons[end])  # set outputvars

    return bounds[end]
end

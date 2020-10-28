# acc satisfiability script
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/acc/acc.jl")
using JLD2

controller_name = ""
# OH NO I DONT HAVE THE ACC CONTROLLER
# 3x20 controller here: https://github.com/souradeep-111/sherlock/tree/master/systems_with_networks/ARCH_2019/ACC
controller = "nnet_files/jair/acc_"*controller_name*"_controller.nnet"
println("Controller is: ", controller)
query = OvertQuery(
    ACC,  # problem
    controller,    # network file
    Id(),      	# last layer activation layer Id()=linear, or ReLU()=relu
    "MIP",     	# query solver, "MIP" or "ReluPlex"
    25,        	# ntime
    0.2,       	# dt
    -1,        	# N_overt
    )
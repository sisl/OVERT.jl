# acc reachability script
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/acc/acc.jl")
include("../../MIP/src/logic.jl")
using JLD2

controller = "acc_controller"
controller_filepath = "nnet_files/jair/"*controller*".nnet"
println("Controller is: ", controller)
query = OvertQuery(
    ACC,  # problem
    controller_filepath,    # network file
    Id(),      	# last layer activation layer Id()=linear, or ReLU()=relu
    "MIP",     	# query solver, "MIP" or "ReluPlex"
    35,        	# ntime
    0.1,       	# dt
    -1,        	# N_overt
    )

# x1,x2,x3 are lead vehicle variables
# x4,x5,x6 are ego vehicle variables
x_lead = [90,110]
v_lead = [32, 32.2]
gamma_lead = [0,0]
x_ego = [10,11]
v_ego = [30, 30.2]
gamma_ego = [0, 0]
var_list = [x_lead, v_lead, gamma_lead, x_ego, v_ego, gamma_ego]
input_set = Hyperrectangle(
    low=[v_range[1] for v_range in var_list], 
    high=[v_range[2] for v_range in var_list]
    )

t1 = Dates.time()
all_sets, all_sets_symbolic = symbolic_reachability_with_concretization(query, input_set,1)
# [20, 35])
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

JLD2.@save "examples/jair/data/new2/acc_reachability_"*string(controller)*"_data.jld2" query input_set all_sets all_sets_symbolic dt controller 

# TODO: Intersect all sets with output constraint and see where
# reachable set is fully within safe set
T_gap = 1.4
safe_set = Constraint([1, 0, 0, -1, -T_gap, 0], :(>=), 10)

 
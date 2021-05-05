# acc reachability script
# ENV["JULIA_DEBUG"] = Main # for debugging
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
    55,        	# ntime
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

concretization_intervals = Int.(ones(query.ntime))
t1 = Dates.time()
concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability_with_concretization(query, input_set, concretization_intervals)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

# TODO: Intersect all sets with output constraint and see if
# reachable set is fully within safe set OR check to see if it ever intersects unsafe set
# they are equivalent
# We want the measurement to be greater than 10, always. So the unsafe set if <= 10
avoid_sets = [HalfSpace([1.], 10.)] # 1*y <= 10

reachable_meas_sets = clean_up_meas_sets(concrete_meas_sets, symbolic_meas_sets, concretization_intervals)

t1 = time()
safe, violations = check_avoid_set_intersection(reachable_meas_sets, input_set, avoid_sets)
dt_check = time() - t1

 
JLD2.@save "examples/jmlr/data/acc_reachability_data_1step_55.jld2" query input_set concretization_intervals concrete_state_sets concrete_meas_sets symbolic_state_sets symbolic_meas_sets dt controller avoid_sets reachable_meas_sets safe violations dt_check

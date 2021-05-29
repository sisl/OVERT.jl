include("../../../models/problems.jl")
include("../../../OverApprox/src/overapprox_nd_relational.jl")
include("../../../OverApprox/src/overt_parser.jl")
include("../../../MIP/src/overt_to_mip.jl")
include("../../../MIP/src/mip_utils.jl")
include("../../../models/single_pendulum/single_pend.jl")

controller_type = "small"
controller = "nnet_files/jair/single_pendulum_$(controller_type)_controller.nnet"
println("Controller: ", controller)
query = OvertQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	10,                # ntime
	0.1,               # dt
	-1,                # N_overt
	)

input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
t1 = Dates.time()
concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability(query, input_set)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

# we want to check the intersection with the avoid set: x_1 <= -.2167
avoid_set = [HalfSpace([1., 0.], -0.2617)] # 1*x_1 + 0*x_2 <= -.2167  -->  x_1 <= -.2167 

init_set0, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

t1 = time()
safe, violations = check_avoid_set_intersection(reachable_state_sets, input_set, avoid_set)
dt_check = time() - t1

using JLD2
JLD2.@save "examples/jmlr/data/comparison_single_pendulum_reachability_$(controller_type)_controller_data.jld2" query input_set concrete_state_sets concrete_meas_sets symbolic_state_sets symbolic_meas_sets dt controller avoid_set reachable_state_sets safe violations dt_check
 
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

controller = "nnet_files/jair/tora_smallest_controller.nnet"

query = OvertQuery(
	Tora,       # problem
	controller, # network file
	Id(),       # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",      # query solver, "MIP" or "ReluPlex"
	15,         # ntime
	0.1,        # dt
	-1,         # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
t1 = Dates.time()
concretization_intervals = [5, 5, 5]
concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability_with_concretization(query, input_set, concretization_intervals)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

# avoid set 1
avoid_set1 = HalfSpace([1., 0., 0., 0.], -2.) # x1 <= -2
avoid_set2 = HalfSpace([-1., 0., 0., 0.], -2.) # -x1 <= -2  -> 2 <= x1 
avoid_sets = [avoid_set1, avoid_set2]

# first clean up sets so it looks like: c_t1, ..., s_t10, c_t11, s_t12
init_set0, reachable_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)
# dims argument is just for debugging

# make sure that reachable set does NOT intersect with either avoid set at any point in time
t1 = time()
safe, violations = check_avoid_set_intersection(reachable_sets, input_set, avoid_sets)
dt_check = time() - t1

using JLD2
JLD2.@save "examples/jmlr/data/tora_reachability_smallest_controller_data.jld2" query input_set concretization_intervals concrete_state_sets symbolic_state_sets dt controller avoid_sets reachable_sets safe violations dt_check



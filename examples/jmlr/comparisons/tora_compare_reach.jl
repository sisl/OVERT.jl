include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

controller_name = "smallest" # e.g. "big"
controller = "nnet_files/jair/tora_$(controller_name)_controller.nnet"
println("controller is: $controller")

query = OvertQuery(
	Tora,      # problem
	controller, # network file
	Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     # query solver, "MIP" or "ReluPlex"
	10,        # ntime
	0.1,       # dt
	-1,        # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
t1 = Dates.time()
concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability(query, input_set)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

# clean up sets 
init_set0, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

# we want to check inclusion in the safe set:
constraint1 = HalfSpace([1., 0., 0., 0.], 2.) # x1 <= 2
constraint2 = HalfSpace([-1., 0., 0., 0.], 2.) # -x1 <= 2 aka x1 >= -2
safe_set = HPolyhedron([constraint1, constraint2])
t1 = time()
safe_steps = reachable_state_sets .⊆ Ref(safe_set)
violations = .!safe_steps
safe = all(safe_steps)
dt_check = time() - t1

using JLD2
JLD2.@save "examples/jmlr/data/comparison_tora_reachability_$(controller_name)_controller_data.jld2" query input_set safe_set concrete_state_sets symbolic_state_sets concrete_meas_sets symbolic_meas_sets reachable_state_sets dt safe violations dt_check concretization_intervals

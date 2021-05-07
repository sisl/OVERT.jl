include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/car/simple_car.jl")
using JLD2
using LazySets
ENV["JULIA_DEBUG"] = Main

controller_name = ARGS[1]
controller = "nnet_files/jair/car_"*controller_name*"_controller.nnet"
println("Controller is: ", controller)

query = OvertQuery(
	SimpleCar,  # problem
	controller,    # network file
	Id(),      	# last layer activation layer Id()=linear
	"MIP",     	# query solver, "MIP" or "ReluPlex"
	10,        	# ntime
	0.2,       	# dt
	-1,        	# N_overt
	)

input_set = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5], high=[9.55, -4.45, 2.11, 1.51])
concretization_intervals = [5, 5]
t1 = Dates.time()
concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability_with_concretization(query, input_set, concretization_intervals)
t2 = Dates.time()
dt = (t2-t1)
print("elapsed time= $(dt) seconds")

# clean up sets
init_set0, reachable_state_sets = clean_up_sets(concrete_state_sets, symbolic_state_sets, concretization_intervals)

# In this example, our property is the following:
# We want the car to reach the box [-.6, .6] [-.2,.2]
# at SOME point in the time history
# constraint 1  x1 >= -0.6  --> -x1 <= 0.6
c1 = HalfSpace([-1.0, 0.0, 0.0, 0.0], 0.6)
# constraint 2 x1 <= 0.6
c2 = HalfSpace([1.0, 0.0, 0.0, 0.0], 0.6)
# constraint 3  x2 >= -0.2   -> -x2 <= 0.2
c3 = HalfSpace([-1.0, 0.0, 0.0, 0.0], 0.2) 
# constraint 4  x2 <= 0.2
c4 = HalfSpace([1.0, 0.0, 0.0, 0.0], 0.2)
goal_set = HPolyhedron([c1, c2, c3, c4])
t1 = time()
goal_reached_steps = reachable_state_sets .⊆ Ref(goal_set)
goal_reached = any(goal_reached_steps)
dt_check = time() - t1
println("Goal reached: $goal_reached")
println("Goal reached at step: $(findfirst(goal_reached_steps))")

JLD2.@save "examples/jmlr/data/car_reachability_"*string(controller_name)*"_controller_data.jld2" query input_set concretization_intervals goal_set concrete_state_sets symbolic_state_sets concrete_meas_sets symbolic_meas_sets reachable_state_sets dt goal_reached goal_reached_steps dt_check 

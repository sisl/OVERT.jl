include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

controller = "nnet_files/jair/tora_smallest_controller.nnet"

query = OvertQuery(
	Tora,      # problem
	controller, # network file
	Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     # query solver, "MIP" or "ReluPlex"
	25,        # ntime
	0.1,       # dt
	-1,        # N_overt
	)

# query 1
input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
avoid_set = Hyperrectangle(low=[-Inf, Inf, Inf, Inf], high=[-2, Inf, Inf, Inf]) # checks if x1 is in [-Inf, -2] 
# the other Infs are just to denote that these constraints should be ignored. So essentially the only constraint being added is:
# x1 <= -2
t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set)
t2 = Dates.time()
dt = (t2-t1)

using JLD2
JLD2.@save "examples/jair/data/new/tora_satisfiability_smallest_controller_data_q1.jld2" query input_set target_set SATus vals stats dt

# query 2
input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
avoid_set = Hyperrectangle(low=[2, Inf, Inf, Inf], high=[Inf, Inf, Inf, Inf]) # checks if x1 is in [2, Inf] 
# only constraint actually being added is: x1 >= 2
t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set)
t2 = Dates.time()
dt = (t2-t1)

using JLD2
JLD2.@save "examples/jair/data/new/tora_satisfiability_smallest_controller_data_q2.jld2" query input_set target_set SATus vals stats dt

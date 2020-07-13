include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/single_pendulum/single_pend.jl")

controller = "nnet_files/jair/single_pendulum_small_controller.nnet"

query = OvertQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	40,                # ntime
	0.05,              # dt
	-1,                # N_overt
	)

input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
target_set = InfiniteHyperrectangle([1, -Inf], [Inf, Inf])
t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set; after_n=10)
t2 = Dates.time()
dt = (t2-t1)

using JLD2
JLD2.@save "examples/benchmark/single_pendulum_satisfiability_small_controller_data.jld2" query input_set target_set SATus vals stats dt

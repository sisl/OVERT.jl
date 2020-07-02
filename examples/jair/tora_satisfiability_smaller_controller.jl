include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/tora/tora.jl")

controller = "nnet_files/jair/tora_smaller_controller.nnet"

query = OvertQuery(
	Tora,      # problem
	controller, # network file
	Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",     # query solver, "MIP" or "ReluPlex"
	25,        # ntime
	0.1,       # dt
	-1,        # N_overt
	)

input_set = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
target_set = Hyperrectangle([-1., 0.1, -0.1, -0.6], [0.3, 0.3, 0.3, 0.3])
t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, target_set)
t2 = Dates.time()
dt = (t2-t1)

using JLD2
JLD2.@save "examples/jair/data/tora_satisfiability_smaller_controller_data.jld2" query input_set target_set SATus vals stats dt

# xvec = [[stats[Meta.parse("x$(i)_$(j)")] for j = 1:21] for i=1:4]
# xvec = hcat(xvec...)
# xvec_mc = monte_carlo_one_simulate(query, xvec[1, :])
# Plots.plot(xvec[:,1], xvec[:,2], linecolor=:red)
# Plots.plot!(xvec_mc[:,1], xvec_mc[:,2], linecolor=:black)

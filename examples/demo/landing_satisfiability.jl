# landing problem
include("../../models/problems.jl")
include("../../OverApprox/src/overapprox_nd_relational.jl")
include("../../OverApprox/src/overt_parser.jl")
include("../../MIP/src/overt_to_mip.jl")
include("../../MIP/src/mip_utils.jl")
include("../../models/landing/landing.jl")
using JLD2

controller = "nnet_files/jan_demo/landingNNpolicy.nnet"

println("Controller: ", controller)

query = OvertQuery(
    Landing,      # problem
    controller, # network file
    Id(),    # last layer activation layer Id()=linear, or ReLU()=relu
    "MIP",     # query solver, "MIP" or "ReluPlex"
    45,        # ntime
    1,       # dt
    -1,        # N_overt
    )

#xc,vc,yp,vp
input_set = Hyperrectangle(low=[500,-15,-5, 99],
                            high=[550, -14, 5, 100])
low_avoid=[400, -Inf, -Inf, -Inf]
high_avoid=[600, Inf, Inf, landing_v_thresh]
avoid_set = InfiniteHyperrectangle(low_avoid,high_avoid)
t1 = Dates.time()
SATus, vals, stats = symbolic_satisfiability(query, input_set, avoid_set)
t2 = Dates.time()
elapsed = (t2-t1)
print("elapsed time = $(elapsed) seconds")
JLD2.@save "landing_satisfiability_data.jld2" query input_set avoid_set SATus vals stats elapsed controller 
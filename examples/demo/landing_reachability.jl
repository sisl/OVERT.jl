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
input_set = Hyperrectangle(low=[700,-15,-5, 99],
                            high=[750, -14, 5, 100])
concretize_at = [10, 10, 10, 8, 7]
t1 = Dates.time()
all_sets, all_sets_symbolic = symbolic_reachability_with_concretization(query, input_set, concretize_at)
t2 = Dates.time()
elapsed = (t2-t1)
print("elapsed time = $(elapsed) seconds")
JLD2.@save "landing_reachability_1_data.jld2" query input_set all_sets all_sets_symbolic elapsed 
# checking the soundness of overapproximations for the benchmark problems

include("../models/acc/acc.jl")
include("../OverApprox/src/overapprox_nd_relational.jl")
include("soundness.jl")

# Checking acc
# In ACC, the derivaties of dimensions 3 and 6 are overapproximated
x_lead = [90,110]
v_lead = [32, 32.2]
gamma_lead = [0,0]
x_ego = [10,11]
v_ego = [30, 30.2]
gamma_ego = [0, 0]
input_domains = [x_lead, v_lead, gamma_lead, x_ego, v_ego, gamma_ego]
domain = Dict(zip(acc_input_vars, input_domains))
# construct overapproximation
dim3_OA = overapprox_nd(acc_ẋ₃, domain, N=-1)
# construct soundness query
dim3_SQ = SoundnessQuery(dim3_OA, domain)
# check soundness query with dreal
result = check('dreal', dim3_SQ, "acc_dim3_soundness_query")
println("result is: ", result)
# checking the soundness of overapproximations for the benchmark problems

include("../models/problems.jl")
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
dim3_OA = overapprox_nd(acc_ẋ₃, domain, N=-1, ϵ=.1)
# construct soundness query
##################### MODE 1: PUT ALL OA RELATIONS INTO ϕ̂
dim3_SQ_mode1 = SoundnessQuery([:($k==$v) for (k,v) in dim3_OA.fun_eq], # ϕ
                         dim3_OA.approx_eq, # definitions for ϕ̂
                         dim3_OA.approx_ineq, # relations for OA, ϕ̂
                         domain
)
##################### MODE 2: PUT ONLY LAST OA EXPR INTO ϕ̂, everything prior goes into definitions
ϕ̂_m2 = [r for r in [dim3_OA.approx_eq..., dim3_OA.approx_ineq...] if dim3_OA.output ∈ find_variables(r)]
defs_m2 = [r for r in [dim3_OA.approx_eq..., dim3_OA.approx_ineq...] if dim3_OA.output ∉ find_variables(r)]
dim3_SQ_mode2 = SoundnessQuery([:($(dim3_OA.output) == $acc_ẋ₃)], # ϕ
                                defs_m2,
                                ϕ̂_m2,
                                domain)
# println("soundness query: ", dim3_SQ)
# check soundness query with dreal
result_m1 = check("dreal", dim3_SQ_mode1, "acc_dim3_soundness_query_m1.smt2")
println("result mode 1 is: ", result_m1)
#
result_m2 = check("dreal", dim3_SQ_mode2, "acc_dim3_soundness_query_m2.smt2")
println("result mode 2 is: ", result_m2)
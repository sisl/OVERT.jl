

include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
using LazySets

N_OVERT = 3
range_dict = Dict(:T1 => [-0.52, 0.83], :T2 => [-0.38, 0.96], :th1 => [-1., 1.],
                  :th2 => [-1., 1.], :u1 => [-2., 2.], :u2 => [-2., 2.])

v1 = :(sin(th1))
v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))

v2 = :(sin(th2))
v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))

v3 = :(sin(th1-th2))
v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

v4 = :(cos(th1-th2))
v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

v5 = :($(v3_oA.output)*u1^2)
v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))

v6 = :($(v3_oA.output)*u2^2)
v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))

v7 = :(sin(th1-2*th2))
v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))

v8 = :(($(v7_oA.output) - $(v6_oA.output) + 8*T1 + 3*$(v1_oA.output) -$(v4_oA.output)*(8*T2 + $(v5_oA.output)))/(2-$(v4_oA.output)^2))
v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)

v9 = :((2*$(v5_oA.output) + 16*T2 + 4*$(v2_oA.output) -$(v4_oA.output)*(8*T1 - $(v6_oA.output) + 4*$(v1_oA.output)))/(2-$(v4_oA.output)^2))
v9_oA = overapprox_nd(v9, range_dict; N=N_OVERT)

oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA, v9_oA])
mip_model = OvertMIP(oA_out)

"""
# adding controller
"""
# controller_expr1 = :(T1== th1*0.5 + u1*0.25)
# controller_expr2 = :(T2== th2*0.5 + u2*0.25)
# eq_2_mip(controller_expr1, mip_model)
# eq_2_mip(controller_expr2, mip_model)

input_set = Hyperrectangle(low =  [range_dict[:th1][1], range_dict[:th2][1], range_dict[:u1][1], range_dict[:u2][1]],
                           high = [range_dict[:th1][2], range_dict[:th2][2], range_dict[:u1][2], range_dict[:u2][2]])
control_input_vars = [get_mip_var(v, mip_model) for v in [:th1, :th2, :u1, :u2]]
control_output_vars = [get_mip_var(v, mip_model) for v in [:T1, :T2]]
network_file = "MIP_stuff/controller_complex_double_pend.nnet"
controller_bound = add_controller_constraints(mip_model.model, network_file,
                                                                input_set,
                                                                control_input_vars,
                                                                control_output_vars,
                                                                #last_layer_activation= ReLU()
                                                                )
@assert controller_bound.center[1] - controller_bound.radius[1] >= range_dict[:T1][1]
@assert controller_bound.center[2] - controller_bound.radius[2] >= range_dict[:T2][1]
@assert controller_bound.center[1] + controller_bound.radius[1] <= range_dict[:T1][2]
@assert controller_bound.center[2] + controller_bound.radius[2] <= range_dict[:T2][2]


# additing next_state
dt = 0.1
du1 = v8_oA.output
du2 = v9_oA.output
integration_map = Dict(:th1 => :u1, :th2 => :u2, :u1 => du1, :u2 => du2)

using Dates
println("begins at: ", Dates.format(now(), "HH:MM"))
for (v, dv) in pairs(integration_map)
    next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]

    @objective(mip_model.model, Min, next_v)
    JuMP.optimize!(mip_model.model)
    min_next_v = objective_value(mip_model.model)
    @objective(mip_model.model, Max, next_v)
    JuMP.optimize!(mip_model.model)
    max_next_v = objective_value(mip_model.model)

    println("next $v, min= $min_next_v, max=$max_next_v")
end

println("ends at: ", Dates.format(now(), "HH:MM"))

#
# next_u1 = mip_model.vars_dict[:u1] + dt*mip_model.vars_dict[du1]
# next_u2 = mip_model.vars_dict[:u2] + dt*mip_model.vars_dict[du2]
#
# @objective(mip_model.model, Min, next_th1)
# optimize!(mip_model.model)
# min_next_th1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_th1)
# optimize!(mip_model.model)
# max_next_th1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_th2)
# optimize!(mip_model.model)
# min_next_th2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_th2)
# optimize!(mip_model.model)
# max_next_th2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_u1)
# optimize!(mip_model.model)
# min_next_u1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_u1)
# optimize!(mip_model.model)
# max_next_u1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_u2)
# optimize!(mip_model.model)
# min_next_u2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_u2)
# optimize!(mip_model.model)
# max_next_u2 = objective_value(mip_model.model)

# we can set range of T to whatever values, because for single pendulum
# it does not affect OVERT.
# range of T has to be wide enough to contain all possible neural network output.

range_dict = Dict(:T => [-Inf, Inf], :th => [-1., 1.], :dth => [-2., 2.])

v1 = :(T + sin(th) - 0.2*dth)
v1_oA = overapprox_nd(v1, range_dict; N=2)
mip_model = OvertMIP(v1_oA)

"""
# adding controller
"""
# controller_expr = :(T== th*0.5 + dth*0.25)
# eq_2_mip(controller_expr, mip_model)
input_set = Hyperrectangle(low =  [range_dict[:th][1], range_dict[:dth][1]],
                           high = [range_dict[:th][2], range_dict[:dth][2]])
control_input_vars = [get_mip_var(v, mip_model) for v in [:th, :dth]]
control_output_vars = [get_mip_var(v, mip_model) for v in [:T]]
network_file = "MIP_stuff/controller_simple_single_pend.nnet"
controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, control_input_vars, control_output_vars)
@assert controller_bound.center[1] - controller_bound.radius[1] > range_dict[:T][1]
@assert controller_bound.center[1] + controller_bound.radius[1] < range_dict[:T][2]



# additing next_state
ddth = mip_model.overt_app.output
integration_map = Dict(:th => :dth, :dth => ddth)


dt = 0.1
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

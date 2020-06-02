include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")

using LazySets
using Dates

function horizontalCAS_dynamics(state, u_own, u_int, v_own, v_int)
    x, y, ψ = state
    xprime_own = v_own * sin(u_own) / u_own
    yprime_own = v_own * (1 - cos(u_own)) / u_own
    xprime_int = x + v_int * (sin(ψ + u_int) - sin(ψ)) / u_int
    yprime_int = y + v_int * (cos(ψ) - cos(ψ + u_int)) / u_int
    xprime_old = xprime_int - xprime_own
    yprime_old = yprime_int - yprime_own

    xnew = xprime_old * cos(u_own) + yprime_old * sin(u_own)
    ynew = yprime_old * cos(u_own) - xprime_old * sin(u_own)
    ψnew = ψ + u_int - u_own
    return [xnew, ynew, ψnew]
end

function horizontalCAS_dynamics_overt(range_dict, N_overt, v_own=200., v_int=185.)
    #xprime_old = :(x + $(v_int) * (sin(ψ + u_int) - sin(ψ)) / u_int - $(v_own) * sin(u_own) / u_own)
    xprime_old = :(x + $(v_int)* (cos(ψ) - 0.5 * u_int * sin(ψ)) - $(v_own) * sin(u_own) / u_own)
    xprime_old_oA = overapprox_nd(xprime_old, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(xprime_old_oA.output => xprime_old_oA.output_range))

    #yprime_old = :(y + $(v_int) * (cos(ψ) - cos(ψ + u_int)) / u_int - $(v_own) * (1 - cos(u_own)) / u_own)
    yprime_old = :(y + $(v_int)* (sin(ψ) + 0.5 * u_int * cos(ψ)) - $(v_own) * (1 - cos(u_own)) / u_own)
    yprime_old_oA = overapprox_nd(yprime_old, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(yprime_old_oA.output => yprime_old_oA.output_range))

    cos_u_own = :(cos(u_own))
    cos_u_own_oA = overapprox_nd(cos_u_own, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(cos_u_own_oA.output => cos_u_own_oA.output_range))

    sin_u_own = :(sin(u_own))
    sin_u_own_oA = overapprox_nd(sin_u_own, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(sin_u_own_oA.output => sin_u_own_oA.output_range))

    xnew = :($(xprime_old_oA.output) * $(cos_u_own_oA.output) + $(yprime_old_oA.output) * $(sin_u_own_oA.output))
    xnew_oA = overapprox_nd(xnew, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(xnew_oA.output => xnew_oA.output_range))

    ynew = :($(yprime_old_oA.output) * $(cos_u_own_oA.output) - $(xprime_old_oA.output) * $(sin_u_own_oA.output))
    ynew_oA = overapprox_nd(ynew, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(ynew_oA.output => ynew_oA.output_range))

    ψnew = :(ψ + u_int - u_own)
    ψnew_oA = overapprox_nd(ψnew, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(ψnew_oA.output => ψnew_oA.output_range))

    oA_out = add_overapproximate([xprime_old_oA, yprime_old_oA, cos_u_own_oA,
                                  sin_u_own_oA, xnew_oA, ynew_oA, ψnew_oA])

    output_var_dict = Dict(:x          => :x,
                           :y          => :y,
                           :ψ          => :ψ,
                           :u_int      => :u_int,
                           :u_own      => :u_own,
                           :xprime_old => xprime_old_oA.output,
                           :yprime_old => yprime_old_oA.output,
                           :cos_u_own  => cos_u_own_oA.output,
                           :sin_u_own  => sin_u_own_oA.output,
                           :xnew       => xnew_oA.output,
                           :ynew       => ynew_oA.output,
                           :ψnew       => ψnew_oA.output,)
    return oA_out, output_var_dict
end

N_OVERT = 6
range_dict = Dict(:x => [2000., 3000.], :y => [2000., 3000.], :ψ => [-π, π],
                  :u_int => [-0.02, 0.02], :u_own => [-0.1, 0.1])
input_set = Hyperrectangle(low=[2000., 2000., -π], high=[3000., 3000., π])

oA, output_var_dict = horizontalCAS_dynamics_overt(range_dict, N_OVERT)
mip_model = OvertMIP(oA)

network_nnet_address = "acasx_net1.nnet"
network = read_nnet(network_nnet_address, last_layer_activation=Id())
neurons = init_neurons(mip_model.model, network)
deltas = init_deltas(mip_model.model, network)
bounds = get_bounds(network, input_set)

n_controller_output = length(neurons[end])
mip_control_input_vars  = [get_mip_var(v, mip_model) for v in [:x, :y, :ψ]]
#mip_control_output_vars = [get_mip_aux_var(mip_model) for i = 1:n_controller_output]

encode_network!(mip_model.model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
@constraint(mip_model.model, mip_control_input_vars .== neurons[1])  # set inputvars
#@constraint(mip_model.model, mip_control_output_vars .== neurons[end])  # set outputvars

u_own_var = get_mip_var(:u_own, mip_model)
u_own_vals = [0., 1.5, -1.5, 3.5, -3.5]

c \leq u + delta
c \geq u - delta



# advisory = argmax(scores)
# u = u_vals[advisory]
# use mip_verify idea for max.

max_score_aux_vars = [get_mip_aux_var(mip_model, binary=true) for i = 1:n_controller_output]
@constraint(mip_model.model, sum(max_score_aux_vars) == 1)
max_score = get_mip_aux_var(mip_model)
for i = 1:n_controller_output
    u_max_sub_i = maximum([bounds[end].center[j] + bounds[end].radius[j] for j in 1:n_controller_output if j != i])
    l_i = bounds[end].center[i] - bounds[end].radius[i]
    Δ = u_max_sub_i - l_i
    @constraint(mip_model.model, max_score <= neurons[end][i] + Δ * (1 - max_score_aux_vars[i]))
    @constraint(mip_model.model, max_score >= neurons[end][i])
end
@constraint(mip_model.model, u_own_var == sum(max_score_aux_vars .* u_own_vals))


mip_summary(mip_model.model)

next_x =  mip_model.vars_dict[output_var_dict[:xnew]]
@objective(mip_model.model, Min, next_x)


t_begin = now()
JuMP.optimize!(mip_model.model)
t_end = now()
println("compute time was $((t_end - t_begin).value / 1000) s")


objective_value(mip_model.model)

for (k, v) in pairs(output_var_dict)
    val = value(mip_model.vars_dict[v])
    println("$k = $(val)")
end

advisory = argmax([value(x) for x in max_score_aux_vars])

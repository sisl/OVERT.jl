include("sound/proof_functions.jl")
include("examples/jmlr/comparisons/dreal_utils.jl")
using LazySets
using NeuralVerification

# test inputs :
state_vars = [:x1, :x2]
control_vars = [:u1]
input_set = Dict(:x1=>[1., 1.2], :x2=>[0., 0.2])
controller_file = "nnet_files/jair/single_pendulum_small_controller.nnet"
include("models/single_pendulum/single_pend.jl")
dynamics_map = Dict(single_pend_input_vars[1]=>single_pend_input_vars[2],
                    single_pend_input_vars[2]=>single_pend_θ_doubledot)
dt = 0.1
output_constraints = [:(x1 <= -0.2167)] # (avoid set)
N_steps=25
experiment_name = "single_pendulum_small_controller"
dirname="examples/jmlr/comparisons/"

# inputs: state_vars::Array{Symbol}, control_vars::Array{Symbol}, input_set::dict{Symbol=>Array{Reals}}, controller_file, dynamics_map, dt, output_constraints

# initializations: 
    # SMTLibFormula container to store smt2, 
    formula = SMTLibFormula()
    # N=0, 
    # load controller as network object
    controller = read_nnet(controller_file)
    # turn controller into u expression(s) just like symbolic dynamics (maybe just RHS?)
    u_expr = network2expr(controller, state_vars)

"""
# assert init state variables in init set 
# for 1 = 1:N
    # u_var = add controller
    # state_vars = add dynamics
# end
# add output constraints 
"""
# assert init state variables in init set
    # timestamp state vars in init set to be e.g. x_0
    timed_input_set = Dict((Meta.parse(string(k)*"_0"),v) for (k,v) in input_set)
    push!(formula.formula, define_domain(timed_input_set, formula.stats)...)

results = []
write_result(dirname*experiment_name, experiment_name*"\n"; specifier="w")
tstart = time()
for t = 0:N_steps-1
    u_t = add_controller(control_vars, u_expr::Array{Expr}, state_vars, formula::SMTLibFormula, t)
    x_tp1 = add_dynamics(state_vars, control_vars, t, dynamics_map, dt, formula)
    result = add_output_constraints_and_check_property(formula, output_constraints, state_vars, t+1)
    t_sofar = time() - tstart
    msg = "Property holds for timestep $(t+1) ? $result . Elapsed time: $(t_sofar) sec \n"
    print(msg)
    write_result(dirname*experiment_name, msg; specifier="a")
    push!(results, result)
end
ΔT = time() - tstart
msg = "Property holds for all timesteps? $(all(results)). Total elpased time: ΔT"
println(msg)
write_result(experiment_name, msg; specifier="a")




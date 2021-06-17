include("../../../sound/proof_functions.jl")
include("dreal_utils.jl")
using LazySets
using NeuralVerification

function compare_to_dreal(state_vars::Array{Symbol}, control_vars::Array{Symbol}, input_set::Dict, controller_file::String, dynamics_map::Dict, dt::T where T<:Real, output_constraints::Array{Expr}, dirname::String, experiment_name::String, N_steps::T where T<:Real; dreal_path="/opt/dreal/4.21.06.1/bin/dreal")
    """
    # Basic overview of what this function does:
    # assert init state variables in init set 
    # for t = 0:N-1
        # u_var = add controller
        # state_vars = add dynamics
        # check property at t+1 
    # end 
    """
    # initializations: 
    # SMTLibFormula container to store smt2
    formula = SMTLibFormula()
    # load controller as network object
    controller = read_nnet(controller_file)
    # turn controller into u expression(s) just like symbolic dynamics (maybe just RHS?)
    u_expr = network2expr(controller, state_vars)

    # assert init state variables in init set
    # timestamp state vars in init set to be e.g. x_0
    timed_input_set = Dict((Meta.parse(string(k)*"_0"),v) for (k,v) in input_set)
    push!(formula.formula, define_domain(timed_input_set, formula.stats)...)

    results = []
    write_result(dirname*experiment_name, experiment_name*"\n"; specifier="w")
    tstart = time()
    for t = 0:N_steps-1
        # actual important lines of code:
        u_t = add_controller(control_vars, u_expr::Array{Expr}, state_vars, formula::SMTLibFormula, t)
        x_tp1 = add_dynamics(state_vars, control_vars, t, dynamics_map, dt, formula)
        result = add_output_constraints_and_check_property(formula, output_constraints, state_vars, t+1; dreal_path=dreal_path)
        # time stuff and results recording:
        t_sofar = time() - tstart
        msg = "Property holds for timestep $(t+1) ? $result . Elapsed time: $(t_sofar) sec \n"
        print(msg)
        write_result(dirname*experiment_name, msg; specifier="a")
        push!(results, result)
        if t_sofar > 18*60*60
            write_result(dirname*experiment_name, "Breaking after $(t_sofar/60/60) hours. TIMEOUT. \n"; specifier="a")
            break;
        end
    end
    ΔT = time() - tstart
    msg = "Property holds for all timesteps? $(all(results)). Total elpased time: $ΔT sec\n"
    println(msg)
    write_result(dirname*experiment_name, msg; specifier="a")
    return ΔT
end



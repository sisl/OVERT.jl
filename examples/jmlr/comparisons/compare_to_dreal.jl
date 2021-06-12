include("sound/proof_functions.jl")

# inputs: state_vars::Array{Symbol}, control_vars::Array{Symbol}, input_set::dict{Symbol=>Array{Reals}}, controller_file, symbolic_dynamics, output_constraints

# initializations: 
    # SMTLibFormula container to store smt2, 
    formula = SMTLibFormula()
    # N=0, 
    # load controller as network object
    controller = read_nnet(controller_file)
    u_expr = network2expr(controller, state_vars)

# prep: 
    # turn controller into u expression(s) just like symbolic dynamics (maybe just RHS?)
    # turn ẋ into dynamics xt+1 = f(xt, u) expression [maybe? unless it's easier not to? ]

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
    # assertions = define_domain(timestamped_init_set, smtlibformula.formula_stats) # TODO: check that smtlibformula.formula stats is properly modified 
    # push assertions into smtlibformula.formula

for t = 1:N
    u_timed = add_controller(control_vars, u_expr::Array{Expr}, state_vars, formula::SMTLibFormula, t)
    # state_vars = add dynamics [separate function]
        # timestamp inputs and outputs 
        # assertion = convert_any_constraints(Expr)
        #add assertion to formula
end
# add output constraints 
# write to file 
formula = gen_full_formula(formula::SMTLibFormula, timed_input_set)
write_to_file(formula::SMTLibFormula, "dreal_test.smt2"; dirname="examples/jmlr/comparisons/")
# call dreal on file


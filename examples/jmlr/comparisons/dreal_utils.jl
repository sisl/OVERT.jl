using NeuralVerification: ReLU, Id, Network, Layer

# helper functions for the conversion to smt2
act_dict = Dict(ReLU()=> :relu)
function network2expr(nn::Network, state::Array{Symbol})
    exprs = symbols.(state) # starts out as [x1, x2] -> [w1*x1+b1, w2*x2+b2]
    for l in nn.layers 
        println("activation is: $(l.activation)")
        if l.activation == Id()
            exprs = l.weights*exprs + l.bias
        else
            exprs = elementwise_apply(act_dict[l.activation], l.weights*exprs + l.bias)
        end
    end
    return Basic2Expr(exprs)
end

# --------stolen from certificateverification repo 
# Generate a metric network which has ReLU activation on its ouput layer.
# the output will be interpreted as the diagonal of our M matrix
function random_network(layer_sizes, activations)
    layers = Layer[]
    for i = 1:(length(layer_sizes)-1)
        layer_in = layer_sizes[i]
        layer_out = layer_sizes[i+1]
        weights = randn(layer_out, layer_in)
        bias = randn(layer_out)
        # All ReLU including the last layer because we want
        # the output to be >= 0
        push!(layers, Layer(weights, bias, activations[i]))
    end
    return Network(layers)
end

# Apply a symbolic function elementwise to a list of Basics
function elementwise_apply(symbol::Symbol, items::Array)
    # Convert to a list of expressions
    expressions = Basic2Expr(items)
    # Wrap each expression in the function 
    wrapped_expressions = [:($symbol($expr)) for expr in expressions]
    # Convert back to a list of basics
    return Basic.(wrapped_expressions)
end

function Basic2Expr(e)
    return Meta.parse.(string.(expand.(e)))
end
# --------------------------------------------------

function test_network2expr()
    network = random_network([2,2,2], [ReLU(), Id()])
    state = [:x1, :x2]
    exprs = network2expr(network, state)
    x_rand = rand(1,2)
    # check equivalency by subbing in values to Expr?
    map = Dict(zip(state, x_rand))
    eval_expr = [eval(substitute(e, map)) for e in exprs]
end

function timestamp_expr(e::Expr, N)
    """
    Convert x to x_$N and x_tp1 to x_$(N+1)
    """
end

function substitute(expression, map)
    # replace each key with each value in the expression 
    e_string = string(expression)
    for (k,v) in map
        e_string = replace(e_string, string(k)=>string(v))
    end
    return Meta.parse(e_string)
end

function add_controller(u::Array{Symbol}, u_expr::Array{Expr}, x::Array{Symbol}, formula::SMTLibFormula, N::Int)
    # u is like [:u_1, :u_2]
    # x is like [:x_1, :x_2]
    # u_expr is like [:(W12*relu(W_11*x_1 + b1) + b2)] where W11, b1, etc. are constants like 3.453

    # u_var = add controller() [separate function]
    # timestamp inputs and outputs (maybe handle LHS and RHS separately...)
    # time inputs:
    u_timed = [Meta.parse("$(v)_$N") for v in u]
    x_timed =  [Meta.parse("$(v)_$N") for v in x]
    map = Dict(zip([u...,x...], [u_timed...,x_timed...]))
    # time exprs
    u_expr_timed = [substitute(e, map) for e in u_expr]
    # assertion = convert_any_constraint(Expr)
    for (i,u_i) in enumerate(u_timed)
        e = :($(u_i) == $(u_expr_timed[i]))
        #println("c.e = $e")
        assertion = assert_literal(e, formula.stats)
        push!(formula.formula, assertion) # add assertion to formula
    end
    return u_timed
end

function test_add_controller()
    u = [:u1, :u2]
    u_exprs = [:(x1 - W*x2 + relu(x3)), :(x1^2 - relu(x2))]
    x = [:x1, :x2, :x3]
    formula = SMTLibFormula()
    N = 2
    u_timed = add_controller(u, u_exprs, x, formula, N)
end

function add_dynamics(x::Array{Symbol}, u::Array{Symbol}, N::Int, dynamics_map::Dict, dt::T where T <: Real, formula::SMTLibFormula)
    """
    inputs:
        # u is like [:u1, :u2]
        # x is like [:x1, :x2]
        # dynamics map is like: Dict(:x1 => :x2, :x2 => :(u1^2 - 5*x1), )

    # state_vars = add dynamics [separate function]
    # timestamp inputs and outputs 
    # construct RHS integration
    # assertion = convert_any_constraints(Expr)
    #add assertion to formula
    """
    # timestamp inputs 
    u_timed = [Meta.parse("$(v)_$N") for v in u]
    x_timed =  [Meta.parse("$(v)_$N") for v in x]
    map = Dict(zip([u...,x...], [u_timed...,x_timed...]))
    for x_i in x
        dx = substitute(dynamics_map[x_i], map)
        x_i_n = Meta.parse("$(x_i)_$N")
        x_i_np1 = Meta.parse("$(x_i)_$(N+1)")
        # x_i_tp1 = x_t + xdot_t * dt
        e = :($(x_i_np1) == $(x_i_n) + $dx*$dt)
        println("d.e = $e")
        assertion = assert_literal(e, formula.stats)
        push!(formula.formula, assertion)
    end
    return [Meta.parse("$(v)_$(N+1)") for v in x]
end

function define_relu()
    return ["(define-fun relu ((arg Real)) Real (max arg 0))"]
end

function gen_full_formula(formula::SMTLibFormula)
    pushfirst!(formula.formula, declare_reals(formula.stats)...)
    pushfirst!(formula.formula, define_relu()...)
    pushfirst!(formula.formula, header()...)
    push!(formula.formula, footer()...)
    return formula
end

function add_output_constraints!(formula::SMTLibFormula, c::Expr, x::Array{Symbol}, N::T where T <: Real)
    x_N = [Meta.parse("$(v)_$N") for v in x]
    map = Dict(zip([x...], [x_N...]))
    c_timed = substitute(c, map)
    push!(formula.formula, assert_literal(c_timed, formula.stats))
end

function run_dreal(full_fname::String; δ=0.001, jobs=1, dreal_path="/opt/dreal/4.20.12.1/bin/dreal")
    # returns "true" if property is unsat (holds)
    println("running dreal on file $(full_fname) with precision $δ and $jobs jobs.")
    result = read(`$dreal_path $full_fname --precision $δ --jobs $jobs`, String)
    @debug("result: ", result)
    # write result file
    println("Writing result to: $(full_fname)")
    write_result(full_fname, result)
    return occursin("unsat", result) 
end

function add_output_constraints_and_check_property(formula::SMTLibFormula, output_constraints::Array{Expr}, x::Array{Symbol}, t::T where T<:Real; δ=0.001, jobs=1, dreal_path="/opt/dreal/4.21.06.1/bin/dreal")
    unsat = true
    # add output constraints 
    ## create a separate file for each output constraint
    for (i,c) in enumerate(output_constraints)
        formula_i = deepcopy(formula)
        add_output_constraints!(formula_i, c, x, t)
        # write to file 
        formula_i = gen_full_formula(formula_i::SMTLibFormula)
        full_fname = write_to_file(formula_i::SMTLibFormula, "dreal_test_constraint$(i)_time$t.smt2"; dirname="examples/jmlr/comparisons/smtlibfiles/")
        # call dreal on file
        unsat &= run_dreal(full_fname; δ=δ, jobs=jobs, dreal_path=dreal_path)
    end
    return unsat
end
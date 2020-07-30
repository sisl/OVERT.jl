# Code to demonstrate the soundness of the OVERT approach in constrast
# to the unsoundness of fitting a neural network to represent the 
# dynamical system directly

include("../models/problems.jl")
include("../models/car/car.jl")
include("../models/car/simple_car.jl")
using Flux
using Intervals
using IterTools: ncycle

global DEBUG = true

""" Types """
mutable struct SoundnessQuery
    ϕ # the original 
    ϕ̂ # the approximate
    domain # the domain over which to check if: phi => phihat   is valid.
end

mutable struct FormulaStats
    bools # boolean vars. arraylike
    reals # real vars. arraylike
    bool_macros #arraylike
    new_bool_var_count::Int
    bool_macro_count::Int
end
FormulaStats() = FormulaStats([],[],[],0,0)

mutable struct SMTLibFormula
    formula # arraylike
    stats::FormulaStats
end
SMTLibFormula() = SMTLibFormula([], FormulaStats())

mutable struct MyError
    message::String
end

mutable struct Problem
    name::String
    executable_fcn
    symbolic_fcn # may be a list of relational expressions representing the true function
    overt_problem::OvertProblem
    oa #::OverApproximation
    domain
end
# maybe this should be:
# exec...
# sym ...
# oa 
# nn

function Problem(name::String, executable, symbolic_fcn, overt_problem::OvertProblem, domain)
    return Problem(name::String, executable, symbolic_fcn, overt_problem::OvertProblem, nothing, domain)
end

function Problem(name::String, executable, overt_problem::OvertProblem, domain)
    return Problem(name::String, executable, nothing, overt_problem::OvertProblem, nothing, domain)
end

"""Printing/Writing Functions"""
function Base.show(io::IO, f::SMTLibFormula)
    s = "SMTLibFormula: "
    s *= "new_bool_var_count = " * string(f.stats.new_bool_var_count)
    s *= ", bools: " * string(f.stats.bools)
    s *= ", reals: " * string(f.stats.reals)
    println(io, s)
end

function write_to_file(f::SMTLibFormula, fname; dirname="smtlibfiles/")
    # print expressions to file
    if DEBUG
        println(join(f.formula, "\n"))
    end
    # TODO: make dir before writing to file in it
    try
        mkdir(pwd() * dirname) # make dir if it doesn't exist
    catch
    end
    full_fname = pwd() * dirname * fname
    file = open(full_fname, "w")
    write(file, join(f.formula, "\n"))
    close(file)
    return full_fname
end

"""High Level Soundness Verification Functions"""

"""
# for the given problem, compare the soundness of approximating
# the dynamics using a neural network vs approximating the 
# dynamics using OVERT
"""
function compare_soundness(problem::String)
    NN_result = check_soundness(problem, approx="NN")
    OVERT_result = check_soundness(problem, approx="OVERT")
    println("Comparing Soundness of Approximations for Problem: ", problem)
    println("The result for NN is: ")
    println(NN_result)
    println("The result for OVERT is: ")
    println(OVERT_result)
end

function check_soundness(problem::String; approx="OVERT")
    # construct SoundnessQuery
    query = construct_soundness_query(problem, approx)::SoundnessQuery
    # check soundness query
    solver = "dreal"
    result = check(solver, query, approx*problem*".smt2")
    return result
end

"""
The soundness of the approximation query asks if 
    ϕ ⟹ ϕ̂ 
is valid (read: always true) (where ϕ is the original function and ϕ̂ is the approximation)
over the domain d, 
(meaning: anything that satisfies ϕ also satisfies ϕ̂, the definition of an overapproximation)
and we can encode this by asking if the negation: 
    ¬(ϕ ⟹ ϕ̂) 
is unsatisfiable.

Implication can be written: 
    a⟹b == ¬a ∨ b 
and so we can rewrite 
    ¬(ϕ ⟹ ϕ̂)
as 
    ¬(¬ϕ ∨ ϕ̂)
and again as
    ϕ ∧ ¬ϕ̂
which is the final formula that we will encode. 
"""
function soundnessquery2smt(query::SoundnessQuery)
    stats = FormulaStats()
    ϕ = assert_conjunction(query.ϕ, stats; conjunct_name="phi")
    notϕ̂ = assert_negation_of_conjunction(query.ϕ̂, stats; conjunct_name="phihat")
    main_formula = vcat(["; assert phi"], ϕ, ["; assert not phi hat"], notϕ̂) 
    #
    whole_formula = vcat(header(), declare_reals(stats), define_domain(query.domain), main_formula, footer())
    return SMTLibFormula(whole_formula, stats)   
end

function check(solver::String, query::SoundnessQuery, fname::String)
    result = nothing
    if solver == "dreal"
        smtlibscript = soundnessquery2smt(query)
        full_fname = write_to_file(smtlibscript, fname)
        # call dreal from command line to execute on smtlibscript
        run(`dreal $full_fname`)
        # read results file? and return result?
        result = read_result(full_fname)
    else
        throw(MyError("Not implemented"))
    end
    return result
end

function read_result(fname)
    # results will be put in a txt  file of the same name but with "result" appended
    io = open(fname[1:end-4]*"_result.txt", "r")
    result = read(io, String)
    close(io)
    return result
end

function define_domain(d)
    # d is a dictionary denoting the domain, e.g. {"x" => [-.3, 5.6], ...}
    assertions = []
    for (k,v) in d
        lb = v[1]
        ub = v[2]
        box = assert_statement(define_box(string(k),lb, ub))
        push!(assertions, box)
    end
    return assertions
end

function define_box(v::String, lb, ub)
    lb = lb < 0 ? prefix_notate("-", [-lb]) : string(lb)
    ub = ub < 0 ? prefix_notate("-", [-ub]) : string(ub)
    lb_e = prefix_notate("<=", [v, ub])
    ub_e = prefix_notate(">=", [v, lb])
    return prefix_notate("and", [lb_e, ub_e])
end

function create_OP_for_dummy_sin()
    return OvertProblem(
        x->sin(x),
        overt_sin,
        nothing,
        nothing,
        nothing
    )
end

function overt_sin(range_dict, N_OVERT::Int)
    v1 = :(sin(x))
    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
    range_dict[v1_oA.output] = v1_oA.output_range
    return v1_oA, v1_oA.output
end

"""
problem::String -> problem::Problem
"""
function get_problem(problem::String)
    if problem == "dummy_sin"
        domain = Dict(:x => [-π, π])
        return Problem("dummy_sin", 
                        x->sin(x),
                        create_OP_for_dummy_sin(), 
                        domain)
        # NOTE: there may be a small problem in that 
        # the symbolic function needs to match symbolic expressions in OVERT...
        # also i think the desired form of the symbolic dynamics will be different
        # depending on whether we are dealing with the NN approx or the OVERT approx...
    elseif problem == "simple_car"
        domain = Dict(:x1 =>[-1,1], :x2=>[-1,1], :x3=>[-1,1], :x4=>[-1,1])
        return Problem("simple_car", 
                        SimpleCar.true_dynamics, 
                        nothing,
                        SimpleCar, 
                        domain)
    else
        throw(MyError("not implemented."))
    end
end

function construct_soundness_query(p::String, approx)
    problem = get_problem(p)
    if approx == "OVERT"
        ϕ̂ = construct_OVERT(problem)
    elseif approx == "NN"
        ϕ̂ = get_NN(problem)
    end
    ϕ = problem.symbolic_fcn
    return SoundnessQuery(ϕ, ϕ̂, problem.domain)
end

function construct_OVERT(problem::Problem)
    range_dict = deepcopy(problem.domain)
    oa::OverApproximation, output_vars = problem.overt_problem.overt_dynamics(range_dict, 2)
    problem.oa = oa
    sym_dict = oa.fun_eq
    sym_funs = [:($k==$v) for (k,v) in sym_dict]
    problem.symbolic_fcn = sym_funs
    # what to return, exactly?
    # return: combine two arrays of oa.approx_eq, oa.approx_ineq
    # TODO: check that problem::Problem is modified once this function exits
    return vcat(oa.approx_eq, oa.approx_ineq)
end

"""If NN file exists and retrain=false, load saved NN. Otherwise, train new one."""
function get_NN(problem::Problem; retrain=false)
    # if NN exists and retrain=false
    # load NN 
    # else 
    # fit NN 
    model = fit_NN(problem)
    ϕ̂ = NN2Exprs(model)
end

"""Turn a Flux model into a list of Julia symbolic expressions"""
function NN2Exprs(model)
end

""" Use executable version of function to 
fit neural network approximation"""
function fit_NN(problem::Problem)
    n_states = length(problem.overt_problem.input_vars)
    n_controls = length(problem.overt_problem.control_vars)
    n_inputs = n_states + n_controls
    n_outputs = n_states
    # set up model
    model  = Chain(
        Dense(n_inputs,20,relu),
        Dense(20,20,relu),
        Dense(20, n_outputs)
    )
    params = Flux.params(model) 
    # get datasets
    X, Y = sample_dataset(problem, n_inputs)
    train_loader = DataLoader((X, Y), batchsize=100, shuffle=true)
    Xtest, Ytest = sample_dateset(problem, n_inputs)
    # setup loss
    loss(x, y) = Flux.Losses.mse(model(x), y)
    # setup callbacks
    evalcb() = @show(sum(loss.(Xtest, Ytest)))
    # train 
    optimizer = ADAM(3e-2)
    Flux.train!(loss, params, ncycle(train_loader, 5), optimizer, cb = Flux.throttle(evalcb, 1))
    println("After training test loss is: ", sum(loss.(Xtest, Ytest)))
    # return network
    return model
end

"""sample dataset to train on.
Assumes overt_problem.input_vars are listed in the same order that the dynamics function expects.
e.g. they are listed in the <model>.jl file as: [:x1, :x2, :x3] and the dynamics function expects 
something of the form [x1, x2, x3] """
function sample_dateset(problem::Problem; N=10)
    # sample points in domain of state
    X = get_random_points(problem.domain, problem.overt_problem.input_vars, N)
    # sample points in domain of control
    U = get_random_points(problem.domain, problem.overt_problem.control_vars, N)
    # compute label using executable function 
    Y = problem.executable_fcn.(X, U)

    F = collect(eachrow(hcat(hcat(X...)',hcat(U...)')))
    return F, Y
end 

"""Get random samples of a multi-dimensional variable. domain contains a range for the 
variable: (:x => [4,5])"""
function get_random_points(domain::Dict, variables, N)
    lbs = [domain[v][1] for v in variables]
    ubs = [domain[v][2] for v in variables]
    deltas = ubs - lbs
    # sample points in domain 
    X = rand(N, length(variables))*deltas .+ lbs'
    @assert all(X[:,1] .∈ lbs[1]..ubs[1]) # sanity check to make sure broadcasting works correctly
    return collect(eachrow(X))
end

"""Low level functions for converting ϕ and ϕ̂ to smtlib2"""

"""
f is an array representing a conjunction.
Returns an array
"""
function assert_conjunction(f::Array, fs::FormulaStats; conjunct_name=nothing)
    if length(f) == 1
        return [assert_literal(f[1], fs)]
    elseif length(f) > 1
        # assert conjunction
        return assert_actual_conjunction(f, fs; conjunct_name=conjunct_name)::Array
    else # empty list
        return []
    end
end

function assert_literal(l, fs::FormulaStats)
    return assert_statement(convert_any_constraint(l, fs::FormulaStats))
end

function assert_negated_literal(l, fs::FormulaStats)
    return assert_statement(negate(convert_any_constraint(l, fs::FormulaStats)))
end

function assert_negation_of_conjunction(f::Array, fs::FormulaStats; conjunct_name=nothing)
    if length(f) == 1
        return [assert_negated_literal(f[1], fs)]
    elseif length(f) >= 1
        return assert_actual_negation_of_conjunction(f, fs; conjunct_name=conjunct_name)::Array
    else # empty list
        return []
    end
end

function not(atom)
    return ~atom::Bool
end

function add_real_var(v, fs::FormulaStats)
    if not(v in fs.reals)
        push!(fs.reals, v)
    end
end

function add_real_vars(vlist::Array, fs::FormulaStats)
    for v in vlist
        add_real_var(v, fs)
    end
end

function get_new_bool(fs::FormulaStats)
    fs.new_bool_var_count += 1
    v = "b"*string(fs.new_bool_var_count) # b for boolean
    @assert not(v in fs.bools)
    push!(fs.bools, v)
    return v
end

"""
Creates prefix notation syntax of smtlib2.
Turn an op into its prefix form: (op arg1 arg2 ...).
A 'pure' style function that doesn't modify state. 
"""
function prefix_notate(op, args)
    expr = "(" * op * " "
    expr *= print_args(args)
    expr *= ")"
    return expr
end

function print_args(args::Array)
    s = ""
    for a in args
        s *= string(a) * " "
    end
    s = s[1:end-1] # chop trailing whitespace
    return s
end

function declare_const(constname, consttype)
    return prefix_notate("declare-const", [constname, consttype])
end

# e.g. (define-fun _def1 () Bool (<= x 2.0))
function define_fun(name, args, return_type, body)
    arg_string = "("*print_args(args)*")"
    return prefix_notate("define-fun", [name, arg_string, return_type, body])
end

function declare_reals(fs::FormulaStats)
    real_decls = []
    for v in fs.reals
        push!(real_decls, declare_const(v, "Real"))
    end
    return real_decls
end

function define_atom(atomname, atomvalue)
    eq_expr = prefix_notate("=", [atomname, atomvalue])
    return assert_statement(eq_expr)
end

function define_bool_macro(macro_name, expr)
    return define_macro(macro_name, expr; return_type="Bool")
end

# (define-fun _def1 () Bool (<= x 2.0))
function define_macro(macro_name, expr; return_type="Bool")
    return define_fun(macro_name, [], return_type, expr)
end

function assert_statement(expr)
    return prefix_notate("assert", [expr])
end

function negate(expr)
    return prefix_notate("not", [expr])
end

function footer()
    return ["(check-sat)", "(get-model)"]
end

function header()
    h = [produce_models(), set_logic()]
    # push!(h, define_max(), define_min(), define_relu()) # min and max already defined by dreal
    return h
end

function set_logic()
    return "(set-logic QF_NRA)"
end

function produce_models()
    return "(set-option :produce-models true)"
end

# define-fun is a macro.
# if x < y return y else x
function define_max()
    return "(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))"
end

# if x < y return x else y
function define_min()
    return "(define-fun min ((x Real) (y Real)) Real (ite (< x y) x y))"
end

function define_relu()
    return "(define-fun relu ((x Real)) Real (max x 0))"
end

function assert_actual_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing)
    formula, conjunct_name = declare_conjunction(constraint_list, fs; conjunct_name=conjunct_name) # declare conjunction
    push!(formula, assert_statement(conjunct_name)) # assert conjunction
    return formula
end

function assert_actual_negation_of_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing)
    """
    Assert the negation of conjunction of the constraints passed in constraint_list.
    not (A and B and C and ...)
    """
    formula, conjunct_name = declare_conjunction(constraint_list, fs; conjunct_name=conjunct_name) # declare conjunction
    push!(formula, assert_statement(negate(conjunct_name))) # assert NEGATED conjunction
    return formula
end

function declare_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing, use_macros=false)
    """
    Given a list of constraints, declare their conjunction but DO NOT
    assert their conjunction.
    can use macros e.g. 
    (define-fun _def1 () Bool (<= x 2.0))
    (define-fun _def2 () Bool (>= y 3.0))
    
    or can use new boolean variables
    (declare-const b1 Bool)
    (assert (== b1 (<= x 2.0)))
    
    ...
    (declare ... phi)
    (assert (= phi (and A B)))

    But notice we are just _defining_ phi, we are not asserting that
    phi _holds_, which would be: (assert phi) [not doing that tho!]
    """
    defs, names = declare_list(constraint_list, fs; use_macros=use_macros)
    if isnothing(conjunct_name)
        conjunct_name = get_new_bool(fs)
    end
    @assert length(names) > 1
    conjunct = prefix_notate("and", names)
    conjunct_decl = [declare_const(conjunct_name, "Bool")]
    conjunct_def = [define_atom(conjunct_name, conjunct)]
    formula = vcat(defs, conjunct_decl, conjunct_def)
    return formula, conjunct_name
end

function assert_disjunction(constraint_list, fs::FormulaStats; disjunct_name=nothing)
    throw(MyError("NotImplementedError"))
end

"""
Convert julian relational operators into smtlib2 friendly ones. 
"""
function convert_f(f::Symbol)
    if f == :(==)
        return "="
    elseif f == :≤ || f == :≦
        return "<="
    elseif f == :≥ || f == :≧
        return ">="
    else 
        return string(f)
    end
end

function convert_any_constraint(c::Expr, fs::FormulaStats)
    # basically just prefix notate the constraint and take from expr -> string
    # base case: numerical number
    try
        constant = eval(c)
        return convert_any_constraint(constant::Real, fs)
    catch e
    end
    # recursive case
    f = convert_f(c.args[1])
    args = c.args[2:end]
    converted_args = []
    for a in args
        push!(converted_args, convert_any_constraint(a, fs))
    end
    return prefix_notate(string(f), converted_args)
end
# base cases:
function convert_any_constraint(s::Symbol, fs::FormulaStats)
    add_real_var(s, fs) # log var
    return string(s)
end
function convert_any_constraint(n::Real, fs::FormulaStats)
    if n >= 0
        return string(n)
    else #if n < 0
        return prefix_notate("-", [-n]) # smtlib2 can't handle negative numbers and needs unary minus, ugh (!)
    end
end

function declare_list(constraint_list::Array, fs::FormulaStats; use_macros=false)
    """
    turn a list of some type of AbstractConstraint <: into 
    
    smtlib macro declarations/definitions:
    (define-fun _def1 () Bool (<= x 2.0))
    
    But DON'T assert either true or false for the macro (e.g. (assert _def1()) )
    
    OR new binary variables:
    
    (declare-const b1 Bool)
    (assert (= b1 (<= v 9)))
    
    And again DON'T assert either (assert b1) or (assert (not b1))
    """
    defs = [] # definitions + declarations
    names = [] # names
    for item in constraint_list 
        expr = convert_any_constraint(item, fs)::String
        if item.args[1] == :(==)
            # if this is a definition of a variable/function, just assert it.
            push!(defs, assert_statement(expr))
        else
            if use_macros 
                macro_name = get_new_macro(fs)
                push!(names, macro_name)
                push!(defs, define_bool_macro(macro_name, expr))
            else # use new bools
                bool_name = get_new_bool(fs)
                push!(names, bool_name)
                push!(defs, declare_const(bool_name, "Bool"))
                push!(defs, define_atom(bool_name, expr))
            end
        end
    end
    return defs, names
end

function get_new_macro(fs::FormulaStats)
    """ Get new macro name """
    fs.bool_macro_count += 1
    v = "_def"*string(fs.bool_macro_count) 
    @assert not(v in fs.bool_macros)
    push!(fs.bool_macros, v)
    return v
end


# Code to demonstrate the soundness of the OVERT approach in constrast
# to the unsoundness of fitting a neural network to represent the 
# dynamical system directly

include("../models/problems.jl")
include("../models/car/car.jl")
include("../models/car/simple_car.jl")

""" Types """
mutable struct SoundnessQuery
    ϕ # the original 
    ϕ̂ # the approximate
    domain # the domain over which to check if: phi => phihat   is valid.
end

mutable struct SMTLibFormula
    formula # arraylike
    stats::FormulaStats
end
SMTLibFormula() = SMTLibFormula([], FormulaStats())

mutable struct FormulaStats
    bools # boolean vars. arraylike
    reals # real vars. arraylike
    bool_macros #arraylike
    new_bool_var_count::Int
    bool_macro_count::Int
end
FormulaStats() = FormulaStats([],[],0)

mutable struct MyError
    message:string
end

mutable struct Problem
    name::string
    executable_fcn
    symbolic_fcn # may be a list of relational expressions representing the true function
    overt_problem::OvertProblem
    oa::OverApproximation
    domain
end

"""Printing/Writing Functions"""
function Base.show(io::IO, f::SMTLibFormula)
    s = "SMTLibFormula: "
    s *= "new_bool_var_count = " * string(f.new_bool_var_count)
    s *= ", bools: " * string(f.bools)
    s *= ", reals: " * string(f.reals)
    println(io, s)
end

function write_to_file(f::SMTLibFormula)
    # print expressions to file
end

"""High Level Soundness Verification Functions"""

"""
# for the given problem, compare the soundness of approximating
# the dynamics using a neural network vs approximating the 
# dynamics using OVERT
"""
function compare_soundness(problem::string)
    NN_result = check_soundness(problem, approx="NN")
    OVERT_result = check_soundness(problem, approx="OVERT")
    println("Comparing Soundness of Approximations for Problem: ", problem)
    println("The result for NN is: ")
    println(NN_result)
    println("The result for OVERT is: ")
    println(OVERT_result)
end

function check_soundness(problem::string; approx="OVERT")
    # construct SoundnessQuery
    query = construct_soundness_query(problem, approx)
    # check soundness query
    solver = "dreal"
    result = check(solver, query)
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

end

function check(solver::string, query::SoundnessQuery)
    if solver == "dreal"
        smtlibscript = soundnessquery2smt(query)
        write_to_file(smtlibscript)
        # call dreal from command line to execute on smtlibscript
        # read results file? and return result?
    else
        throw(MyError("Not implemented"))
    end
end

function create_OP_for_dummy_sin()
    return OvertProblem()
end

"""
problem::string -> problem::Problem
"""
function get_problem(string)
    if problem == "dummy_sin"
        domain = Dict(:x => [-π, π])
        return Problem("dummy_sin", 
                        x->sin(x), 
                        create_OP_for_dummy_sin(), 
                        domain)
    elseif problem == "simple_car"
        domain = Dict(:x1 =>[-1,1], :x2=>[-1,1], :x3=>[-1,1], :x4=>[-1,1])
        return Problem("simple_car", 
                        SimpleCar.true_dynamics, 
                        SimpleCar, 
                        domain)
    else
        throw(MyError("not implemented."))
    end
end

function construct_soundness_query(p::string, approx)
    problem = get_problem(p)
    if approx == "OVERT"
        phihat = construct_OVERT(problem)
    elseif approx == "NN"
        phihat = fit_NN(problem)
    end
    phi = problem.symbolic_fcn
    return SoundnessQuery(phi, phihat, problem.domain)
end

function construct_OVERT(problem::Problem)
    oa::OverApproximation, output_vars = problem.overt_problem.overt_dynamics(problem.domain, -1)
    # what to return, exactly?
    # return: combine two arrays of oa.approx_eq, oa.approx_ineq
    return vcat(oa.approx_eq, oa.approx_ineq)
end

function fit_NN(problem::Problem)
    # TODO
    # some code where I call flux and stuff
end

"""Low level functions for converting ϕ and ϕ̂ to smtlib2"""

"""
f is an array representing a conjunction.
Returns an array
"""
function assert_conjunction(f::Array, fs::FormulaStats)
    if length(f) == 1
        return [assert_literal(f[1], fs)]
    elseif length(f) > 1
        # assert conjunction
        return assert_actual_conjunction(f, fs)::Array
    else # empty list
        return []
    end
end

function assert_literal(l, fs::FormulaStats)
    return assert_statement(convert_any_constraint(l, fs::FormulaStats)[1])
end

function assert_negated_literal(l, fs::FormulaStats)
    return assert_statement(negate(convert_any_constraint(l, fs::FormulaStats)[1]))
end

function assert_negated_conjunction(f::Array, fs::FormulaStats)
    if length(f) == 1
        return [assert_negated_literal(f[1], fs)]
    elseif length(f) >= 1
        return assert_actual_negated_conjunction(f, fs)::Array
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
    h = [set_logic(), produce_models()]
    push!(h, [define_max(), define_relu()])
    return h
end

function set_logic()
    return "(set-logic ALL)"
end

function produce_models()
    return "(set-option :produce-models true)"
end

function define_max()
    return "(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))"
end

function define_relu()
    return "(define-fun relu ((x Real)) Real (max x 0))"
end

function assert_actual_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing)
    formula, conjunct_name = declare_conjunction(constraint_list, fs; conjunct_name=conjunct_name) # declare conjunction
    push!(formula, assert_statement(conjunct_name)) # assert conjunction
    return formula
end

function assert_actual_negated_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing)
    """
    Assert the negation of conjunction of the constraints passed in constraint_list.
    not (A and B and C and ...)
    """
    formula, conjunct_name = declare_conjunction(constraint_list, fs; conjunct_name=conjunct_name) # declare conjunction
    formula += [assert_statement(negate(conjunct_name))] # assert NEGATED conjunction
    return formula
end

function declare_conjunction(constraint_list, fs::FormulaStats; conjunct_name=nothing)
    """
    Given a list of constraints, declare their conjunction but DO NOT
    assert their conjunction.
    e.g. 
    (define-fun _def1 () Bool (<= x 2.0))
    (define-fun _def2 () Bool (>= y 3.0))
    ...
    (declare ... phi)
    (assert (= phi (and A B)))
    But notice we are just _defining_ phi, we are not asserting that
    phi _holds_, which would be: (assert phi) [not doing that tho!]
    """
    macro_defs, macro_names = declare_list(constraint_list, fs)
    if isnothing(conjunct_name)
        conjunct_name = get_new_bool(fs)
    end
    @assert length(macro_names) > 1
    conjunct = prefix_notate("and", macro_names)
    conjunct_decl = [declare_const(conjunct_name, "Bool")]
    conjunct_def = [define_atom(conjunct_name, conjunct)]
    formula = vcat(macro_defs + conjunct_decl + conjunct_def)
    return formula, conjunct_name
end

function assert_disjunction(constraint_list, fs::FormulaStats; disjunct_name=nothing)
    throw(MyError("NotImplementedError"))
end

function convert_any_constraint(c::Expr, fs::FormulaStats)
    # basically just prefix notate the constraint and take from expr -> string
    # base case: numerical number
    try
        constant = eval(c)
        return string(constant)
    catch e
    end
    # recursive case
    f = c.args[1]
    args = c.args[2:end]
    converted_args = []
    for a in args
        push!(converted_args, convert_any_constraint(a, fs))
    end
    return prefix_notate(string(f), converted_args)
end
# base cases:
function convert_any_constraint(s::Symbol, fs::FormulaStats)
    return string(s)
end

function declare_list(constraint_list::Array, fs::FormulaStats)
    """
    turn a list of some type of AbstractConstraint <: into 
    smtlib macro declarations/definitions:
    (define-fun _def1 () Bool (<= x 2.0))
    But DON'T assert either true or false for the macro (e.g. (assert _def1()) )
    """
    macro_defs = [] # definitions + declarations
    macro_names = [] # names
    for item in constraint_list 
        expr = convert_any_constraint(item, fs) # may return a list of constraints when converted
        for e in expr
            macro_name = get_new_macro(fs)
            push!(macro_names, macro_name)
            push!(macro_defs, define_bool_macro(macro_name, e))
        end
    end
    return macro_defs, macro_names
end

function get_new_macro(fs::FormulaStats)
    """ Get new macro name """
    fs.bool_macro_count += 1
    v = "_def"*string(fs.bool_macro_count) 
    @assert not(v in fs.bool_macros)
    push!(fs.bool_macros, v)
    return v
end


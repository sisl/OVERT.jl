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
    bools # arraylike
    reals # arraylike
    new_var_count::Int
end
SMTLibFormula() = SMTLibFormula([], [], [], 0)

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
    s *= "new_var_count = " * string(f.new_var_count)
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
function soundnessquer2smt(query::SoundnessQuery)

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
function assert_conjunction(f::Array)
    if length(f) == 1
        return [assert_literal(f[1])]
    elseif length(f) > 1
        # assert conjunction
        return assert_actual_conjunction(f)::Array
    else # empty list
        return []
    end
end

function assert_literal(l)
    return assert_statement(convert_any_constraint(l)[1])
end

function assert_negated_literal(l)
    return assert_statement(negate(convert_any_constraint(l)[1]))
end

function assert_negated_conjunction(f::Array)
    if length(f) == 1
        return [assert_negated_literal(f[1])]
    elseif length(f) >= 1
        return assert_actual_negated_conjunction(f)::Array
    else # empty list
        return []
    end
end


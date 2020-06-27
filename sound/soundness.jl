# Code to demonstrate the soundness of the OVERT approach in constrast
# to the unsoundness of fitting a neural network to represent the 
# dynamical system directly

include("../models/problems.jl")
include("../models/car/car.jl")
include("../models/car/simple_car.jl")

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

mutable struct SoundnessQuery
    phi # the original 
    phihat # the approximate
    domain # the domain over which to check if: phi => phihat   is valid.
end

function check_soundness(problem::string; approx="OVERT")
    # construct SoundnessQuery
    query = construct_soundness_query(problem, approx)
    # check soundness query
    solver = "dreal"
    result = check(solver, query)
    return result
end

function check(solver::string, query::SoundnessQuery)
    if solver == "dreal"
        arg = convert_to_smtlib(query)
    else
        throw(MyError("Not implemented"))
    end
end

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
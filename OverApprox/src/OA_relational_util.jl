# Relational OverApproximation Utilities

mutable struct OverApproximation
    output::Symbol
    output_range::Array{T, 1} where {T <: Real}
    ranges::Dict{Symbol, Array{T, 1}} where {T <: Real}
    nvars::Integer
    consts::Array{Symbol, 1}
    approx_eq::Array{Expr, 1}
    approx_ineq::Array{Expr, 1}
    fun_eq::Dict{Symbol, Any}
    N::Integer # number of points in a bound in a [0,1] interval
    ϵ::Real # construct overapprox with ϵ leeway from function
end
# default constructor
OverApproximation() = OverApproximation(:(null_output), 
                                        Array{Float64, 1}(), 
                                        Dict{Symbol, Array{Float64,1}}(), 
                                        0, 
                                        Array{Symbol, 1}(), 
                                        Array{Expr, 1}(), 
                                        Array{Expr, 1}(), 
                                        Dict{Symbol, Any}(), 
                                        3,
                                        1e-2)

function add_var(bound)
    bound.nvars += 1
    # @ is the symbol preceding A in ascii
    return Symbol('v'*('@'+bound.nvars))
end

function is_number(bound::OverApproximation, var::Symbol)
    # return true if a variable is marked as a constant
    # this works because we are doing DFS, and args are parsed
    # BEFORE functions are parsed
    if (var in bound.consts)
        return true
    else
        return false
    end 
end
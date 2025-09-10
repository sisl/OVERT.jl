# Relational OverApproximation Utilities

mutable struct OverApproximation
    output::Union{Symbol, Real}
    output_range::Array{T, 1} where {T <: Real}
    ranges::Dict{Symbol, Array{T, 1}} where {T <: Real}
    nvars::Integer
    consts::Array{Symbol, 1}
    approx_eq::Array{Expr, 1}
    approx_ineq::Array{Expr, 1}
    fun_eq::Dict{Symbol, Any}
    N::Integer # number of points in a bound in a [0,1] interval
    ϵ::Real # add ±ϵ to bounds after computation to give "air gap" from function 
    rel_error_tol::Real # tightness parameter for bounds on nonlinear functions (determines N when N=-1 (automatic))
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
                                        1e-4,
                                        5e-3)

Base.print(oA::OverApproximation) = print_overapproximate(oA)
Base.display(oA::OverApproximation) = print_overapproximate(oA)
function print_overapproximate(oA::OverApproximation)
    println("output = $(oA.output)")
    for eq in oA.approx_eq
        println(eq)
    end
    for ineq in oA.approx_ineq
        println(ineq)
    end
    println(oA.output_range)
end

"""
This function combines a list of OverApproximation objects and return one
    that contains all their equality and ineqaulity constraints.

    ** for output, output_range, nvars, const, fun_eq, N and ϵ, we keep the default values
"""
function add_overapproximate(list_oA::Array{OverApproximation, 1})

    out_oA = OverApproximation()
    out_oA.ranges = merge([oA.ranges for oA in list_oA]...)
    out_oA.approx_eq = vcat([oA.approx_eq for oA in list_oA]...)
    out_oA.approx_ineq = vcat([oA.approx_ineq for oA in list_oA]...)
    return out_oA
end
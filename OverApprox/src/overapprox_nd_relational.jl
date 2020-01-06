include("autoline.jl")
include("overest_new.jl")
include("utilities.jl")
using SymEngine
using Revise

# TODO: return whole range dict for use in SBT
# TODO: make more readable/modular

"""
Overapproximate an n-dimensional function using a relational abstraction. 
"""
function overapprox_nd(expr, range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real}; nvars::Integer=0, N::Integer=3) # range_dict::Dict{Symbol, Array{<:Real, 1}}
    # returns: (output_variable::Symbol, range_of_output_variable::Tuple, #variables, [list of equalities], [list of inequalities])

    range_dict = floatize(range_dict)
    expr = reduce_args_to_2(expr)

    # base cases
    if expr isa Symbol
        return (expr, range_dict[expr], nvars, [], [])
    elseif is_number(expr)
        # newvar, nvars = get_new_var(nvars)
        # c  = eval(expr)
        # return (newvar, [c,c], nvars, [:($newvar = $expr)], []) # for use with infinite precision solvers
        c  = eval(expr)
        return (c, [c,c], nvars, [], []) # for easier parsing with marabou
    elseif is_affine(expr)
        newvar, nvars = get_new_var(nvars)
        expr_range = [find_affine_range(expr, range_dict)...]
        println("typeof(range) is ", typeof(expr_range))
        return (newvar, expr_range, nvars, [:($newvar = $expr)], [])
    # recursive cases
    else # is a non-affine function
        if is_unary(expr)
            f = expr.args[1]
            xexpr = expr.args[2]
            # recurse on argument
            x, xrange, nvars, eq_list, ineq_list = overapprox_nd(xexpr, range_dict, nvars=nvars)
            # handle outer function f 
            LBvar, UBvar, OArange, eq_list, ineq_list, nvars = bound_unary_function(f, x, xrange, N, eq_list, ineq_list, nvars)
            ## create variable representing overapprox and add inequality to ineq list
            OAvar, nvars = get_new_var(nvars)
            push!(ineq_list, (:($LBvar ≦ $OAvar)) )
            push!(ineq_list, (:($OAvar ≦ $UBvar)) )
            return (OAvar, OArange, nvars, eq_list, ineq_list)
        else # is binary (assume all multiplication of variables has been processed out)
            # pull out the big guns
            @assert is_binary(expr)
            f = expr.args[1]
            xexpr = expr.args[2]
            yexpr = expr.args[3]
            # recurse on aguments 
            x, xrange, nvars, eq_list_x, ineq_list_x = overapprox_nd(xexpr, range_dict, nvars=nvars)
            y, yrange, nvars, eq_list_y, ineq_list_y = overapprox_nd(yexpr, range_dict, nvars=nvars)
            # merge constraints
            eq_list = [eq_list_x..., eq_list_y...]
            ineq_list = [ineq_list_x..., ineq_list_y...]
            # handle outer function f
            (z, zrange, nvars, eq_list, ineq_list) = bound_binary_functions(f, xexpr, xrange, x, yexpr, yrange, y, N, eq_list, ineq_list, nvars)
            return (z, zrange, nvars, eq_list, ineq_list)
        end
    end
end

function bound_binary_functions(f, xexpr, xrange, x, yexpr, yrange, y, N, eq_list, ineq_list, nvars)
    if f == :+
        z, nvars = get_new_var(nvars)
        push!(eq_list, :($z == $x + $y))
        sum_range = (xrange[1] + yrange[1], xrange[2] + yrange[2])
        return (z, sum_range, nvars, eq_list, ineq_list)
    elseif f == :-
        z, nvars = get_new_var(nvars)
        push!(eq_list, :($z == $x - $y))
        diff_range = (xrange[1] - yrange[2],  xrange[2] - yrange[1])
        return (z, diff_range, nvars, eq_list, ineq_list)
    elseif f == :*
        if (is_number(xexpr) | is_number(yexpr))
            # multiplication of VARIABLES AND SCALARS ONLY
            z, nvars = get_new_var(nvars)
            (z, prod_range, nvars, eq_list_prod, ineq_list_prod) = multiply_variable_and_scalar(xexpr, xrange, x, yexpr, yrange, y, z, nvars)
            return (z, prod_range, nvars, [eq_list..., eq_list_prod...], [ineq_list..., ineq_list_prod...])
        else # both args contain variables
            # first expand multiplication expr
            expanded_mul_expr, range_dict, eq_list_exp, nvars = expand_multiplication(x, y, Dict(x=>xrange, y=>yrange), nvars)
            # and then bound that expression in a "stand alone" way (but pass # vars)
            outputvar, outputrange, nvars, eq_list_oa, ineq_list_oa = overapprox_nd(expanded_mul_expr, range_dict; nvars=nvars, N=N)
            # and then combine constraints (eqs and ineqs) and return them to continue on with recursion
            return (outputvar, outputrange, nvars, [eq_list..., eq_list_exp..., eq_list_oa...], [ineq_list..., ineq_list_oa...]) 
        end
    else
        error("unimplemented")
        return (:0, [0,0], 0, [], []) # for type stability, maybe?
    end
end

function multiply_variable_and_scalar(xexpr, xrange, x, yexpr, yrange, y, z, nvars)
    @assert (is_number(xexpr) | is_number(yexpr))
    eq_list = []
    ineq_list = []
    if is_number(xexpr)
        c = eval(xexpr)
        push!(eq_list, :($z == $y * $c))
        prod_range = multiply_interval(yrange, c)
        return (z, prod_range, nvars, eq_list, ineq_list)
    elseif is_number(yexpr)
        c = eval(yexpr)
        push!(eq_list, :($z == $x * $c))
        prod_range = multiply_interval(xrange, c)
        return (z, prod_range, nvars, eq_list, ineq_list)
    end
end

function expand_multiplication(x, y, range_dict, nvars; δ=0.1)
    """
    Re write multiplication e.g. x*y using exp(log()) and affine expressions
    e.g. x*y, x ∈ [a,b] ∧ y ∈ [c,d]
         x2 = x - a + δ  , x2 ∈ [δ, b - a + δ] aka x2 > 0   (recall, b > a)
            x = x2 + a - δ
         y2 = y - c + δ , y2 ∈ [δ, d - c + δ] aka y2 > 0   (recall, d > c)
            y = y2 + c - δ
        x*y = (x2 + a - δ)*(y2 + c - δ) 
            = x2*y2 + (a - δ)*y2 + (c - δ)*x2 + (a - δ)*(c - δ)
            = exp(log(x2*y2)) + (a - δ)*y2 + (c - δ)*x2 + (a - δ)*(c - δ)
            = exp(log(x2) + log(y2)) + (a - δ)*y2 + (c - δ)*x2 + (a - δ)*(c - δ)
        In this final form, everything is decomposed into unary functions, +, and affine functions!
    """ 
    x2, nvars = get_new_var(nvars)
    y2, nvars = get_new_var(nvars)
    a,b = range_dict[x]
    c,d = range_dict[y]
    eq_list = [:($x2 = $x - $a + $δ), :($y2 = $y - $c + $δ) ]
    range_dict[x2] = [δ, b - a + δ]
    range_dict[y2] = [δ, d - c + δ]
    expr = :( exp(log($x2) + log($y2)) + ($a - $δ)*$y2 + ($c - $δ)*$x2 + ($a - $δ)*($c - $δ) )
    return expr, range_dict, eq_list, nvars
end

function apply_fx(f, a)
    substitute!(f, :x, a)
end

function bound_unary_function(f, x, xrange, N, eq_list, ineq_list, nvars, plotflag=true)   
    ## create upper and lower bounds
    eval(:(fun = $f))
    p = plotflag ? plot(0,0) : nothing
    UBpoints, UBfunc_sym, UBfunc_eval = find_UB(fun, xrange[1], xrange[2], N; lb=false, plot=plotflag, existing_plot=p)
    fUBrange = [find_1d_range(UBpoints)...]
    LBpoints, LBfunc_sym, LBfunc_eval = find_UB(fun, xrange[1], xrange[2], N; lb=true, plot=plotflag, existing_plot=p)
    fLBrange = [find_1d_range(LBpoints)...]
    ## create new vars for these expr, equate to exprs, and add them to equality list 
    # e.g. y = fUB(x), z = fLB(x)
    UBvar, nvars = get_new_var(nvars)
    push!(eq_list, :($UBvar == $(apply_fx(UBfunc_sym, x))))
    LBvar, nvars = get_new_var(nvars)
    push!(eq_list, :($LBvar == $(apply_fx(LBfunc_sym, x))))
    OArange = [min(fLBrange...,fUBrange...), max(fLBrange..., fUBrange...)]
    return LBvar, UBvar, OArange, eq_list, ineq_list, nvars
end

function floatize(dict::Dict{Symbol, Array{T, 1}} where {T <: Real})
    # values of dict must be convertible to float
    newdict = Dict{Symbol,Array{Float64,1}}()
    for k=keys(dict)
        newdict[k] = float(dict[k])
    end
    return newdict::Dict{Symbol, Array{T, 1}} where {T <: Real}
end
include("autoline.jl")
include("overest_new.jl")
include("utilities.jl")
using SymEngine
using Revise

# TODO: return whole range dict for use in SBT
# TODO: make more readable/modular

struct OverApproximation
    output::Symbol
    output_range::Array{T, 1} where {T <: Real}
    ranges::Dict{Symbol, Array{T, 1}} where {T <: Real}
    nvars::Integer
    consts::Array{Symbol, 1}
    approx_eq::Array{Expr, 1}
    approx_ineq::Array{Expr, 1}
    fun_eq::Dict{Symbol, Expr}
    N::Integer # number of points in a bound in a [0,1] interval
    # default constructor
    OverApproximation() = OverApproximation(:(null_output), Array{Float64, 1}(), Dict{Symbol, Array{Float64,1}}(), 0, Array{Symbol, 1}(), Array{Expr, 1}(), Array{Expr, 1}(), Dict{Symbol, Expr}(), 3)
end

"""
Overapproximate an n-dimensional function using a relational abstraction. 
"""
function overapprox_nd(expr, 
                       range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real}; 
                       N::Integer=3)
    bound = OverApproximation()
    range_dict = floatize(range_dict)
    bound.ranges = range_dict
    bound.N = N
    return overapprox_nd(expr, bound)
end

function overapprox_nd(expr,  
                       bound::OverApproximation)
    # returns: bound::OverApproximation
    expr = reduce_args_to_2(expr)

    # base cases
    if expr isa Symbol
        bound.output = expr
        bound.output_range = bound.ranges[expr] # TODO: make sure all fields of bound are filled out properly
        return bound
    elseif is_number(expr)
        bound.output = add_var(bound)
        c  = eval(expr)
        bound.output_range = [c,c]
        # use :($newvar = $expr) # for use with infinite precision solvers
        push!(bound.approx_eq, :($bound.output == $c))
        bound.fun_eq[bound.output] = c
        bound.ranges[bound.output] = bound.output_range
        push!(bound.consts, bound.output)
        return bound # for easier parsing with marabou
    elseif is_affine(expr)
        bound.output = add_var(bound)
        bound.output_range = [find_affine_range(expr, bound.ranges)...]
        push!(bound.approx_eq, :($bound.output = $expr))
        bound.fun_eq[bound.output] = expr
        bound.ranges[bound.output] = bound.output_range
        #println("typeof(range) is ", typeof(expr_range))
        return bound
    # recursive cases
    else # is a non-affine function
        if is_unary(expr)
            f = expr.args[1]
            xexpr = expr.args[2]
            # recurse on argument
            bound = overapprox_nd(xexpr, bound)
            # handle outer function f 
            bound = bound_unary_function(f, bound)
            return bound
        else # is binary 
            # pull out the big guns
            @assert is_binary(expr)
            f = expr.args[1]
            xexpr = expr.args[2]
            yexpr = expr.args[3]
            # recurse on aguments 
            bound = overapprox_nd(xexpr, bound)
            x = bound.output
            bound = overapprox_nd(yexpr, bound)
            y = bound.output
            # handle outer function f
            bound = bound_binary_functions(f, x, y, bound)
            return bound
        end
    end
end

function bound_binary_functions(f, x, y, bound)
    if f == :+
        z = add_var(bound)
        push!(bound.approx_eq, :($z == $x + $y))
        bound.fun_eq[z] = :($x + $y)
        bound.output = z
        sum_range = (bound.ranges[x][1] + bound.ranges[y][1], bound.ranges[x][2] + bound.ranges[y][2])
        bound.output_range = sum_range
        bound.ranges[z] = sum_range
        return bound
    elseif f == :-
        z = add_var(bound)
        push!(bound.approx_eq, :($z == $x - $y))
        bound.fun_eq[z] = :($x - $y)
        bound.output = z
        diff_range = (bound.ranges[x][1] - bound.ranges[y][2],  bound.ranges[x][2] - bound.ranges[y][1])
        bound.output_range = diff_range
        bound.ranges[z] = diff_range
        return bound
    elseif f == :*
        if (is_number(bound, x) | is_number(bound, y))
            # multiplication of VARIABLES AND SCALARS ONLY
            bound.output  = add_var(bound)
            bound = multiply_variable_and_scalar(x, y, bound)
            return bound
        else # both args contain variables
            # first expand multiplication expr
            expanded_mul_expr, bound = expand_multiplication(x, y, bound)
            # and then bound that expression wrt x2, y2
            bound = overapprox_nd(expanded_mul_expr, bound)
            return bound 
        end
    else
        error("unimplemented")
        return OverApproximation() # for type stability, maybe?
    end
end

function multiply_variable_and_scalar(x, y, bound)
    @assert (is_number(bound, x) | is_number(bound, y))
    if is_number(bound, x)
        bound = mul_var_sca_helper(y, x, bound)
        return bound
    elseif is_number(bound, y)
        bound = mul_var_sca_helper(x, y, bound)
        return bound
    end
end

function mul_var_sca_helper(var, constant, bound)
    z = bound.output
    push!(bound.approx_eq, :($z == $var * $constant))
    bound.fun_eq[z] = :($var * $constant)
    prod_range = multiply_interval(bound.ranges[var], bound.ranges[constant][1])
    bound.ranges[z] = prod_range
    bound.output_range = prod_range
    return bound
end

function expand_multiplication(x, y, bound; ξ=0.1)
    """
    Re write multiplication e.g. x*y using exp(log()) and affine expressions
    e.g. x*y, x ∈ [a,b] ∧ y ∈ [c,d], ξ>0
         x2 = x - a + ξ  , x2 ∈ [ξ, b - a + ξ] aka x2 > 0   (recall, b > a)
            x = x2 + a - ξ
         y2 = y - c + ξ , y2 ∈ [ξ, d - c + ξ] aka y2 > 0   (recall, d > c)
            y = y2 + c - ξ
        x*y = (x2 + a - ξ)*(y2 + c - ξ) 
            = x2*y2 + (a - ξ)*y2 + (c - ξ)*x2 + (a - ξ)*(c - ξ)
            = exp(log(x2*y2)) + (a - ξ)*y2 + (c - ξ)*x2 + (a - ξ)*(c - ξ)
            = exp(log(x2) + log(y2)) + (a - ξ)*y2 + (c - ξ)*x2 + (a - ξ)*(c - ξ)
        In this final form, everything is decomposed into unary functions, +, and affine functions!
    """ 
    x2 = add_var(bound)
    y2 = add_var(bound)
    a,b = bound.ranges[x]
    c,d = bound.ranges[y]
    @assert(b >= a)
    @assert(d >= c)
    push!(bound.approx_eq, :($x2 == $x - $a + $ξ))
    push!(bound.approx_eq, :($y2 == $y - $c + $ξ))
    bound.fun_eq[x2] = :($x - $a + $ξ)
    bound.fun_eq[y2] = :($y - $c + $ξ)
    bound.ranges[x2] = [ξ, b - a + ξ]
    bound.ranges[y2] = [ξ, d - c + ξ]
    expr = :( exp(log($x2) + log($y2)) + ($a - $ξ)*$y2 + ($c - $ξ)*$x2 + ($a - $ξ)*($c - $ξ) )
    return expr, bound
end

function apply_fx(f, a)
    substitute!(f, :x, a)
end

function bound_unary_function(f, x_bound; plotflag=true)   
    ## create upper and lower bounds of function f(x)
    eval(:(fun = $f))
    p = plotflag ? plot(0,0) : nothing
    UBpoints, UBfunc_sym, UBfunc_eval = find_UB(fun, x_bound.output_range[1], x_bound.output_range[2], x_bound.N; lb=false, plot=plotflag, existing_plot=p)
    fUBrange = [find_1d_range(UBpoints)...]
    LBpoints, LBfunc_sym, LBfunc_eval = find_UB(fun, x_bound.output_range[1], x_bound.output_range[2], x_bound.N; lb=true, plot=plotflag, existing_plot=p)
    fLBrange = [find_1d_range(LBpoints)...]
    ## create new vars for these expr, equate to exprs, and add them to equality list 
    # e.g. y = fUB(x), z = fLB(x)
    bound = x_bound
    UBvar = add_var(bound)
    push!(bound.approx_eq, :($UBvar == $(apply_fx(UBfunc_sym, x_bound.output))))
    bound.ranges[UBvar] = fUBrange
    LBvar = add_var(bound)
    push!(bound.approx_eq, :($LBvar == $(apply_fx(LBfunc_sym, x_bound.output))))
    bound.ranges[LBvar] = fLBrange
    OArange = [min(fLBrange...,fUBrange...), max(fLBrange..., fUBrange...)]
    ## create variable representing overapprox and add inequality to ineq list
    OAvar = add_var(bound)
    push!(bound.approx_ineq, (:($LBvar ≦ $OAvar)) )
    push!(bound.approx_ineq, (:($OAvar ≦ $UBvar)) )
    bound.fun_eq[OAvar] = :($(f)($x_bound.output))
    bound.ranges[OAvar] = OArange
    bound.output = OAvar
    bound.output_range = OArange
    return bound
end

function floatize(dict::Dict{Symbol, Array{T, 1}} where {T <: Real})
    # values of dict must be convertible to float
    newdict = Dict{Symbol,Array{Float64,1}}()
    for k=keys(dict)
        newdict[k] = float(dict[k])
    end
    return newdict::Dict{Symbol, Array{T, 1}} where {T <: Real}
end
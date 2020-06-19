#include("overt_header.jl")
include("autoline.jl")
include("overest_new.jl")
include("overt_utils.jl")
include("OA_relational_util.jl")
using SymEngine
using Revise

plotflag = true

"""
    overapprox_nd(expr,
                  range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real};
                  N::Integer=2,
                  ϵ::Real=1e-2)
    -> OverApproximation
Overapproximate an n-dimensional function using a relational abstraction.
"""
function overapprox_nd(expr,
                       range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real};
                       N::Integer=2,
                       ϵ::Real=1e-2)
    bound = OverApproximation()
    range_dict = floatize(range_dict)
    bound.ranges = range_dict
    bound.N = N
    bound.ϵ = ϵ
    #expr = simplify(expr) # turns x/6 to (1/6)*x, does not affect 6/x or x/y
    return overapprox_nd(expr, bound)
end

"""
    overapprox_nd(expr, bound::OverApproximation) -> bound::OverApproximation

Helper function. Bounds the expression _expr_ in a recursive manner. bound is used as a container
to collect the relations describing the bound along the way.
"""
function overapprox_nd(expr,
                       bound::OverApproximation)
    # println(expr)
    # all operations should have at most two arguments.
    expr = rewrite_division_by_const(expr)
    expr = reduce_args_to_2(expr)
    @debug(expr)

    # base cases
    if expr isa Symbol
        @debug("symbol base case")
        bound.output = expr
        bound.output_range = bound.ranges[expr]
        return bound
    elseif is_number(expr)
        @debug("is number base case")
        bound.output = eval(expr)
        return bound
        # bound.output = add_var(bound)
        # c  = eval(expr)
        # bound.output_range = [c,c]
        # # use :($newvar = $expr) # for use with infinite precision solvers
        # push!(bound.approx_eq, :($(bound.output) == $c))
        # bound.fun_eq[bound.output] = c
        # bound.ranges[bound.output] = bound.output_range
        # push!(bound.consts, bound.output)
        # return bound
    elseif is_affine(expr)
        @debug("is affine base case")
        bound.output = add_var(bound)
        bound.output_range = [find_affine_range(expr, bound.ranges)...]
        push!(bound.approx_eq, :($(bound.output) == $expr))
        bound.fun_eq[bound.output] = expr
        bound.ranges[bound.output] = bound.output_range
        #println("typeof(range) is ", typeof(expr_range))
        return bound
    elseif is_1d(expr)
        @debug("is 1d base case")
        sym = find_variables(expr)
        arg = sym[1]
        bound = bound_1arg_function(expr, arg, bound)
        return bound
    # recursive cases
    else # is a non-affine function
        if is_unary(expr)
            @debug("is unary")
            # unary expression with two args, like sin(x), cos(x), ...
            f = expr.args[1]
            xexpr = expr.args[2]
            # recurse on argument
            bound = overapprox_nd(xexpr, bound)
            if f == :- # handle unary minus
                @debug "let affine handle unary minus"
                new_expr = :(-$(bound.output))
                return overapprox_nd(new_expr, bound)
            end
            # handle outer function f
            bound = bound_unary_function(f, bound)
            return bound
        else # is binary
            # pull out the big guns
            @assert is_binary(expr)
            @debug("is binary")
            f = expr.args[1]
            xexpr = expr.args[2]
            yexpr = expr.args[3]
            # recurse on aguments
            bound = overapprox_nd(xexpr, bound)
            x = bound.output
            #
            bound = overapprox_nd(yexpr, bound)
            y = bound.output
            # handle outer function f
            bound = bound_binary_functions(f, x, y, bound)
            return bound
        end
    end
end

function bound_binary_functions(f, x, y, bound) # TODO: should only be for when both args are vars or 6/x
    @debug "bound binary functions: f x y:" f x y
    mul_two_vars = !is_number(x) && !(is_number(y)) && f  == :*
    divide_by_var = !is_number(y) && f == :/
    is_power = f == :^
    if !(mul_two_vars || divide_by_var || is_power)
        # then is affine
        new_expr = :($f($x,$y))
        @debug "rewriting into affine expr: " new_expr
        return overapprox_nd(new_expr, bound)
    end

    if f == :+
        @debug("bounding +. should not be used anymore")
        z = add_var(bound)
        push!(bound.approx_eq, :($z == $x + $y))
        bound.fun_eq[z] = :($x + $y)
        bound.output = z
        if is_number(x)
            lb_x, ub_x = x, x
        else
            lb_x, ub_x = bound.ranges[x][1], bound.ranges[x][2]
        end
        if is_number(y)
            lb_y, ub_y = y, y
        else
            lb_y, ub_y = bound.ranges[y][1], bound.ranges[y][2]
        end
        sum_range = [lb_x + lb_y, ub_x + ub_y]
        bound.output_range = sum_range
        bound.ranges[z] = sum_range
        return bound
    elseif f == :-
        @debug("bounding -. should not be used.")
        z = add_var(bound)
        push!(bound.approx_eq, :($z == $x - $y))
        bound.fun_eq[z] = :($x - $y)
        bound.output = z
        if is_number(x)
            lb_x, ub_x = x, x
        else
            lb_x, ub_x = bound.ranges[x][1], bound.ranges[x][2]
        end
        if is_number(y)
            lb_y, ub_y = y, y
        else
            lb_y, ub_y = bound.ranges[y][1], bound.ranges[y][2]
        end
        diff_range = [lb_x - lb_y,  ub_x - ub_y]
        bound.output_range = diff_range
        bound.ranges[z] = diff_range
        return bound
    elseif f == :*
        if (is_number(x) | is_number(y))
            @debug("bounding * with a var and a const")
            # multiplication of VARIABLES AND SCALARS ONLY
            # TODO: Actually, I think this is redundant and COULD be captured by "is_affine"
            bound.output  = add_var(bound)
            bound = multiply_variable_and_scalar(x, y, bound)
            return bound
        else # both args contain variables
            @debug("bounding * btw two variables")
            # first expand multiplication expr
            expanded_mul_expr, bound = expand_multiplication(x, y, bound)
            # and then bound that expression wrt x2, y2
            bound = overapprox_nd(expanded_mul_expr, bound)
            return bound
        end
    elseif f == :/
        # handles x/y and c/y
        # x/c should be transformed into (1./c)*x by simplify
        if y isa Symbol # if y is symbol (variable)
            @debug("division x/y or c/y")
            # range of y should not include 0
            if bound.ranges[y][1]*bound.ranges[y][2] <= 0
                error("range of y should not include 0")
            end
            if is_number(x) # c/y
                f_new = :($x/$y)
                bound = bound_1arg_function(f_new, y, bound)
                return bound
            elseif x isa Symbol # x/y
                @debug "converting: $x / $y"
                inv_denom = :(1. / $y)
                f_new = :($x * $inv_denom)
                @debug "into: $f_new"
                return overapprox_nd(f_new, bound)
            end
        else # y is a constant
        error(
            """
            x/c should be handled by simplify and turned into: (1/c)*x
            and c/c should be handled by affine or is_number
            """)
        end
    # The following operation is not yet fully tested.
    elseif f == :^
        if is_number(y) # this is f(x)^a, where a is a constant.
            @debug "handling f(x)^a"
            f_new = :($x^$y)
            bound = bound_1arg_function(f_new, x, bound)
            return bound
        elseif is_number(x)  # this is a^f(x), where a is a constant.
            @debug "handling  a^f(x)"
            f_new = :($x^$y)
            bound = bound_1arg_function(f_new, y, bound)
            return bound
        else  # this is f(x)^g(y)
            error("f(x)^g(y) is not implemented yet.")
        end
    else
        error(" operation $f with operands $x and $y is not implemented")
        return OverApproximation() # for type stability, maybe?
    end
end

function multiply_variable_and_scalar(x, y, bound)
    @assert (is_number(x) | is_number(y))
    if is_number(x)
        bound = mul_var_sca_helper(y, x, bound)
        return bound
    elseif is_number(y)
        bound = mul_var_sca_helper(x, y, bound)
        return bound
    end
end

function mul_var_sca_helper(var, constant, bound)
    z = bound.output
    push!(bound.approx_eq, :($z == $var * $constant))
    bound.fun_eq[z] = :($var * $constant)
    #prod_range = multiply_interval(bound.ranges[var], bound.ranges[constant][1])
    prod_range = multiply_interval(bound.ranges[var], constant)
    bound.ranges[z] = prod_range
    bound.output_range = prod_range
    return bound
end

function expand_multiplication(x, y, bound; ξ=0.1)
    """
        expand_multiplication(x, y, bound; ξ=1.0)
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
    @debug("Expanding multiplication")
    bound.fun_eq[x2] = :($x - $a + $ξ)
    bound.fun_eq[y2] = :($y - $c + $ξ)
    bound.ranges[x2] = [ξ, b - a + ξ]
    bound.ranges[y2] = [ξ, d - c + ξ]
    expr = :( exp(log($x2) + log($y2)) + ($a - $ξ)*$y2 + ($c - $ξ)*$x2 + (($a - $ξ)*($c - $ξ)) )
    return expr, bound
end

function apply_fx(f, a)
    substitute!(f, :x, a)
end

"""
    bound_1arg_function(e::Expr, x::Symbol, bound::OverApproximation; plotflag=true)
Bound one argument functions like x^2.
Create upper and lower bounds of function f(x)
"""
function bound_1arg_function(e::Expr, arg::Symbol, bound::OverApproximation; plotflag=plotflag)
    @debug "bound effectively unary" e arg bound
    fun = SymEngine.lambdify(e, [arg])
    lb, ub, npoint = bound.ranges[arg][1], bound.ranges[arg][2], bound.N
    return bound_unary_function(fun, e, arg, lb, ub, npoint, bound, plotflag=plotflag)
end

"""
    bound_unary_function(f::Symbol, x_bound::OverflowError; plotflag=true)
Bound one argument unary functions like sin(x). Create upper and lower bounds of function f(x)
"""
function bound_unary_function(f::Symbol, x_bound::OverApproximation; plotflag=plotflag)
    @debug "bound true unary" f x_bound.output
    fun = eval(:($f))
    lb, ub, npoint = x_bound.output_range[1], x_bound.output_range[2], x_bound.N
    f_x_expr = :($(f)($(x_bound.output)))
    return bound_unary_function(fun, f_x_expr, x_bound.output, lb, ub, npoint, x_bound, plotflag=plotflag)
end

"""
    bound_unary_function(f::Function, lb, ub, npoint, bound)
Bound one argument functions like sin(x) or x -> x^2 or x -> 1/x. Create upper and lower bounds of function f(x)
"""
function bound_unary_function(fun::Function, f_x_expr, x, lb, ub, npoint, bound; plotflag=plotflag)
    p = plotflag ? plot(0,0) : nothing
    UBpoints, UBfunc_sym, UBfunc_eval = find_UB(fun, lb, ub, npoint; lb=false, plot=plotflag, existing_plot=p, ϵ= bound.ϵ)
    fUBrange = [find_1d_range(UBpoints)...]
    LBpoints, LBfunc_sym, LBfunc_eval = find_UB(fun, lb, ub, npoint; lb=true, plot=plotflag, existing_plot=p, ϵ= -bound.ϵ)
    fLBrange = [find_1d_range(LBpoints)...]
    ## create new vars for these expr, equate to exprs, and add them to equality list
    # e.g. y = fUB(x), z = fLB(x)
    UBvar = add_var(bound)
    push!(bound.approx_eq, :($UBvar == $(apply_fx(UBfunc_sym, x))))
    bound.ranges[UBvar] = fUBrange
    LBvar = add_var(bound)
    push!(bound.approx_eq, :($LBvar == $(apply_fx(LBfunc_sym, x))))
    bound.ranges[LBvar] = fLBrange
    OArange = [min(fLBrange...,fUBrange...), max(fLBrange..., fUBrange...)]
    ## create variable representing overapprox and add inequality to ineq list
    OAvar = add_var(bound)
    push!(bound.approx_ineq, (:($LBvar ≦ $OAvar)) )
    push!(bound.approx_ineq, (:($OAvar ≦ $UBvar)) )
    bound.fun_eq[OAvar] = f_x_expr
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

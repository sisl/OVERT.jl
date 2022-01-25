include("overapprox_1d.jl")
include("autoline.jl")
include("overt_utils.jl")
include("definitions.jl")
using SymEngine
using Plots
plotly()
using PGFPlots

# TODO: put params into a mutable struct
plotflag = false
plottype = "pgf"
DELETE_MUL_BY_ZERO = true
DELETE_DEAD_RELUS = true

function set_plotflag(bool)
    plotflag = bool
end
function set_plottype(t)
    """
    You can change to "html" or "pgf" (latex).
    """
    plottype = t
end

"""
    overapprox(expr,
                  range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real};
                  N::Integer=2,
                  ϵ::Real=1e-2)
    -> OverApproximation
Overapproximate an n-dimensional function using a relational abstraction.

N    is the number of linear segments used per region of constant convexity. To allow OVERT to automatically choose this number, set N to -1. It will add segments until the maximum deviation between the function and the bound is within 2%.

ϵ    is the RELATIVE "air gap" added to the bounds so that the bounds don't touch the function at the point of closest approach. If the function bound lies in the range [a,b] a gap of size ϵ*(b-a) will be added to the y values of all points of the upper bound and subtracted from all points of the lower bound. The default value is 1% or ϵ = 0.01
"""
function overapprox(expr,
                       range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real};
                       N::Integer=2,
                       ϵ::Real=1e-2)
    println("Using N=$N, ϵ=$ϵ")
    bound = OverApproximation()
    range_dict = floatize(range_dict)
    bound.ranges = range_dict
    bound.N = N
    bound.ϵ = ϵ
    return overapprox(expr, bound)
end

"""
    overapprox(expr, bound::OverApproximation) -> bound::OverApproximation

Helper function. Bounds the expression _expr_ in a recursive manner. bound is used as a container
to collect the relations describing the bound along the way.
"""
function overapprox(expr,
                       bound::OverApproximation)
    # all operations should have at most two arguments.
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
        ### This code below preserves an exact numeric expression but adds variables
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
        @debug("$expr is affine base case")
        bound.output = add_var(bound)
        bound.output_range = [find_affine_range(expr, bound.ranges)...]
        push!(bound.approx_eq, :($(bound.output) == $expr))
        bound.fun_eq[bound.output] = expr
        @debug "turning $expr into $(bound.output) = $expr"
        bound.ranges[bound.output] = bound.output_range
        return bound
    # elseif is_1d(expr)
    #     """
    #     Currently, this case would resort to relying on fzero to find roots of d2f which 
    #     is not sound as this is a numerical procedure. Thus it has been removed. 
    #     TODO in the future is to implement a sound version of this for arbitrary
    #     1D functions using symbolic differentiation from Calculus.jl:differentiate to find the 
    #     d2f function and then use the package IntervalRootFinding to find all roots in the 
    #     interval in question in a GUARANTEED way!
    #     Allowing the usage of this case (is_1d) can speed up verification with OVERT for some functions by 
    #     reducing the number of approximations needed and thus reducing the number of relus
    #     used. E.g. sin(x) + log(x^2) is a 1D function but without using the is_1d case, this
    #     would incur 3 separate approximations: x^2, log(v_3) and sin(x) . If this case can be used,
    #     it will recognize that sin(x) + log(x^2) is a 1D function and only construct a single
    #     OVERT approximation.
    #     """
    #     @debug("is 1d base case")
    #     sym = find_variables(expr)
    #     arg = sym[1]
    #     bound = bound_1arg_function(expr, arg, bound)
    #     return bound
    # recursive cases
    else # is a non-affine function
        if is_unary(expr)
            @debug("is unary")
            # unary expression with two args, like sin(x), cos(x), ...
            f = expr.args[1]
            xexpr = expr.args[2]
            # recurse on argument
            bound = overapprox(xexpr, bound)
            if f == :- # handle unary minus
                @debug "let affine handle unary minus"
                new_expr = :(-$(bound.output))
                return overapprox(new_expr, bound)
            elseif f == :relu 
                @debug "PWL case"
                bound = handle_PWL(f, bound.output, bound)
                return bound
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
            bound = overapprox(xexpr, bound)
            x = bound.output
            #
            bound = overapprox(yexpr, bound)
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
    divide_by_const = is_number(y) && f ==:/
    if mul_two_vars
        # both args contain variables
        @debug("bounding * btw two variables")
        # first expand multiplication expr
        expanded_mul_expr, bound = expand_multiplication(x, y, bound)
        # and then bound that expression wrt x2, y2
        bound = overapprox(expanded_mul_expr, bound)
        return bound
    elseif divide_by_var
        # handles x/y and c/y
        # x/c should be transformed into (1./c)*x by simplify
        @debug("division x/y or c/y")
        # range of y should not include 0
        if bound.ranges[y][1]*bound.ranges[y][2] <= 0
            error("range of y should not include 0")
        end
        if is_number(x) # c/y
            @debug("c/y")
            f_new = :($x/$y)
            bound = bound_1arg_function(f_new, y, bound)
            return bound
        elseif x isa Symbol # x/y -> x*(1/y)
            @debug "converting: $x / $y"
            inv_denom = :(1. / $y)
            f_new = :($x * $inv_denom)
            @debug "into: $f_new"
            return overapprox(f_new, bound)
        end
    elseif divide_by_const
        new_expr = rewrite_division_by_const(:($f($x,$y)))
        return overapprox(new_expr, bound)
    # The following operation is not yet fully tested.
    elseif f == :^ # f(x,y)   x^2
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
            return OverApproximation() # type stability?
        end
    elseif is_affine(:($f($x,$y)))
        new_expr = :($f($x,$y))
        @debug "Recursing to let affine handle: " new_expr
        return overapprox(new_expr, bound)
    elseif f ∈ [:min, :max, :relu]
        @debug "PWL case, f=" f
        bound = handle_PWL(f, x, y, bound)
        return bound
    else
        error(" operation $f with operands $x and $y is not implemented")
        return OverApproximation() # for type stability, maybe?
    end
end

function handle_PWL(f, x, y, bound::OverApproximation)
    # two arg case
    f_range = find_range(:($f($x, $y)), bound.ranges)
    bound.output_range = f_range
    if f_range == [0.0, 0.0] && DELETE_DEAD_RELUS # dead relu 
        println("Deleting dead relu. f_range is $(f_range). ")
        bound.output = 0.0
    else # live relu or max or something 
        bound.output = add_var(bound)
        expr = :($(bound.output) == $f($x, $y))
        push!(bound.approx_eq, expr)
        bound.fun_eq[bound.output] = :($f($x, $y))
        @debug "turning $f($x,$y) into $(bound.output) = $(bound.fun_eq[bound.output])"
        bound.ranges[bound.output] = bound.output_range
    end
    return bound
end

function handle_PWL(f, x, bound::OverApproximation)
    # one arg case 
    f_range = find_range(:($f($x)), bound.ranges)
    bound.output_range = f_range
    if f_range == [0.0, 0.0] && DELETE_DEAD_RELUS # dead relu 
        println("Deleting dead relu. f_range is $(f_range). ")
        bound.output = 0.0
    else # live relu or something 
        bound.output = add_var(bound)
        expr = :($(bound.output) == $f($x))
        push!(bound.approx_eq, expr)
        bound.fun_eq[bound.output] = :($f($x))
        @debug "turning $f($x) into $(bound.output) = $(bound.fun_eq[bound.output])"
        bound.ranges[bound.output] = bound.output_range
    end
    return bound
end

function expand_multiplication(x, y, bound; ξ=0.1)
    return expand_multiplication_with_scaling(x, y, bound; ξ=ξ)
end

function expand_multiplication_with_scaling(x, y, bound; ξ=0.1)
    """
        expand_multiplication(x, y, bound; ξ=1.0)
    Re write multiplication e.g. x*y using exp(log()) and affine expressions
    e.g. x*y, x ∈ [a,b] ∧ y ∈ [c,d], ξ>0
         x2 = (x - a)/(b - a) + ξ  , x2 ∈ [ξ, 1 + ξ] aka x2 > 0   (recall, b > a)
            x = (b - a)(x2 - ξ) + a
         y2 = (y - c)/(d - c) + ξ , y2 ∈ [ξ, 1 + ξ] aka y2 > 0   (recall, d > c)
            y = (d - c)(y2 - ξ) + c
        x*y = ((b - a)(x2 - ξ) + a )*((d - c)(y2 - ξ) + c)
            = (b - a)*(d - c)*x2*y2  + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))
            = (b - a)*(d - c)*exp(log(x2*y2)) + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))
            = (b - a)*(d - c)*exp(log(x2) + log(y2)) + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))
        In this final form, everything is decomposed into unary functions, +/-, and affine functions!
    """

    a,b = bound.ranges[x]
    c,d = bound.ranges[y]
    @debug "mult ranges: [$a, $b], [$c, $d]"

    if ((a==0 && b==0) || (c==0 && d==0)) && DELETE_MUL_BY_ZERO
        println("Multiplication by zero.")
        # NOTE: in the future... might want to perserve e.g. v_3 = 0*v_2 and set e.g. v_3 as the output with range 0.0 rather than setting the output to 0.0 directly ¯\_(ツ)_/¯ not sure
        bound.output = 0.0
        return 0.0, bound
    else 
        @assert(b >= a)
        @assert(d >= c)
        b_minus_a = b - a
        d_minus_c = d - c
        x2 = add_var(bound)
        y2 = add_var(bound)
        push!(bound.approx_eq, :($x2 == ($x - $a)/$b_minus_a + $ξ))
        push!(bound.approx_eq, :($y2 == ($y - $c)/$d_minus_c + $ξ))
        @debug("Expanding multiplication")
        bound.fun_eq[x2] = :(($x - $a)/$b_minus_a + $ξ)
        bound.fun_eq[y2] = :(($y - $c)/$d_minus_c + $ξ)
        @debug "$x2 = $(bound.fun_eq[x2])"
        @debug "$y2 = $(bound.fun_eq[y2])"
        bound.ranges[x2] = [ξ, 1. + ξ]
        bound.ranges[y2] = [ξ, 1. + ξ]

        b_minus_a_times_d_minus_c = (b - a)*(d - c)
        mult_expr = :($b_minus_a_times_d_minus_c*exp(log($x2) + log($y2)))
        x_coeff = (b - a)*(c - ξ*(d - c))
        xterm = :($x_coeff*$x2)
        y_coeff = (d - c)*(a - ξ*(b - a))
        yterm = :($y_coeff*$y2)
        constant_term = (c - ξ*(d - c))*(a - ξ*(b - a))

        expr = :($mult_expr + $xterm + $yterm + $constant_term)
        @debug "expanded multiplication expr is: $expr"
        return expr, bound
    end
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
    d2f_zeros, convex = get_regions_1arg(e, arg, lb, ub)
    return bound_unary_function(fun, e, arg, lb, ub, npoint, bound, plotflag=plotflag, d2f_zeros=d2f_zeros, convex=convex)
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
    d2f_zeros, convex = get_regions_unary(f, lb, ub)
    return bound_unary_function(fun, f_x_expr, x_bound.output, lb, ub, npoint, x_bound, plotflag=plotflag, d2f_zeros=d2f_zeros, convex=convex)
end

"""
    bound_unary_function(f::Function, lb, ub, npoint, bound)
Bound one argument functions like sin(x) or x -> x^2 or x -> 1/x. Create upper and lower bounds of function f(x)
"""
function bound_unary_function(fun::Function, f_x_expr, x, lb, ub, npoint, bound; plotflag=plotflag, d2f_zeros=nothing, convex=nothing)
    UBpoints, UBfunc_sym = find_UB(fun, lb, ub, npoint; lb=false, plot=plotflag, ϵ= bound.ϵ, d2f_zeros=d2f_zeros, convex=convex)
    fUBrange = [find_1d_range(UBpoints)...]
    LBpoints, LBfunc_sym = find_UB(fun, lb, ub, npoint; lb=true, plot=plotflag, ϵ= -bound.ϵ, d2f_zeros=d2f_zeros, convex=convex)
    fLBrange = [find_1d_range(LBpoints)...]

    # plot after adding air gap and after turning into closed form expression
    @debug "bounding true unary..."
    if plotflag
        p = Plots.plot(0,0)
        global NPLOTS
        NPLOTS += 1
        if plottype != "pgf"
            p = plot(range(lb, ub, length=100), fun, label="function", color="red")
            plot!(p, [p[1] for p in LBpoints], [p[2] for p in LBpoints],  color="purple", marker=:o, markersize=1, label="lower bound")
            plot!(p, [p[1] for p in UBpoints], [p[2] for p in UBpoints], color="blue", marker=:diamond, markersize=1,  label="upper bound", legend=:right, title="Function and bounds")
            # display(p)
            savefig(p, "plots/bound_"*string(NPLOTS)*".html")
        else # plottype == pgf
            println("Saving PGF plot")
            if f_x_expr.args[1] ∈ [:/, :^]
                # use whole expression in title 
                fun_string = "\$"*string(f_x_expr)*"\$"
            else 
                fun_string = "\$"*string(fun)*"\$"
            end
            println("funstring = $(fun_string)")
            x_plot_points = range(lb, ub, length=100)
            f_x_plot_points = fun.(x_plot_points)
            fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$f(x)\$", xlabel="x", title=fun_string)
            push!(fig, PGFPlots.Plots.Linear(x_plot_points, f_x_plot_points, legendentry=fun_string, style="solid, black, mark=none"))
            push!(fig, PGFPlots.Plots.Linear([p[1] for p in LBpoints], [p[2] for p in LBpoints], legendentry="lower bound", style="solid, purple, mark=none"))
            push!(fig, PGFPlots.Plots.Linear([p[1] for p in UBpoints], [p[2] for p in UBpoints], legendentry="upper bound", style="solid, blue, mark=none"))
            fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"
            PGFPlots.save("plots/bound_"*string(NPLOTS)*".tex", fig)
            #PGFPlots.save("plots/bound_"*string(NPLOTS)*".pdf", fig)
        end

    end
    @debug "LB function is: $LBfunc_sym"
    @debug "UB function is: $UBfunc_sym"

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

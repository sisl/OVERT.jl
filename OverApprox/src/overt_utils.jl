# utilities for overapprox_nd_relational.jl and overest_nd.jl

# opeartions and functions that are supported in overest_nd.jl
special_oper = [:+, :-, :/, :*, :^]
special_func = [:exp, :log, :log10,
                :sin, :cos, :tan,
                :sinh, :cosh, :tanh,
                :asin, :acos, :atan,
                :asinh, :acosh, :atanh]

increasing_special_func = [:exp, :log, :log10,
                         :tan, :sinh, :tanh,
                         :asin, :atan,
                         :asinh, :atanh, :acosh]

function to_pairs(B)
    """
    This function converts the output of overest, a tuple of (x points, y points) in the form ready
        for closed form generator: [(x1,y1), (x2, y2), ...]
    B is an array of points.
    """
    xp,yp = B
    x = vcat(xp...)
    y = vcat(yp...)
    pairs = collect(zip(x,y))
    return pairs
end

# todo: does this do the same thing as SymEngine, free_symbols or Tomer's autoline.get_symbols ?
function find_variables(expr)
    """
    given an expression expr, this function finds all the variables.
    it is useful to identify 1d functions that can be directly implemented
    in the overest algorithm, without any composition hacking.
    Example: find_variables(:(x+1))     = [:x]
             find_variables(:(-x+2y-z)) = [:x, :y, :z]
             find_variables(:(log(x)))  = [:x]
             find_variables(:(x+ x*z))   = [:x, :z]
    """
    all_vars = []
    for arg in expr.args
        if arg isa Expr
            all_vars = vcat(all_vars, find_variables(arg))
        elseif arg isa Symbol
            if !(arg in special_oper) && !(arg in special_func)
                all_vars = vcat(all_vars, arg)
            end
        end
    end
    return unique(all_vars)
end

function is_affine(expr)
    """
    given an expression expr, this function determines if the expression
    is an affine function.

    Example: is_affine(:(x+1))       = true
             is_affine(:(-x+(2y-z))) = true
             is_affine(:(log(x)))    = false
             is_affine(:(x + x*z))   = false
             is_affine(:(x/6))       = true
             is_affine(:(5*x))       = true
             is_affine(:(log(2)*x))  = true
             is_affine(:(-x))        = true
     """
    # it is number
    if is_number(expr)
        return true
    elseif expr isa Symbol # symbol
        return true
    elseif expr isa Expr
        check_expr_args_length(expr)
        func = expr.args[1]
        if func ∉ [:+, :-, :*, :/] # only these operations are allowed
            return false
        else  # func ∈ [:+, :-, :*, :/]
            if func == :* # one of args has to be a number
                option1 =  is_number(expr.args[2]) && is_affine(expr.args[3])
                option2 =  is_number(expr.args[3]) && is_affine(expr.args[2])
                return (option1 || option2)
            elseif func == :/ # second arg has to be a number
                return is_number(expr.args[3])
            else # func is + or -
                return all(is_affine.(expr.args[2:end]))
            end
        end
    else
        return false # if not a number, symbol, or Expr, return false
    end
end

is_outer_affine(s::Symbol) = true
is_outer_affine(r::Real) = true
function is_outer_affine(expr::Expr)
    """
    5*sin(x) - 3*cos(y) is _outer_ affine. It can be re-written:
    5*z - 3*w    where z = sin(x)   w = cos(y)
    """
    if is_number(expr)
        return true
    else
        check_expr_args_length(expr)
        func = expr.args[1]
        if func ∈ [:+, :-]
            # check args
            return all(is_outer_affine.(expr.args[2:end]))
        elseif func == :* # one of args has to be a number
            option1 =  is_number(expr.args[2]) && is_outer_affine(expr.args[3])
            option2 =  is_number(expr.args[3]) && is_outer_affine(expr.args[2])
            return (option1 || option2)
        elseif func == :/ # second arg has to be a number
            return is_number(expr.args[3])
        else
            return false
        end
    end
end

function add_ϵ(points, ϵ)
    `Add ϵ to the y values of all points in a container`
    new_points = []
    for p in points
        push!(new_points, (p[1], p[2] + ϵ))
    end
    return new_points
end

function rewrite_division_by_const(e)
    return e
end
function rewrite_division_by_const(expr::Expr)
    if expr.args[1] == :/ && !is_number(expr.args[2]) && is_number(expr.args[3])
        return :( (1/$(expr.args[3])) * $(expr.args[2]) )
    else
        return expr
    end
end

function find_UB(func, a, b, N; lb=false, digits=nothing, plot=false, existing_plot=nothing, ϵ=0)

    """
    This function finds the piecewise linear upperbound (lowerbound) of
        a given function func in internval [a,b] with N
        sampling points over each concave/convex region.
        see the overest_new.jl for more details.

    Return values are points (UB_points), the min-max closed form (UB_sym)
    as well the lambda function form (UB_eval).
    """

    UB = bound(func, a, b, N; lowerbound=lb, plot=plot, existing_plot=existing_plot)
    UB_points = unique(sort(to_pairs(UB), by = x -> x[1]))
    #println("points: ", UB_points)
    if abs(ϵ) > 0
        UB_points = add_ϵ(UB_points, ϵ) # return new points shifted by epsilon up or down
    end
    #println("points: ", UB_points)
    UB_sym = closed_form_piecewise_linear(UB_points)
    # if !isnothing(digits)
    #     # note this degrades numerical precision, use with care
    #     UB_sym = round_expr(UB_sym)
    # end
    UB_eval = eval(:(x -> $UB_sym))
    return UB_points, UB_sym, UB_eval
end

function find_1d_range(B)
    """
    given a set of points B representing a piecewise linear bound function
        this function finds the range of the bound.
    """

    y_pnts = [point[2] for point in B]
    min_B = minimum(y_pnts)
    max_B = maximum(y_pnts)
    return min_B, max_B
end

function check_expr_args_length(expr)
    if length(expr.args) > 3
        throw(ArgumentError("""
        Operation $(expr.args[1]) has $(length(expr.args)-1) arguments.
        Use parantheses to make this into multiple operations, each with two arguments,
        For example, change (x+y+z)^2 to ((x+y)+z)^2.
                          """))
    end
end

# Think we could use stuff from https://github.com/JuliaIntervals/IntervalArithmetic.jl
# but whose to say if it's better tested?
function find_affine_range(expr, range_dict)
    """
    given a an affine expression expr, this function finds the
        lower and upper bounds of the range.
    """
    @assert is_affine(expr)

    expr isa Symbol ? (return range_dict[expr]) : nothing
    try (return eval(expr), eval(expr)) catch; nothing end # works if expr is a number

    check_expr_args_length(expr)
    all_vars = find_variables(expr)

    if length(all_vars) == 1
        a, b = range_dict[all_vars[1]]
        expr_copy = copy(expr)
        c = eval(substitute!(expr_copy, all_vars[1], a))
        expr_copy = copy(expr)
        d = eval(substitute!(expr_copy, all_vars[1], b))
        return min(c,d), max(c,d)
    end

    func = expr.args[1]
    a, b = find_affine_range(expr.args[2], range_dict)
    c, d = find_affine_range(expr.args[3], range_dict)
    if func == :+
        return (a+c), (b+d)
    elseif func == :-
        return (a-d), (b-c)
    elseif func == :*
        # Does this assume anything about the order of the multiplication being const*variable ?
        try
            a = eval(expr.args[2])
            a > 0 ? (return a*c, a*d) : return a*d, a*c
        catch
            c = eval(expr.args[3])
            c > 0 ? (return c*a, c*b) : (return c*b, c*a)
        end
    elseif func == :/
        c = eval(expr.args[3])
        c > 0 ? (return a/c, b/c) : return( b/c, a/c)
    end
end

mutable struct MyException <: Exception
    var::String
end

function find_range(expr, range_dict)
    """
    Find range of PWL function.
    """
    if is_relu(expr)
        if expr.args[1] == :relu
            inner_expr = expr.args[2]
        elseif expr.args[1] == :max # max with 0
            inner_expr = expr.args[3]
        end
        l,u = find_range(inner_expr, range_dict)
        return [0, max(0, u)]
    elseif is_affine(expr)
        l,u = find_affine_range(expr, range_dict)
        return [l,u]
    elseif is_min(expr)
        x = expr.args[2]
        y = expr.args[3]
        xl, xu = find_range(x, range_dict)
        yl, yu = find_range(y, range_dict)
        return [min(xl,yl), min(xu,yu)]
    else
        throw(MyException("not implemented yet"))
    end
end

function is_relu(expr)
    if length(expr.args) < 2
        return false
    end
    return (expr.args[1] == :max && expr.args[2] == 0.) || expr.args[1] == :relu
end

function is_min(expr)
    return expr.args[1] == :min
end

# todo: pretty sure we can use SymEngine.subs.
# maybe better tested? but subs is also overly complicated...
function substitute!(expr, old, new)

    """
    This function substitutes the old value `old' with the
         new value `new' in the expression expr

    Example: substitute!(:(x^2+1), :x, :(y+1)) = :((y+1)^2+1))
    """

    for (i,arg) in enumerate(expr.args)
       if arg==old
           expr.args[i] = new
       elseif arg isa Expr
           substitute!(arg, old, new)
       end
    end
    return expr
end

function substitute!(expr::Expr, old_list::Vector{Any}, new_list::Array{Any})
    for (k, v) in zip(old_list, new_list)
        substitute!(expr, k, v)
    end
    return expr
end

# function reduce_args_to_2!(expr)

#     """
#     if expr has operations with more than two arguments, this function reduces the arguments to 2
#         Example: reduce_args_to_2!(:(x+y+z)) = (:(x+y)+z))
#                  reduce_args_to_2!(:sin(x*y*z)) = (:(sin((x*y)*z))
#     Modifies expression in place and returns expr as well.
#     """
#     func = expr.args[1]
#     args = expr.args[2:end]
#     larg = length(args)
#     if larg > 2
#         for i=1:div(larg,2)
#            expr.args[i+1] = Expr(:call, func, args[2*i-1], args[2*i])
#         end
#         if isodd(larg)
#             expr.args[div(larg,2)+2] = expr.args[end]
#             expr.args = expr.args[1:div(larg,2)+2]
#         else
#             expr.args = expr.args[1:div(larg,2)+1]
#         end
#     end
#     return expr
# end

∉(e, set) = !(e ∈ set)

function reduce_args_to_2(f::Symbol, arguments::Array)
    """
    reduce_args_to_2(x+y) = x+y
    reduce_args_to_2(x+y+z) = x+(y+z)
    reduce_args_to_2(0.5+y+z) = (0.5+y)+z
    """
    if (length(arguments) <= 2) | (f ∉ [:+, :*])
        e = Expr(:call)
        e.args = [f, map(reduce_args_to_2, arguments)...]
        return e
    elseif length(arguments) ==3 #length(args) = 3 and f ∈ [:+, :*]
        if is_number(arguments[1])
            a = reduce_args_to_2(arguments[2])
            fbc = reduce_args_to_2(f, arguments[1:3 .!= 2])
        else
            a = reduce_args_to_2(arguments[1])
            fbc = reduce_args_to_2(f, arguments[2:3])
        end
        return :($f( $a, $fbc))
    else
        first_half = reduce_args_to_2(f, arguments[1:2])
        second_half = reduce_args_to_2(f, arguments[3:end])
        return reduce_args_to_2(f, [first_half, second_half])
        error("""
            operations with more than 3 arguments are not supported.
            use () to break the arguments.
            Example= x+y+z+w -> (x+y) + (y+z)
            """)
    end
end

reduce_args_to_2(x::Number) = x
reduce_args_to_2(x::Symbol) = x

function reduce_args_to_2(expr::Expr)
    #println(expr)
    f = expr.args[1]
    arguments = expr.args[2:end]
    return reduce_args_to_2(f::Symbol, arguments::Array)
end

# function get_rid_of_division(x)
#     if (x isa Expr) && (x.args[1] == :/) && !is_number(x.args[2])
#         # all moved to division case in binary functions
#         println("*"^30)
#         println("division is $x")
#         println("*"^30)
#         inv_denom = Expr(:call, :/, 1., x.args[3])
#         println("turned to $(Expr(:call, :*, x.args[2], inv_denom))")
#         return Expr(:call, :*, x.args[2], inv_denom)
#     else
#         return x
#     end
# end

function is_number(expr)
    try
        eval(expr)
        return true
    catch
        return false
    end
end

function is_unary(expr::Expr)
    # one for function, one for single argument to function
    return length(expr.args) == 2
end

function is_1d(expr::Expr)
    return length(find_variables(expr)) == 1
end


function is_effectively_unary(expr::Expr)
    # has 3 args, but first is function and one of next 2 is a constant
    f = expr.args[1]
    x = expr.args[2]
    y = expr.args[2]
    #TODO: finish
    # if x is_number(expr) hten.... elseif y is_number(expr) then...
    return false # placeholder
end

is_binary(expr::Expr) = length(expr.args) == 3

function multiply_interval(range, constant)
    S = [range[1]*constant, range[2]*constant]
    return [min(S...), max(S...)]
end

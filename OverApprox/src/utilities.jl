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
             is_affine(:(x + x*z))    = false
    """
    # it is symbol or number
    if typeof(expr) == Symbol
        return true
    else
        try 
            eval(expr)
            return true 
        catch
            nothing
        end
    end

    check_expr_args_length(expr)
    func = expr.args[1]
    func ∈ [:+, :-, :*, :/] ? nothing : (return false)  # only these operations are allowed

    if func == :* # one of args has to be a number
        n_x = 0
        try eval(expr.args[2]) catch; n_x += 1 end
        try eval(expr.args[3]) catch; n_x += 1 end
        n_x > 1 ? (return false) : nothing
    end

    if func == :/ # second arg has to be a number
        try eval(expr.args[3]) catch; (return false) end
    end


    is_affine(expr.args[2]) ? nothing : (return false)
    if length(expr.args) > 2
        is_affine(expr.args[3]) ? nothing : (return false)
    end

    return true
end

function find_UB(func, a, b, N; lb=false, digits=nothing, plot=false, existing_plot=nothing)

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

function reduce_args_to_2!(expr)

    """
    if expr has operations with more than two arguments, this function reduces the arguments to 2
        Example: reduce_args_to_2!(:(x+y+z)) = (:(x+y)+z))
                 reduce_args_to_2!(:sin(x*y*z)) = (:(sin((x*y)*z))
    Modifies expression in place and returns expr as well.
    """
    func = expr.args[1]
    args = expr.args[2:end]
    larg = length(args)
    if larg > 2
        for i=1:div(larg,2)
           expr.args[i+1] = Expr(:call, func, args[2*i-1], args[2*i])
        end
        if isodd(larg)
            expr.args[div(larg,2)+2] = expr.args[end]
            expr.args = expr.args[1:div(larg,2)+2]
        else
            expr.args = expr.args[1:div(larg,2)+1]
        end
    end
    return expr
end
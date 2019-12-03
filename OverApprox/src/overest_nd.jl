include("autoline.jl")
include("overest_new.jl")

# opeartions and functions that are supported in this module
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
    expr isa Expr ? nothing : (return true) # it is symbol or number

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

function find_UB(func, a, b, N; lb=false, digits=nothing)

    """
    This function finds the piecewise linear upperbound (lowerbound) of
        a given function func in internval [a,b] with N
        sampling points over each concave/convex region.
        see the overest_new.jl for more details.

    Return values are points (UB), the min-max closed form (UB_sym)
    as well the lambda function form (UB_eval).
    """

    UB = bound(func, a, b, N; lowerbound=lb, plot=false)
    UB_points = unique(sort(to_pairs(UB), by = x -> x[1]))
    UB_sym = closed_form_piecewise_linear(UB_points)
    if !isnothing(digits)
        UB_sym = round_expr(UB_sym)
    end
    UB_eval = eval(:(x -> $UB_sym))
    return UB_points, UB_sym, UB_eval
end

function find_1d_range(UB)
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

# todo: pretty sure we can use SymEngine.subs. maybe better tested? but subs is also overly complicated...
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


function round_expr(expr; digits=2)
    """ round all numbers in expr to n=digits decimal places. """
    for i=1:length(expr.args)
        if expr.args[i] isa Expr
            expr.args[i] = round_expr(expr.args[i]; digits = digits)
        elseif typeof(expr.args[i]) ∈ [Float16, Float32, Float64]
            expr.args[i] = round(expr.args[i], digits=digits)
        end
    end
    return expr
end

function count_min_max(expr)
    """ count number of min and max in expr"""
    c = [0, 0]
    if expr == :min
        c[1] = 1
    elseif expr == :max
        c[2] = 1
    elseif expr isa Expr
        for arg in expr.args
            c += count_min_max(arg)
        end
    end
    return c
end

function upperbound_expr_compositions(func, arg, N, range_dict, lb_local, lb_global)
    """ this function computes the linear overapproximation of func(arg)
        lb_local=true finds a lowerbound for the inner function; i.e. arg
        lb_glocal=true finds a lowerbound for the outer function; i.e. func
     """
    arg_UB, (a, b) = upperbound_expr(arg; N=N, range_dict=range_dict, lowerbound=lb_local)
    UBfunc, UBfunc_sym, UBfunc_eval = find_UB(func, a, b, N; lb=lb_global)
    UBfunc_composed = substitute!(UBfunc_sym, :x, arg_UB)
    expr_range = find_1d_range(UBfunc)
    return UBfunc_composed, expr_range
end

function reduce_args_to_2!(expr)

    """
    if expr has operations with more than two arguments, this function reduce the arguments to 2
        Example: reduce_args_to_2!(:(x+y+z)) = (:(x+y)+z))
                 reduce_args_to_2!(:sin(x*y*z)) = (:(sin((x*y)*z))
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
end

function upperbound_expr(expr; N=2, lowerbound=false, range_dict=nothing)
    # TODO: what is the return type? an expression?
    """
    This function generates a min-max closed form expression for the upper bound? of
        an expression expr.
    expr can be of any dimensions, and can contain algebraic operations
    as well as nonlinear functions.
    List of supported operations and functions is
    special_oper = [:+, :-, :/, :*, :^]
    special_func = [:exp, :log, :log10,
                    :sin, :cos, :tan,
                    :sinh, :cosh, :tanh,
                    :asin, :acos, :atan,
                    :asinh, :acosh, :atanh]
    if expr is a Symbol, it is returned as is.
    if expr is an affine function, it is returned as is.
    if expr is a 1d function, it is passed into find_UB without
        any additional composition 

    if expr is of the form f(a), where f is one of the special
        functions listed above, and a is an Expr, we return fUB(a) if
        f is increasing, fLB(a) if f is decreasing and
        max(fUB(a), fLB(a)) if f is not monotonic.

    if expr includes a + b, where a and b can be Symbols or Expr,
        we return upperbound_expr(a) + upperbound_expr(b)
    if expr includes a - b, where a and b can be Symbols or Expr,
        we return upperbound_expr(a) - upperbound_expr(b, lowerbound=true)
    if expr includes a * b, where a and b can be Symbols or Expr,
        we return exp(log(a) + log(b))
    if expr includes a / b, where a and b can be Symbols or Expr,
        we return exp(log(a) - log(b))
    if expr includes a^b, where a can be a >1d Expr, and b is a
        number, we treat this as special function composition

    in order to do composition properly, we need to carry an interval for the range of
    each overapproximating function. For 1d functions and affine function, we calculate this
        using separate functions. For function composition, we use the bounds on the
        linear piecewise approximator. For + and -, we appropriately add or subtract the
        bounds of arguments.

    Remark1 : The algorithm only works when operations and functions
        have only two arguments. For example, it does not work
        if expr = :(x+y+z). In this case, we need to use parantheses
        to have multiple operations with two arguments. i.e.
        expr = :((x+y)+z). function reduce_args_to_2!(expr) does this job for us.

    TO DO: I would like to pre-process expr so that we turn all the * and / into log and exp.
        and then we don't need to worry about them.
    """

    # parse expression if it is a string
    if expr isa String
        expr = Meta.parse(expr)
    end

    # if a symbol, return as is
    if expr isa Symbol
        expr_range = range_dict[expr]
        return expr, expr_range
    end

    # checking of expr has more than two arguments
    reduce_args_to_2!(expr)
    check_expr_args_length(expr)

    # checking we have a range for all variables. If not, the default range is [0,1]
    all_vars = find_variables(expr)
    if isnothing(range_dict)
        range_dict = Dict()
        for sym in all_vars
            range_dict[sym] = (0,1)
        end
    else
        for sym in all_vars
            @assert sym ∈ collect(keys(range_dict))
        end
    end

    # if an affine function, return as is
    if is_affine(expr)
        expr_range = find_affine_range(expr, range_dict)
        return expr, expr_range
    end

    # if a 1d function, just find UB and return
    if length(all_vars) == 1
        (a, b) = range_dict[all_vars[1]]
        UBfunc_lambda = SymEngine.lambdify(expr, all_vars)
        UBpoints, UBfunc_sym, UBfunc_eval = find_UB(UBfunc_lambda, a, b, N; lb=lowerbound)
        expr_range = find_1d_range(UBpoints)
        return substitute!(UBfunc_sym, :x, all_vars[1]), expr_range
    end

    # handle function composition
    func = expr.args[1]
    if func in special_func
        func_eval = x -> eval(func)(x)
        UBfunc_composed, expr_range1 = upperbound_expr_compositions(func_eval, expr.args[2], N, range_dict, false, lowerbound)
        if func in increasing_special_func  # if increasing, only consider upperbound.
            return UBfunc_composed, expr_range1
        else # if not monotonic, consider both upper and lower bounds.
            LBfunc_composed, expr_range2 = upperbound_expr_compositions(func_eval, expr.args[2], N, range_dict,  true, lowerbound)
            expr_range = (min(expr_range1[1], expr_range2[1]),
                          max(expr_range1[2], expr_range2[2])) # this is pretty conservative
            if lowerbound
                return :(min($UBfunc_composed, $LBfunc_composed)), expr_range
            else
                return :(max($UBfunc_composed, $LBfunc_composed)), expr_range
            end
        end
    end

    # handle algebraic operations
    if func == :+
        sum1, expr_range1 = upperbound_expr(expr.args[2]; N=N, range_dict=range_dict)
        sum2, expr_range2 = upperbound_expr(expr.args[3]; N=N, range_dict=range_dict)
        expr_range = (expr_range1[1] + expr_range2[1],  expr_range1[2] + expr_range2[2])
        return Expr(:call, :+, sum1, sum2), expr_range
    end

    if func == :-
        sum1, expr_range1 = upperbound_expr(expr.args[2]; N=N, range_dict=range_dict)
        sum2, expr_range2 = upperbound_expr(expr.args[3]; N=N, range_dict=range_dict, lowerbound=true)
        expr_range = (expr_range1[1] - expr_range2[2],  expr_range1[2] - expr_range2[1])
        return Expr(:call, :-, sum1, sum2), expr_range
    end

    # does the shifting of the domain happen?
    if func == :*
        # since log and exp are both increasing, no need to calculate lower bounds)
        log_eval = x -> log(x)
        UBlog1_composed, expr_range1 = upperbound_expr_compositions(log_eval, expr.args[2], N, range_dict, lowerbound, lowerbound)
        UBlog2_composed, expr_range2 = upperbound_expr_compositions(log_eval, expr.args[3], N, range_dict, lowerbound, lowerbound)
        sum_logs = Expr(:call, :+, UBlog1_composed, UBlog2_composed)
        a = expr_range1[1] + expr_range2[1]
        b = expr_range1[2] + expr_range2[2]
        UBexp, UBexp_sym, UBexp_eval = find_UB(x -> exp(x), a, b, N; lb=lowerbound)
        expr_range = find_1d_range(UBexp)
        return substitute!(UBexp_sym, :x, sum_logs), expr_range
    end
    if func == :/
        log_eval = x -> log(x)
        UBlog1_composed, expr_range1 = upperbound_expr_compositions(log_eval, expr.args[2], N, range_dict, lowerbound, lowerbound)
        UBlog2_composed, expr_range2 = upperbound_expr_compositions(log_eval, expr.args[3], N, range_dict, lowerbound, lowerbound)
        sub_logs = Expr(:call, :-, UBlog1_composed, UBlog2_composed)
        a = expr_range1[1] - expr_range2[2]
        b = expr_range1[2] - expr_range2[1]
        UBexp, UBexp_sym, UBexp_eval = find_UB(x -> exp(x), a, b, N; lb=lowerbound)
        expr_range = find_1d_range(UBexp)
        return substitute!(UBexp_sym, :x, sum_logs), expr_range
    end
    if func == :^
        pow = expr.args[3]
        if (pow isa Expr) || (pow isa Symbol)
            throw(ArgumentError("power has to be a number."))
        end
        pow_eval = x -> x^pow
        UBpow_composed, expr_range1 = upperbound_expr_compositions(pow_eval, expr.args[2], N, range_dict, false, lowerbound)
        LBpow_composed, expr_range2 = upperbound_expr_compositions(pow_eval, expr.args[2], N, range_dict,  true, lowerbound)
        expr_range = (min(expr_range1[1], expr_range2[1]),
                      max(expr_range1[2], expr_range2[2])) # this is pretty conservative
        if lowerbound
            return :(min($UBpow_composed, $LBpow_composed)), expr_range
        else
            return :(max($UBpow_composed, $LBpow_composed)), expr_range
        end
    end

    # if it is not returned by now, that means the operation is not supported, hence return 0
    return 0
end

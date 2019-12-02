include("autoline.jl")
include("overest_new.jl")

# opeartions and functions that are supported in this module
special_oper = [:+, :-, :/, :*, :^]
special_func = [:exp, :log, :log10,
                :sin, :cos, :tan,
                :sinh, :cosh, :tanh,
                :asin, :acos, :atan,
                :asinh, :acosh, :atanh]

function to_pairs(B)
    """
    This function converts the output of overest in the form ready
        for closed form generator.
    B is an array of points.
    """
    xp,yp = B
    x = vcat(xp...)
    y = vcat(yp...)
    pairs = collect(zip(x,y))
    return pairs
end

# todo: does this do the same thing as SymEngine.free_symbols or Tomer's autoline.get_symbols ?
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

    Example: is_affine(:(x+1))     = true
             is_affine(:(-x+2y-z)) = true
             is_affine(:(log(x)))  = false
             is_affine(:(x+ xz))   = false
    """

    func = expr.args[1]
    if func ∉ [:+, :-, :*, :/]  # only these operations are allowed
        return false
    end
    if func ∈ [:*, :/]
        if length(find_variables(expr)) > 1
            return false
        end
    end
    for arg in expr.args[2:end]
        if arg isa Expr
            if !(is_affine(arg))
                return false
            end
        end
    end
    return true
end

function find_UB(func, a, b, N; lowerbound=false)

    """
    This function finds the piecewise linear upperbound (lowerbound) of
        a given function func in internval [a,b] with N
        sampling points over each concave/convex region.
        see the overest_new.jl for more details.

    Return values are points (UB), the min-max closed form (UB_sym)
    as well the lambda function form (UB_eval).
    """

    UB = bound(func, a, b, N; lowerbound=lowerbound)
    UB_points = unique(sort(to_pairs(UB), by = x -> x[1]))
    UB_sym = closed_form_piecewise_linear(UB_points)
    UB_eval = eval(:(x -> $UB_sym))
    return UB_points, UB_sym, UB_eval
end

function get_range(B)
    """
    given a set of points B representing a piecewise linear bound function
        this function finds the range of the bound.
    """

    y_pnts = [point[2] for point in B]
    min_B = minimum(y_pnts)
    max_B = maximum(y_pnts)
    return min_B, max_B
end

# todo: pretty sure we can use SymEngine.subs. maybe better tested? but subs is also overly complicated...
function substitute!(expr::Expr, old, new)

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

function upperbound_expr(expr; a=1, b=5, N=2, lowerbound=false)
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
        functions listed above, and a is an Expr, we return
        max(fUB(a), fLB(a)). The reason for including both upperbound
        and lowerbound of a is that f can be increasing or decreasing.
        if lowerbound=true, we return min(fUB(a), fLB(a))

    if expr includes a + b, where a and b can be Symbols or Expr,
        we return upperbound_expr(a) + upperbound_expr(b)
    if expr includes a - b, where a and b can be Symbols or Expr,
        we return upperbound_expr(a) - upperbound_expr(b, lowerbound=true)
    if expr includes a * b, where a and b can be Symbols or Expr,
        we return sign(a)sign(b)exp(log(abs(a)) + log(abs(b)))
    if expr includes a^b, where a can be a >1d Expr, and b is a
        number, we treat this as special function

    in order to do composition properly, we need to carry an interval for the range of
    each overapproximating function. This is performed using get_range(UB)

    Warning: The algorithm only works when operations and functions
        have only two arguments. For example, it does not work
        if expr = :(x+y+z). In this case, we need to use parantheses
            to have multiple operations with two arguments. i.e.
            expr = :((x+y)+z)

    TO DO: implement /. Have to be careful with division by zero.
        and making sure the range of log is not infinite.
    TO DO: carry the range of each when composing.

    """

    # parse expression if it is a string
    if expr isa String
        expr = Meta.parse(expr)
    end

    # if a symbol, return as is
    if expr isa Symbol
        return expr
    end

    # the algorithm only works when operations have only two arguments.
    if length(expr.args) > 3
        throw(ArgumentError("""
        Operation $(expr.args[1]) has $(length(expr.args)-1) arguments.
        Use parantheses to divide this into multiple operations, each with two arguments,
        For example, change (x+y+z)^2 to ((x+y)+z)^2.
                          """))
    end

    # if an affine function, return as is
    if is_affine(expr)
        return expr
    end

    # if a 1d function, just find UB and return
    all_vars = find_variables(expr)
    if length(all_vars) == 1
        func2approx = SymEngine.lambdify(expr, all_vars)
        UBpoints, UBfunc_sym, UBfunc_eval = find_UB(func2approx, a, b, N; lowerbound=lowerbound)
        return substitute!(UBfunc_sym, :x, all_vars[1]) # TODO: this seems brittle to have find_UB always use x as the independent
                                                        # variable.... what if x appears elsewhere? IDK maybe it's fine...
    end

    # BOOKMARK

    # handle function composition
    func = expr.args[1]
    if func in special_func
        # finding the upperbound
        UBfunc, UBfunc_sym, UBfunc_eval = find_UB(x -> eval(func)(x), a, b, N)
        inner_arg_UB = upperbound_expr(expr.args[2])
        UBfunc_composed = substitute!(UBfunc_sym, :x, inner_arg_UB)

        #finding the lowerbound
        LBfunc, LBfunc_sym, LBfunc_eval = find_UB(x -> eval(func)(x), a, b, N; lowerbound=true)
        inner_arg_LB = upperbound_expr(expr.args[2]; lowerbound=true)
        LBfunc_composed = substitute!(LBfunc_sym, :x, inner_arg_LB)
        if lowerbound
            return :(min($UBfunc_composed, $LBfunc_composed))
        else
            return :(max($UBfunc_composed, $LBfunc_composed))
        end
    end

    # handle algebraic operations
    if func == :+
        sum1 = upperbound_expr(expr.args[2])
        sum2 = upperbound_expr(expr.args[3])
        return Expr(:call, :+, sum1, sum2)
    end
    if func == :-
        sum1 = upperbound_expr(expr.args[2])
        sum2 = upperbound_expr(expr.args[3]; lowerbound=true)
        return Expr(:call, :-, sum1, sum2)
    end
    if func == :*
        # since log and exp are both increasing, no need to calculate lower bounds)
        UBlog, UBlog_sym, UBlog_eval = find_UB(x -> log(abs(x)), a, b, N; lowerbound=lowerbound)
        inner_arg1_UB = upperbound_expr(expr.args[2]; lowerbound=lowerbound)
        inner_arg2_UB = upperbound_expr(expr.args[3]; lowerbound=lowerbound)
        UBlog1_composed = substitute!(UBlog_sym, :x, inner_arg1_UB)
        UBlog2_composed = substitute!(UBlog_sym, :x, inner_arg2_UB)
        sum_logs = Expr(:call, :+, UBlog1_composed, UBlog2_composed)

        UBexp, UBexp_sym, UBexp_eval = find_UB(x -> exp(x), a, b, N; lowerbound=lowerbound)
        return substitute!(UBexp_sym, :x, sum_logs)
    end
    # if func == :/
    #     # division by zero has to be taken care off
    #     # range of x can't include 0.
    #     throw(ArgumentError("Division is not yet implemented"))
    # end
    # if func == :^
    #     pow = expr.args[3]
    #     if (pow isa Expr) || (pow isa Symbol)
    #         throw(ArgumentError("power has to be a number"))
    #     end
    #     UBpow, UBpow_sym, UBpow_eval = find_UB(x -> x^pow)
    #     inner_arg_UB = upperbound_expr(expr.args[2])
    #     return substitute!(UBpow_sym, :x, inner_arg_UB)
    # end

    # if it is not returned by now, that means the operation is not supported, hence return 0
    return 0
end

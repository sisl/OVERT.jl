# utilities for overapprox_nd_relational.jl and overest_nd.jl
EVAL_CONSTS = true # some solvers can handle expressions like "log(2)" or "1/6" but others can only handle 0.16666*x 
# When EVAL_CONSTS is true, it will evaluate an expression like "1/6" and turn it into 0.16666 
# TODO: add this functionality to be turned on or off for expressions like log(2)

# list of operators and fucntions that are not counted as variables
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
special_consts = [π]

N_VARS = 0 # number of variables; has to be defined globally.
@debug("N_VARS := 0")

add_var(bound) = add_var()

function add_var()
    # bound.nvars += 1
    # @ is the symbol preceding A in ascii
    #return Symbol('v'*('@'+bound.nvars))

    # the counter has to be a global variable.
    global N_VARS
    N_VARS += 1
    return Symbol("v_$N_VARS")
end
                         
function reset_NVARS()
    global N_VARS
    N_VARS = 0
end

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
function find_variables(expr::Expr)
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
            if !(arg in special_oper) && !(arg in special_func) && !(arg in special_consts)
                all_vars = vcat(all_vars, arg)
            end
        end
    end
    # # use symengine function instead
    # all_vars = Symbol.(free_symbols(Basic(expr))) # this bungles the case of 0*x because Basic makes it just 0 and then the expression looks like it has no free vars
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
        func = expr.args[1]::Symbol
        if func ∉ [:+, :-, :*, :/] # only these operations are allowed
            return false
        else  # func ∈ [:+, :-, :*, :/]
            if func == :* # one of args has to be a number
                option1 =  is_number(expr.args[2]) && is_affine(expr.args[3])
                option2 =  is_number(expr.args[3]) && is_affine(expr.args[2])
                return (option1 || option2)
            elseif func == :/ # second arg has to be a number
                return is_affine(expr.args[2]) && is_number(expr.args[3])
            else # func is + or -
                return all(is_affine.(expr.args[2:end]))
            end
        end
    else
        @error "This case should not be reached. Expression of the wrong type was passed to is_affine()."
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

function add_ϵ(points, ϵ::T where T <: Real)
    `Add ϵ to the y values of all points in a container`
    @debug "Added $ϵ to all points"
    new_points = Tuple{Float64, Float64}[]
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
        if EVAL_CONSTS
            c = eval(1/expr.args[3])
            return :($c * $(expr.args[2]))
        else
            return :( (1/$(expr.args[3])) * $(expr.args[2]) )
        end
    else
        return expr
    end
end

function get_sincos_regions(a,b; offset=0)
    """
    Return inflection points for sin, cos
    """

    n̂_a = ceil((a - offset) / π)
    n̂_b = floor((b - offset) / π)
    n_array = [i for i in n̂_a:n̂_b]

    if length(n_array) == 0 # no inflection points within interval
        @assert sin(a + offset) != 0.0 
        return n_array, sin(a + offset) < 0 # true if convex, false implies concave
    else # greater than 0 length
        return [offset + n*π for n in n_array], nothing # nothing denotes mixed convexity
    end
end

function get_regions_unary(func::Symbol, a, b)
    """
    Return the zeros of the second derivative of function func and/or whether it is convex or not

    d2f_zeros, convex = get_regions(func, a, b)
    """
    if func == :cos
        d2f_zeros, convex = get_sincos_regions(a, b, offset=π/2)
    elseif func == :sin
        d2f_zeros, convex = get_sincos_regions(a, b, offset=0)
    elseif func == :exp
        d2f_zeros, convex = [], true
    elseif func == :log
        @assert a > 0
        d2f_zeros, convex = [], false
    elseif func == :tanh
        if b ≤ 0
            d2f_zeros, convex = [], true
        elseif a ≥ 0
            d2f_zeros, convex = [], false
        else # spans 0
            d2f_zeros, convex = [0], nothing
        end
    else
        d2f_zeros, convex = nothing, nothing
    end

    return d2f_zeros, convex
end

function division_d2f_regions(e, arg, a, b)
    # function defined over the interval [a,b]
    c = e.args[2] 
    @assert is_number(c)
    @assert (a <= b)
    if eval(c) > 0
        if a > 0
            d2f_zeros, convex = [], true
        elseif b < 0
            d2f_zeros, convex = [], false
        else
            error("ERROR: interval ["*string(a)*","*string(b)*"] straddles zero for function c/x.The function is discontinuous and cannot be bounded.")
        end     
    elseif eval(c) < 0
        if a > 0
            d2f_zeros, convex = [], false
        elseif b < 0
            d2f_zeros, convex = [], true
        else
            error("ERROR: interval ["*string(a)*","*string(b)*"] straddles zero for function c/x.The function is discontinuous and cannot be bounded.")
        end
    end
    return d2f_zeros, convex
end

function exponent_d2f_regions(e, arg, a, b)
    if is_number(e.args[2]) # c^x 
        @assert eval(e.args[2]) > 0 # only real valued over real arguments for c > 0
        d2f_zeros, convex = [], true
    elseif is_number(e.args[3]) # x^c (polynomials)
        x = e.args[2]
        c = e.args[3]
        # a few cases
        # 1) c is fractional. only valid for x >= 0. check. and convex (increasing) no inflection points
        if (c % 1) != 0
            @assert a >= 0 # ensures whole interval is > 0
            if (c > 0) && (c < 1)
                d2f_zeros, convex = [], false
            else
                d2f_zeros, convex = [], true
            end
        # 2) c is odd (3, 5, 7): inflection point at zero may be applicable. convex for x>0, concave for x<0
        elseif (c % 2) == 1
            if a > 0
                d2f_zeros, convex = [], true
            elseif b < 0
                d2f_zeros, convex = [], false
            else # interval overlaps 0
                d2f_zeros, convex = [0], nothing
            end
        # 3) c is even (2, 4, 6): convex, no inflection points 
        elseif (c % 2) == 0
            if c < 0
                # discontinuous at zero for negative even numbers
                @assert ((a > 0) || (b < 0)) && (a ≤ b)
            end
            d2f_zeros, convex = [], true
        end
    else
        # TODO: handle x^y
        error("Expression type not handled.")
    end
    return d2f_zeros, convex
end

function get_regions_1arg(e::Expr, arg::Symbol, a, b)
    # TODO:
    # check if c/x or c^x or x^c 
    # multiplication between two variables is expanded using log and exp
    # TODO: add case to catch x^1 which is affine
    func = e.args[1]
    if func == :/
        d2f_zeros, convex = division_d2f_regions(e, arg, a, b)
    elseif func == :^
        d2f_zeros, convex = exponent_d2f_regions(e, arg, a, b)
    else
        @debug "Expression not supported."
        d2f_zeros, convex = nothing, nothing
    end
    @debug "Expression $e is convex: $convex and has inflection points: $d2f_zeros"
    return d2f_zeros, convex
end

function find_UB(func, a, b, N; lb::Bool=false, plot::Bool=false, ϵ::Float64=0.0, d2f_zeros::Union{Nothing, Vector{Any}}=nothing, convex::Union{Nothing, Bool}=nothing)

    """
    This function finds the piecewise linear upperbound (lowerbound) of
        a given function func in internval [a,b] with N
        sampling points over each concave/convex region.
        see the overest_new.jl for more details.

    Return values are points (UB_points), the min-max closed form (UB_sym)
    as well the lambda function form (UB_eval).
    """
    UB = bound(func, a, b, N; lowerbound=lb, d2f_zeros=d2f_zeros, convex=convex, plot=plot)
    UB_points = unique(sort(to_pairs(UB), by = x -> x[1]))
    # if !lb
    #     println("Is upper bound.")
    # else
    #     println("Is lower bound.")
    # end
    #println("Max of points: ", maximum(UB_points))
    if abs(ϵ) > 0
        UB_points = add_ϵ(UB_points, ϵ) # return new points shifted by epsilon up or down
    end
    #println("Max of points after ϵ adjustment: ", maximum(UB_points))
    UB_sym = closed_form_piecewise_linear(UB_points)
    #UB_eval = SymEngine.lambdify(:(x -> $UB_sym), [:x])
    return UB_points::Vector{Tuple{Float64, Float64}}, UB_sym #, UB_eval
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
    @debug "$expr is affine"

    @debug "testing if $expr can be eval'ed to a number"
    try (return eval(expr), eval(expr)) catch; nothing end # works if expr is a number
    @debug "testing if $expr is a symbol"
    expr isa Symbol ? (return range_dict[expr]) : nothing

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
    if is_affine(expr)
        l,u = find_affine_range(expr, range_dict)
        return [l,u]
    elseif is_relu(expr)
        @debug expr " is relu"
        if expr.args[1] == :relu
            inner_expr = expr.args[2]
        elseif expr.args[1] == :max # max with 0
            if expr.args[2] == 0.0
                inner_expr = expr.args[3]
            elseif expr.args[3] == 0.0
                inner_expr = expr.args[2]
            end
        end
        l,u = find_range(inner_expr, range_dict)
        @debug "l = " l ", u = " u
        return [0, max(0, u)]
    elseif expr.args[1] == :min
        x = expr.args[2]
        y = expr.args[3]
        xl, xu = find_range(x, range_dict)
        yl, yu = find_range(y, range_dict)
        return [min(xl,yl), min(xu,yu)]
    elseif expr.args[1] == :max
        x = expr.args[2]
        y = expr.args[3]
        xl, xu = find_range(x, range_dict)
        yl, yu = find_range(y, range_dict)
        return [max(xl,yl), max(xu,yu)]
    else
        throw(MyException("not implemented yet"))
    end
end

function is_relu(expr)
    if length(expr.args) < 2
        return false
    end
    c1 = (expr.args[1] == :max && expr.args[2] == 0.)
    c2 = length(expr.args) > 2 ? (expr.args[1] == :max && expr.args[3] == 0.) : false
    c3 = expr.args[1] == :relu
    return c1 || c2 || c3 
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

is_binary(expr::Expr) = length(expr.args) == 3

function multiply_interval(range, constant)
    S = [range[1]*constant, range[2]*constant]
    return [min(S...), max(S...)]
end

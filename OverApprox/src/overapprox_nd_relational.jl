include("autoline.jl")
include("overest_new.jl")
include("utilities.jl")
using SymEngine

function overapprox_nd(expr, range_dict; nvars=0, N=3)
    # returns: (output_variable::Symbol, range_of_output_variable::Tuple, [list of equalities], [list of inequalities])

    # base cases
    if expr isa Symbol
        return (expr, range_dict[expr], nvars, [], [])
    elseif is_number(expr)
        c = eval(expr)
        return (c, (c,c), nvars, [], [])
    elseif is_affine(expr)
        newvar, nvars = get_new_var(nvars)
        return (newvar, find_affine_range(expr, range_dict), [:($newvar = $expr)], [])
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
        else # is binary (assume all multiplication has been processed out)
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
            ineq_list = [ineq_list_x..., ineq_list_y]
            # handle outer function f
            z, nvars = get_new_var(nvars)
            if f == :+
                push!(eq_list, :(z = $x + $y))
                sum_range = (xrange[1] + yrange[1], xrange[2] + yrange[2])
                return (z, sum_range, nvars, eq_list, ineq_list)
            elseif f == :-
                push!(eq_list, :(z = $x - $y))
                diff_range = (xrange[1] - yrange[2],  xrange[2] - yrange[1])
                return (z, diff_range, nvars, eq_list, ineq_list)
            else
                error("unimplemented")
                return (:0, (0,0), 0, [], []) # for type stability, maybe?
            end
        end
    end
end

function apply_fx(f, a)
    substitute!(f, :x, a)
end

function bound_unary_function(f, x, xrange, N, eq_list, ineq_list, nvars)   
    ## create upper and lower bounds
    eval(:(fun = $f))
    UBpoints, UBfunc_sym, UBfunc_eval = find_UB(fun, xrange[1], xrange[2], N; lb=false)
    fUBrange = find_1d_range(UBpoints)
    LBpoints, LBfunc_sym, LBfunc_eval = find_UB(fun, xrange[1], xrange[2], N; lb=true)
    fLBrange = find_1d_range(LBpoints)
    ## create new vars for these expr, equate to exprs, and add them to equality list 
    # e.g. y = fUB(x), z = fLB(x)
    UBvar, nvars = get_new_var(nvars)
    push!(eq_list, :($UBvar = $(apply_fx(UBfunc_sym, x))))
    LBvar, nvars = get_new_var(nvars)
    push!(eq_list, :($LBvar = $(apply_fx(LBfunc_sym, x))))
    OArange = (min(fLBrange...,fUBrange...), max(fLBrange..., fUBrange...))
    return LBvar, UBvar, OArange, eq_list, ineq_list, nvars
end

function get_new_var(nvars)
    nvars += 1
    # @ is the symbol preceding A in ascii
    return Symbol('@'+nvars), nvars
end

function is_number(expr::Expr)
    try 
        eval(expr)
        return true
    catch
        return false
    end
end

function is_unary(expr::Expr)
    if length(expr.args) == 2
        # one for function, one for single argument to function
        return true
    else
        return false
    end
end

is_binary(expr::Expr) = length(expr.args) == 3
    
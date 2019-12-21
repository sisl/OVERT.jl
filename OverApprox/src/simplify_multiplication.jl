# simplify multiplication
include("overapprox_nd_relational.jl")

# top level function
function rewrite_multiplication(expr, range_dict)
    if (expr isa Symbol) | is_number(expr)
        return (expr, range_dict)
    else # is a function call containing variables
        recurse_on_call(expr, range_dict)
end

function recurse_on_call(expr, range_dict)
    if is_unary(expr)
        return recurse_unary_function(expr, range_dict)
    elseif is_binary(expr)
        return recurse_binary_function(expr, range_dict)
    else
        error("unimplemented")
        return (:(), Dict())
    end
end

function recurse_unary_function(expr, range_dict)
    f = expr.args[1]
    xexpr= expr.args[2]
    if contains_mul(xexpr)
        new_xexpr, new_range_dict = rewrite_multiplication(xexpr, range_dict)
        rwn_expr = :($f($new_xexpr) )
        return (rwn_expr, new_range_dict)
    else # does not have a multiplication operation that needs to be re-written
        return (expr, range_dict)
    end 
end

function recurse_binary_function(expr, range_dict)
    f = expr.args[1]
    if !(f == :*) # f not multiplication
        return recurse_binary_args(expr, range_dict) 
    else # if f IS multiplication
       return recurse_mul(expr, range_dict)
    end
end

function recurse_binary_args(expr, range_dict)
    f = expr.args[1]
    xexpr = expr.args[2]
    yexpr = expr.args[3]
    # recurse on args
    if contains_mul(xexpr)
        xexpr, range_dict = rewrite_multiplication(xexpr, range_dict)
    end
    if contains_mul(yexpr)
        yexpr, range_dict = rewrite_multiplication(yexpr, range_dict)
    end
    return (:($f($xexpr, $yexpr)), range_dict)
end


function recurse_mul(expr, range_dict)
    f = expr.args[1]
    xexpr = expr.args[2]
    yexpr = expr.args[3]
    x_is_number = is_number(xexpr)
    y_is_number = is_number(yexpr)
    if (x_is_number | y_is_number) # if either are just scalars
        # rewrite each of those, and continue on
        if x_is_number & !y_is_number
            return recurse_scalar_times_var(xexpr, yexpr, range_dict)
        elseif y_is_number & !x_is_number
            return recurse_scalar_times_var(yexpr, xexpr, range_dict)
        end
    else # neither are scalars
        # first recurse on args
        rwn_expr, new_range_dict = recurse_binary_args(expr, range_dict)
        # then actually do the multiplication expansion
        return expand_multiplication(rwn_expr, new_range_dict)
    end
end

function recurse_scalar_times_var(scalar, var, range_dict)
    new_var, new_range_dict = rewrite_multiplication(var, range_dict)
    rwn_expr = :($scalar * $new_var)
    return (rwn_expr, new_range_dict)
end

function expand_multiplication(expr, range_dict)
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
    # Thought: this pre-processing can't be done "ahead of time" UNLESS you want to make nonlinear
    # optimization a part of the routine. BECAUSE this re-writing uses the range of a variable or 
    # expression. 
    f = expr.args[1]
    xexpr = expr.args[2]
    yexpr = expr.args[3]
    @assert f == :*
    nvars = ?
    a, nvars = get_new_var(nvars)
    
end



        


    
            
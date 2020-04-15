using HDF5

"""
extend simplify to Symbols and number
"""
simplify(sym::Symbol) = sym
simplify(sym::Number) = sym


"""
contain_max:
returns true if the expression contains any min or max operation
"""
contains_max(expr::Symbol) = expr in [:min, :max]
contains_max(expr::Number) = false
function contains_max(expr::Expr)
    if is_number(expr)
        return false
    else
        for arg in expr.args
            if contains_max(arg)
                return true
            end
        end
    end
    return false
end

"""
if expr includes any nested min or max, rename them to new variables.
ouputs:
 -new_expr: expr with all inner min and max operations are renamed
to new variables.
 -all_constraint_list: list of all additional variable assignment introduced
"""

rename_all_max(expr::Number, all_constraint_list::Union{Vector{Any}, Array{Any,1}}) = expr, all_constraint_list
rename_all_max(expr::Symbol, all_constraint_list::Union{Vector{Any}, Array{Any,1}}) = expr, all_constraint_list
function rename_all_max(expr::Expr, all_constraint_list::Union{Vector{Any}, Array{Any,1}})
    f = expr.args[1]
    if f in [:max, :min]
        args = expr.args[2:end]
        @assert length(args) == 2
        x, all_constraint_list = rename_all_max(args[1], all_constraint_list)
        y, all_constraint_list = rename_all_max(args[2], all_constraint_list)
        if !is_number(x) && !is_number(y)
            x_var = add_var()
            push!(all_constraint_list, :($x_var = $x))

            y_var = add_var()
            push!(all_constraint_list, :($y_var = $y))

            new_expr = Expr(:call)
            new_expr.args = [f, x_var, y_var]
            #out_var = add_var()
            #push!(all_constraint_list, :($out_var = $new_expr))
        elseif !is_number(x) # y is number
            x_var = add_var()
            push!(all_constraint_list, :($x_var = $x))

            new_expr = Expr(:call)
            new_expr.args = [f, x_var, y]
            # out_var = add_var()
            # push!(all_constraint_list, :($out_var = $new_expr))
        elseif !is_number(y) # x is number
            y_var = add_var()
            push!(all_constraint_list, :($y_var = $y))


            new_expr = Expr(:call)
            new_expr.args = [f, x, y_var]
            # out_var = add_var()
            # push!(all_constraint_list, :($out_var = $new_expr))
        else
            error("can't have two numbers.")
        end
    else
        new_expr = Expr(:call)
        new_args = []
        for arg in expr.args
            arg_processed, all_constraint_list = rename_all_max(arg, all_constraint_list)
            push!(new_args, arg_processed)
        end
        new_expr.args = new_args
    end

    return new_expr, all_constraint_list
end

"""
for any equality expression, marabou_friendify_equality assigns new
    variables to all nested max and min expressions.
    The output is a list of simple expressions that is equivalent to the
    original expression, except with no nested expressions.
    The output constraint are in either of the following forms.
    v1 = a*v2 + b
    v1 = a*min(v2, v3)
"""
function marabou_friendify_equality(expr::Expr)
    if length(expr.args) == 2 # this is = constraint
        left_expr = expr.args[1]
        right_expr = expr.args[2]
    elseif length(expr.args) == 3 # this is == constraint
        left_expr = expr.args[2]
        right_expr = expr.args[3]
    else
        error("equality constrain has to have only 2 or 3 args.")
    end

    new_right_expr, min_max_cons = rename_all_max(right_expr, [])

    if length(min_max_cons) > 0
        final_expr = Expr(:call)
        final_args = [:+]
        for arg in new_right_expr.args[2:end]
            new_var = add_var()
            push!(min_max_cons, :($new_var = $arg))
            push!(final_args, new_var)
        end
        final_expr.args = final_args
    else
        final_expr = new_right_expr
    end
    push!(min_max_cons, :($left_expr = $final_expr))
    return min_max_cons
end



function marabou_friendify!(bound::OverApproximation)
    """
    run marabou_friendify_equality for all equality constraint in bound.
    """
    all_eq = []
    for eq in bound.approx_eq
       new_eq = marabou_friendify_equality(eq)
       all_eq = vcat(all_eq, new_eq)
    end
    bound.approx_eq = all_eq
    return bound
end

linear_1d_expr(expr::Symbol) = true
function linear_1d_expr(expr::Expr)
    """
    returns true if expr is a 1d linear expression like
        expr = ax +b
        expr = -x
        expr = bx
    """
    if length(expr.args) == 2 && expr.args[1] == :- && expr.args[2] isa Symbol
        return true
    end
    if (length(expr.args) != 3)
        return false
    elseif !(xor(is_number(expr.args[2]), is_number(expr.args[3])))
        return false
    elseif expr.args[1] ∉ [:*, :-, :+]
        return false
    elseif contains_max(expr)
        return false
    else
        return true
    end
end


"""
parse_linear_expr
find x, a and b in a linear 1d expression.

# expr = ax + b, return (x, a, b)
# expr = ax - b, return (x, a, -b)
# expr = b - ax, return (x, -a, b)
# expr = x + b, return  (x, 1, b)
# expr = 2x, return  (x, 2, 0)
# expr = -x, return (x, -1, 0)
# expr = :((1/2)*x -1), return (:x, 1/2, -1) # notice the second coefficient is number and not expr.
"""
parse_linear_expr(expr::Symbol) = expr, 1, 0
function parse_linear_expr(expr::Expr)
    @assert linear_1d_expr(expr)

    if length(expr.args) == 2  # this takes care of -x
        @assert expr.args[1] == :-
        return expr.args[2], -1, 0
    end

    @assert expr.args[1] in [:+, :-, :*]
    @assert length(expr.args)  == 3
    @assert xor(is_number(expr.args[2]), is_number(expr.args[3]))
    if expr.args[1] in [:+, :-]
        # turn subtraction to summation
        if expr.args[1] == :-
            expr.args[1] = :+
            expr.args[3] = simplify(:(-$(expr.args[3])))
        end

        b = is_number(expr.args[2]) ? expr.args[2] : expr.args[3]
        ax = is_number(expr.args[2]) ? expr.args[3] : expr.args[2]
        b = eval(b) # makes expr to number

        if ax isa Symbol
            return (ax, 1, b)
        elseif ax isa Expr
            x, a, b0 = parse_linear_expr(ax)
            @assert b0 == 0
            return (x, a, b)
        else
            error("something is wrong")
        end
    else  # expr.args[1] == :*
        a = is_number(expr.args[2]) ? expr.args[2] : expr.args[3]
        a = eval(a) # makes expr to number
        x = is_number(expr.args[2]) ? expr.args[3] : expr.args[2]
        return (x, a, 0)
    end
end


function bound_2_txt(bound::OverApproximation, file_name::String; state_vars=[], control_vars=[])
    """
    this function returns all equality and inequality constraints
    in forms of three lists and saves them in .h5 file.

    aq_list contains lists with this format:
          [[v1, v2, ...], [a1, a2, ...], b]
    for example, if aq_list[1] = [[:x, :y, :z], [1, -1, 2], 3],
    then, :(x - y + 2z = 3)

    min_list contains lists with this format:
          [[left, right_1, right_2], [c1, c2]]
    for example, if min_list[1] = [[:x, :y, :z], [2, 3]]
    then, x = min(2*y, 3*z)

    ineq_list contains lists with this format:
              [left, right]
    for example, if ineq_list[1] = [:x, :y],
    then, x ≦ y
    """

    ineq_list = []
    eq_list   = []
    min_list  = []
    max_list  = []

    # process equalities
    for eq in bound.approx_eq

        # preprocess eq. eq has to be of form :(v1 = v2). v1 must be a Symbol.
        @assert length(eq.args) == 2
        @assert eq.args[1] isa Symbol

        # simplify simplifies the numerical expressions like
        #:(x+ (2-1)) = :(x+1) or :(2(x-1)) = :(2x-2)
        #left_arg = simplify(eq.args[1])
        left_arg = eq.args[1]
        rite_arg = simplify(eq.args[2])

        # take care of cases like :(-(x-1)), rewrite :(-1*(x-1))
        if length(rite_arg.args) == 2 && rite_arg.args[1] == :-
            rite_arg = :(-1*$(rite_arg.args[2]))
        end

        # make sure ##args are consistent.
        @assert length(rite_arg.args) >= 3

        f = rite_arg.args[1]
        if linear_1d_expr(rite_arg) # parse cases like "(y=3x-2)"
            v, a, b = parse_linear_expr(rite_arg)
            push!(eq_list, [[string(left_arg), string(v)], [1, -a], b])
        else
            if f == :+ # parse cases like "(z = 2x+3y or w = 2x+3y+4z)"
                sym_list = [string(left_arg)]
                a_list = [1]
                for e in rite_arg.args[2:end]
                    v, a, b = parse_linear_expr(e)
                    @assert b == 0
                    push!(sym_list, string(v))
                    push!(a_list, -a)
                end
                push!(eq_list, [sym_list, a_list, 0])
            elseif f == :- # parse cases like "(z = 2x-3y)". it does NOT support (w = 2x-3y-4z), which should never happen.
                @assert(length(rite_arg.args) == 3)
                @assert(linear_1d_expr(rite_arg.args[2]))
                @assert(linear_1d_expr(rite_arg.args[3]))

                v1, a1, b1 = parse_linear_expr(rite_arg.args[2])
                v2, a2, b2 = parse_linear_expr(rite_arg.args[3])
                @assert(b1 == b2 == 0)
                push!(eq_list, [[string(left_arg), string(v1), string(v2)], [1, -a1, a2], 0])
            elseif f == :*  # parse cases like (z = 2*min(x,2y))

                # one argument has to be a number.
                @assert xor(is_number(rite_arg.args[2]), is_number(rite_arg.args[3]))
                c = is_number(rite_arg.args[2]) ? rite_arg.args[2] : rite_arg.args[3]
                minx = is_number(rite_arg.args[2]) ? rite_arg.args[3] : rite_arg.args[2]

                # the non-number argument has to be a min or max expression
                @assert minx.args[1] in [:min, :max]

                # if c > 0, z = c*min(x, y) = min(cx, cy)
                # if c < 0, z = c*min(x, y) = max(cx, cy)
                # if c > 0, z = c*max(x, y) = max(cx, cy)
                # if c < 0, z = c*max(x, y) = min(cx, cy)
                if c > 0
                    if minx.args[1] == :min
                        push!(min_list, [[string(left_arg), string(minx.args[2]), string(minx.args[3])], [c, c]])
                    else
                        push!(max_list, [[string(left_arg), string(minx.args[2]), string(minx.args[3])], [c, c]])
                    end
                else
                    if minx.args[1] == :min
                        push!(max_list, [[string(left_arg), string(minx.args[2]), string(minx.args[3])], [c, c]])
                    else
                        push!(min_list, [[string(left_arg), string(minx.args[2]), string(minx.args[3])], [c, c]])
                    end
                end
            elseif f == :min # parse cases like z = min(x,y)
                push!(min_list, [[string(left_arg), string(rite_arg.args[2]), string(rite_arg.args[3])], [1, 1]])
            elseif f == :max # parse cases like z = max(x,y)
                push!(max_list, [[string(left_arg), string(rite_arg.args[2]), string(rite_arg.args[3])], [1, 1]])
            else
                error("the operation $f is not recognized.")
            end
        end
    end

    # process inequalities
    for ineq in bound.approx_ineq
        @assert ineq.args[1] == :≦
        @assert length(ineq.args) == 3
        push!(ineq_list, [[string(ineq.args[2])], [string(ineq.args[3])]])
    end

    # write input and output variables to file
    if state_vars isa Array
        state_vars_txt = [string(v) for v in state_vars]
    else
        state_vars_txt = [string(state_vars)]
    end

    if control_vars isa Array
        control_vars_txt = [string(v) for v in control_vars]
    else
        control_vars_txt = [string(control_vars)]
    end

    if bound.output isa Array
        output_vars_txt = [string(v) for v in bound.output]
    else
        output_vars_txt = [string(bound.output)]
    end

    h5write(file_name, "vars/states", state_vars_txt)
    h5write(file_name, "vars/controls", control_vars_txt)
    h5write(file_name, "vars/outputs", output_vars_txt)

    # write constrains to file
    for (i, eq) in enumerate(eq_list)
        h5write(file_name, "eq/v$i", eq[1])
        h5write(file_name, "eq/c$i", eq[2])
        h5write(file_name, "eq/b$i", eq[3])
    end
    for (i, eq) in enumerate(min_list)
        h5write(file_name, "min/v$i", eq[1])
        h5write(file_name, "min/c$i", eq[2])
    end
    for (i, eq) in enumerate(max_list)
        h5write(file_name, "max/v$i", eq[1])
        h5write(file_name, "max/c$i", eq[2])
    end
    for (i, eq) in enumerate(ineq_list)
        h5write(file_name, "ineq/l$i", eq[1])
        h5write(file_name, "ineq/r$i", eq[2])
    end
    return eq_list, min_list, max_list, ineq_list
end

using HDF5
include("overt_utils.jl")

"""
Note:
    - In Overt all max expressions are max(0, something), so they all
turned into relu. Perhaps could've been more explicit to have relu instead of max. < I agree.
    - Marabou accepts MaxConstraints (and not min), so all min expressions are turend
into max expressions.
    -z = min(x,y) is equivalent to {z2 = min(x2, y2), z2=-z, x2=-x, y2=-x}
    - Overt can be rewritten to generate min of max's (as opposed to max of min's).
    That could possibly reduce number of variables introduced.
"""

"""
----------------------------------------------
General structures
----------------------------------------------
"""

" e.g. :(z=max(x,y)) -> MaxConstraint([:x, :y], :z) "
mutable struct MaxConstraint
    varsin::Array{Symbol, 1}
    varout::Symbol
end

""" e.g. :(z=relu(x)) -> ReluConstraint(:x, :z) """
mutable struct ReluConstraint
    varin::Symbol
    varout::Symbol
end

""" e.g. :(z=a*x+b*y+c*w+d) -> EqConstraint([:z, :x, :y, :w], [1, -a, -b, -c], d) """
mutable struct EqConstraint
    vars::Array{Symbol, 1}
    coeffs::Array{Real, 1}
    scalar::Real
end

""" e.g. :(x ≦ z) -> IneqConstraint(:x, :z) """
mutable struct IneqConstraint
    varleft::Symbol
    varrite::Symbol
end

"""Nonlinear constraint: left R right ----> :y := :(5*sin(x)) , indep_var = :x """
mutable struct NLConstraint
    left::Symbol
    R::Symbol
    right::Expr
    indep_var::Symbol
end

"""Piecewise linear or linear constraints"""
PWLoLConstraint = Union{MaxConstraint, ReluConstraint, EqConstraint, IneqConstraint}

mutable struct OverApproximationParser
    max_list::Array{MaxConstraint, 1}
    relu_list::Array{ReluConstraint, 1}
    eq_list::Array{EqConstraint, 1}
    ineq_list::Array{IneqConstraint, 1}
    # all above lists correspond to relations that make up the bound
    # all below lists refer to the original smooth nonlinear function
    true_L_list::Array{PWLoLConstraint,1} # list of linear or PWL constraints
    true_NL_list::Array{NLConstraint} # list of all nonlinear consraints
    ranges
end

OverApproximationParser() = OverApproximationParser(Array{MaxConstraint, 1}(),
                                                    Array{ReluConstraint, 1}(),
                                                    Array{EqConstraint, 1}(),
                                                    Array{IneqConstraint, 1}(),
                                                    Array{PWLoLConstraint, 1}(),
                                                    Array{NLConstraint, 1}(),
                                                    Dict(),
                                                    )


Base.print(oAP::OverApproximationParser) = print_overapproximateparser(oAP)
Base.display(oAP::OverApproximationParser) = print_overapproximateparser(oAP)
function print_overapproximateparser(oAP::OverApproximationParser)
    for eq in oAP.relu_list
        println("$(eq.varout) = relu($(eq.varin))")
    end
    for eq in oAP.max_list
        println("$(eq.varout) = max($(eq.varsin[1]), $(eq.varsin[2]))")
    end
    for eq in oAP.eq_list
        str = ""
        for (var, coeff) in zip(eq.vars, eq.coeffs)
            str *= " $coeff*$var +"
        end
        str = chop(str)
        str *= "= $(eq.scalar)"
        println(str)
    end
    for ineq in oAP.ineq_list
        println("$(ineq.varleft) ≦ $(ineq.varrite)")
    end

    println("True function pieces:")
    for item in oAP.true_L_list
        println(item)
    end

    for item in oAP.true_NL_list
        println(item)
    end

end


#Base.add_sum(x_oAP::OverApproximationParser, y_oAP::OverApproximationParser) = add_overapproximateparser(x_oAP,y_oAP)
function add_overapproximateparser(x_oAP::OverApproximationParser, y_oAP::OverApproximationParser)
    z_oAP = OverApproximationParser()
    z_oAP.relu_list = vcat(x_oAP.relu_list, y_oAP.relu_list)
    z_oAP.max_list = vcat(x_oAP.max_list, y_oAP.max_list)
    z_oAP.eq_list = vcat(x_oAP.eq_list, y_oAP.eq_list)
    z_oAP.ineq_list = vcat(x_oAP.ineq_list, y_oAP.ineq_list)
    z_oAP.true_L_list = vcat(x_oAP.true_L_list, y_oAP.true_L_list)
    z_oAP.true_NL_list = vcat(x_oAP.true_NL_list, y_oAP.true_NL_list)
    # TODO: add range_dicts together
    return z_oAP
end

"""
----------------------------------------------
High level functions
----------------------------------------------
"""

"""
parse_bound: parses every eq and ineq in bound.
    it assume eq's are one of these three forms:
        - a*x + b*y + c*z + ... = d
        - a*max(0, expr1) + b*max(0, expr2) + ... + c*max(0, expr3) = var where
            -- expr's can be e*x+f or min(g*x+h, i*x+j)
            -- var is a Symbol
    it assume ineq's are in the following form:
        - v1 ≦ v2, where v1 and v2 are both symbols
"""
function parse_bound(bound::OverApproximation,
                     bound_parser::OverApproximationParser)

    bound_parser.ranges = bound.ranges
    # parsing equalities
    i=0
    for eq in bound.approx_eq
        parse_eq(eq, bound_parser)
        println("parsed ", i, "th eq constraint")
        i+=1
    end

    # parsing inequalities
    for ineq in bound.approx_ineq
        parse_ineq(ineq, bound_parser)
    end

    # parsing equations that represent the true function
    tru_func_parser = OverApproximationParser() # a temporary holder
    ordered_fun_eq = sort(collect(bound.fun_eq), by=x->parse(Int, string(x[1])[3:end]) ) # sorts pairs of fun_eq dict by keys, assuming that variables are of the form "v_XXX" where XXX is a number of any length
    for func in ordered_fun_eq
        eq = :($(func[1]) == $(func[2]) )
        parse_eq(eq, tru_func_parser)
    end
    # put constraints from tru_func_parser into bound_parser in the right places
    bound_parser.true_NL_list = tru_func_parser.true_NL_list
    [push!(bound_parser.true_L_list, c) for c in tru_func_parser.eq_list]
    [push!(bound_parser.true_L_list, c) for c in tru_func_parser.ineq_list]
    [push!(bound_parser.true_L_list, c) for c in tru_func_parser.max_list]
    [push!(bound_parser.true_L_list, c) for c in tru_func_parser.relu_list]
end

function parse_eq(expr::Expr, bound_parser::OverApproximationParser)
    println(expr)
    fix_negation!(expr)
    assert_expr(expr, "eq")
    if is_max_expr(expr)
        parse_max_expr(expr, bound_parser)
    elseif is_min_expr(expr)
        parse_min_expr(expr, bound_parser)
    elseif is_linear_expr(expr)
        parse_linear_expr(expr, bound_parser)
    else
        parse_nonlinear_expr(expr, bound_parser)
    end

    # each expression is tested here with random inpus.
    for i = 1:2 #10
        @assert test_random_input(expr, bound_parser)
    end
end

function parse_ineq(expr::Expr, bound_parser::OverApproximationParser)
    assert_expr(expr, "ineq")
    push!(bound_parser.ineq_list, IneqConstraint(expr.args[2], expr.args[3]))
end

""" this function turns :(w == -(x+1)) to :(w == -1(x+1)) """
function fix_negation!(expr::Expr)
    if length(expr.args[3].args) == 2
        if expr.args[3].args[1] == :-
            arg = expr.args[3].args[2]
            expr.args[3].args = [:*, -1, arg]
        end
    end
    return expr
end

"""
----------------------------------------------
low level functions: linear constraints
----------------------------------------------
"""

""" find if this expr is linear"""
# TODO: Equivalence classes have been confused here.
# This function takes equations, not expressions
# (the type is Expr but it expects an equation).
function is_linear_expr(expr::Expr)
    # x == 6*x + 7*y - 43
    return is_affine(expr.args[3])
end

""" parse terms like z = a*x + b*y + c """
# TODO note: this code mixes equivalence classes which is confusing and makes
# maintenance more difficult. E.g. this function is called "parse linear expr"
# but it also makes the assumption that this is an equality constraint.
# This makes the code brittle (prone to breaking due to small changes) because
# it relies on _implicit properties_ that are not well documented.
function parse_linear_expr(expr::Expr, bound_parser::OverApproximationParser)
    #make sure the expr is of form :(z == a*x + b) or :(z == a*x + b*y)
    assert_expr(expr, "linear")
    var_list, coeff_list, scalar = get_linear_coeffs(expr.args[3])
    var_list = vcat(var_list, expr.args[2])
    coeff_list = vcat(-coeff_list, 1.)
    push!(bound_parser.eq_list, EqConstraint(var_list, coeff_list, scalar))
end

"given expr = :(2x+3y-3z+1) return [:x, :y, :c], [2, 3, -3], 1"
get_linear_coeffs(expr::Symbol) = [expr], [1.0], 0
function get_linear_coeffs(expr::Expr)
    vars = find_variables(expr)
    coeffs = zeros(length(vars))
    scalar = deepcopy(expr)
    for v in vars
        substitute!(scalar, v, 0)
    end
    scalar = eval(scalar)


    for i = 1:length(vars)
        expr_copy = deepcopy(:($expr - $scalar))
        for v in vars
            if vars[i] == v
                substitute!(expr_copy, v, 1)
            else
                substitute!(expr_copy, v, 0)
            end
        end
        coeffs[i] = eval(:($expr_copy))
    end
    return vars, coeffs, scalar
end

"""
----------------------------------------------
low level functions: max constraints
----------------------------------------------
"""

""" find if this expr constains max """
is_max_expr(expr::Union{Symbol, Expr}) = :max in get_symbols(expr)

""" count number of maximums in the expression """
function number_of_max(expr::Union{Expr, Symbol})
   n_max = 0
   postwalk(e -> e== :max ? n_max +=1 : nothing, expr)
   return n_max
end

""" parse a big max expression by breaking it into single max expressions"""
function parse_max_expr(expr::Expr, bound_parser::OverApproximationParser)
    # make sure expr is of form (z == a1*max(0,x1) + a2*max(0,x2) +...)
    assert_expr(expr, "max")

    new_var_list = [expr.args[2]] # left hand side variable added to the list
    new_coeff_list = [1.] # coefficient of left hand side variable is 1
    rite_expr = expr.args[3]

    if rite_expr.args[1] == :+  # all max in one sum
        for arg in rite_expr.args[2:end]
            if number_of_max(arg) == 1
                new_var = add_var() # e.g. new_var = max(0, -1.6666666666666667 * (x2 - 0.0))
                new_coeff = -arg.args[2]
                push!(new_var_list, new_var)
                push!(new_coeff_list, new_coeff)
                new_expr = :($new_var == $(arg.args[3]))
                parse_single_max_expr(new_expr, bound_parser)
                bound_parser.ranges[new_var] = find_range(arg.args[3], bound_parser.ranges)
            else
                new_var = add_var()
                new_coeff = -1.
                push!(new_var_list, new_var)
                push!(new_coeff_list, new_coeff)
                new_expr = :($new_var == $arg)
                parse_max_expr(new_expr, bound_parser)
                bound_parser.ranges[new_var] = find_range(arg.args[3], bound_parser.ranges)
            end
        end
    else # rite_expr.args == :- two argument, each could include multiple max's.
        # first max
        first_arg = rite_expr.args[2]
        if number_of_max(first_arg) == 1
            new_var1 = add_var()
            new_coeff1 = -first_arg.args[2]
            push!(new_var_list, new_var1)
            push!(new_coeff_list, new_coeff1)
            new_expr1 = :($new_var1 == $(first_arg.args[3]))
            parse_single_max_expr(new_expr1, bound_parser)
            bound_parser.ranges[new_var1] = find_range(first_arg.args[3], bound_parser.ranges)
        else
            new_var1 = add_var()
            new_coeff1 = -1.
            push!(new_var_list, new_var1)
            push!(new_coeff_list, new_coeff1)
            new_expr1 = :($new_var1 == $first_arg)
            parse_max_expr(new_expr1, bound_parser)
            bound_parser.ranges[new_var1] = find_range(first_arg.args[3], bound_parser.ranges)
        end

        # second max
        second_arg = rite_expr.args[3]
        if number_of_max(second_arg) == 1
            new_var2 = add_var()
            new_coeff2 = second_arg.args[2]
            push!(new_var_list, new_var2)
            push!(new_coeff_list, new_coeff2)
            new_expr2 = :($new_var2 == $(second_arg.args[3]))
            parse_single_max_expr(new_expr2, bound_parser)
            bound_parser.ranges[new_var2] = find_range(second_arg.args[3], bound_parser.ranges)
        else
            new_var2 = add_var()
            new_coeff2 = 1.
            push!(new_var_list, new_var2)
            push!(new_coeff_list, new_coeff2)
            new_expr2 = :($new_var2 == $second_arg)
            parse_max_expr(new_expr2, bound_parser)
            bound_parser.ranges[new_var2] = find_range(second_arg.args[3], bound_parser.ranges)
        end
    end
    push!(bound_parser.eq_list, EqConstraint(new_var_list, new_coeff_list, 0.))
end

function parse_single_max_expr(expr::Expr, bound_parser::OverApproximationParser)
    # truly for parsing relus
    # make sure expr is of form :(z == max(0, x)) or :(z == max(0, min(x,y)))
    assert_expr(expr, "single max")
    z = expr.args[2]
    x = expr.args[3].args[3]
    if x isa Symbol
        push!(bound_parser.relu_list, ReluConstraint(x, z))
    else
        new_var = add_var()
        push!(bound_parser.relu_list, ReluConstraint(new_var, z)) # z=relu(new_var)
        new_expr = :($new_var == $x) # new_var = x
        parse_eq(new_expr, bound_parser)
        bound_parser.ranges[new_var] = find_range(x, bound_parser.ranges)
    end
end

"""
----------------------------------------------
low level functions: min constraints
----------------------------------------------
"""

""" find if this expr constains max or min"""
is_max_expr(expr::Union{Symbol, Expr}) = :max in get_symbols(expr)
is_min_expr(expr::Union{Symbol, Expr}) = :min in get_symbols(expr)

""" parse min expression. min should already be a single min expression. """
function parse_min_expr(expr::Expr, bound_parser::OverApproximationParser)
    parse_single_min_expr(expr, bound_parser)
end

function parse_single_min_expr(expr::Expr, bound_parser::OverApproximationParser)
    # make sure expr is of form :(z == min(x, y))
    # MAJOR ASSUMPTION: MIN IS NEVER NESTED
    assert_expr(expr, "single min")
    z = expr.args[2]
    x = expr.args[3].args[2]
    y = expr.args[3].args[3]

    # define x2 = -x
    x2 = add_var()
    if x isa Symbol
        push!(bound_parser.eq_list, EqConstraint([x2, x], [1., 1.], 0.))
        xl, xu = bound_parser.ranges[x]
        bound_parser.ranges[x2] = [-xu, -xl]
    else
        e = simplify(:(-1*$x))
        new_expr = :($x2 == $(e))
        parse_eq(new_expr, bound_parser)
        bound_parser.ranges[x2] = find_range(e, bound_parser.ranges)
    end

    # define y2 = -y
    y2 = add_var()
    if y isa Symbol
        push!(bound_parser.eq_list, EqConstraint([y2, y], [1., 1.], 0.))
        yl, yu = bound_parser.ranges[y]
        bound_parser.ranges[y2] = (-yu, -yl)
    else
        e = simplify(:(-1*$y))
        new_expr = :($y2 == $(e))
        parse_eq(new_expr, bound_parser)
        bound_parser.ranges[y2] = find_range(e, bound_parser.ranges)
    end

    # define z2 = -z
    z2 = add_var()
    push!(bound_parser.eq_list, EqConstraint([z2, z], [1., 1.], 0.))
    if ~haskey(bound_parser.ranges, z)
        xl, xu = find_range(x, bound_parser.ranges)
        yl, yu = find_range(y, bound_parser.ranges)
        bound_parser.ranges[z] = [min(xl,yl), min(xu, yu)] # TODO: DOUBLE CHECK CORRECTNESS
    end
    zl, zu = bound_parser.ranges[z]
    bound_parser.ranges[z2] = [-zu, -zl]

    # add z2 = max(x2, y2)
    push!(bound_parser.max_list, MaxConstraint([x2, y2], z2))
end

"""
----------------------------------------------
low level functions: nonlinear constraints that are not max/min/relu
----------------------------------------------
"""
function parse_nonlinear_expr(expr::Expr, bound_parser::OverApproximationParser)
    # return a object of type NLConstraint(left, R, right,  indep_var)
    # currently only supports a single independent variable
    # Also assumes that expressions are of the form: v1 R sin(v3)
    # (has a single variable on the left hand side)
    indep_var = find_variables(expr.args[3])
    @assert (length(indep_var) == 1) #only handles 1D functions right now
    push!(bound_parser.true_NL_list, NLConstraint(expr.args[2], expr.args[1], expr.args[3], indep_var[1]))
end

"""
----------------------------------------------
all assertation compiled here
----------------------------------------------
"""

function assert_expr(expr::Expr, type::String)
    @assert length(expr.args) == 3
    @assert expr.args[2] isa Symbol
    if type == "eq"
        @assert expr.args[1] == :(==)
#        @assert is_min_expr(expr) || is_max_expr(expr) || is_linear_expr(expr)
    elseif type == "ineq"
        # make sure expr is of form (x ≦ y)
        @assert expr.args[1] == :≦
        @assert expr.args[3] isa Symbol
    elseif type == "max"
        # make sure expr is of form (z == a1*max(0,x1)+ a2*max(0,x2)+...)
        @assert expr.args[1] == :(==)
        # @assert expr.args[3].args[1] == :+
        # for arg in expr.args[3].args[2:end]
        #     @assert arg.args[1] == :*
        #     @assert arg.args[2] isa Number
        #     @assert arg.args[3].args[1] == :max
        #     @assert arg.args[3].args[2] == 0
        # end
    elseif type == "single max"
        # make sure expr is of form :(z == max(0, x))
        @assert expr.args[1] == :(==)
        @assert length(expr.args[3].args) == 3
        @assert expr.args[3].args[1] == :max
        @assert expr.args[3].args[2] == 0
    elseif type == "single min"
        # make sure expr is of form :(z == min(x, y))
        @assert expr.args[1] == :(==)
        @assert length(expr.args[3].args) == 3
        @assert expr.args[3].args[1] == :min
    elseif type == "linear"
        #make sure the expr is of form :(z == ax + b) or :(z == a*x + b*y + c*w)
        right_expr = expr.args[3]
        @assert expr.args[1] == :(==)
        @assert length(right_expr.args) >= 3
        @assert is_affine(right_expr)
        # if length(rite_expr.args) == 3 #form :(z == ax + b)
        #     @assert xor(rite_expr.args[2] isa Number, rite_expr.args[3] isa Number)
        #     ax = rite_expr.args[2] isa Number ? rite_expr.args[3] : rite_expr.args[2]
        #     @assert ax.args[1] == :*
        #     @assert xor(ax.args[2] isa Number, ax.args[3] isa Number)
        #     @assert xor(ax.args[2] isa Symbol, ax.args[3] isa Symbol)
        # else
        #     @assert rite_expr.args[1] == :+ #form :(z == ax + by + cw)
        #     for arg in rite_expr.args[2:end]
        #         @assert length(arg.args) == 3
        #         @assert arg.args[1] == :*
        #         xor(arg.args[2] isa Number, arg.args[3] isa Number)
        #         xor(arg.args[2] isa Symbol, arg.args[3] isa Symbol)
        #     end
        # end
    else
        throw("type $type is not recognized")
    end
end

"""
----------------------------------------------
write to file
----------------------------------------------
"""
function write_constraint(file_name, c::EqConstraint, i; prefix="")
    h5write(file_name, prefix*"eq/vars$i", [string(v) for v in c.vars])
    h5write(file_name, prefix*"eq/coeffs$i", [string(v) for v in c.coeffs])
    h5write(file_name, prefix*"eq/scalar$i", [string(c.scalar)])
end

function write_constraint(file_name, c::ReluConstraint, i; prefix="")
    h5write(file_name, prefix*"relu/varin$i", [string(c.varin)])
    h5write(file_name, prefix*"relu/varout$i", [string(c.varout)])
end

function write_constraint(file_name, c::MaxConstraint, i; prefix="")
    h5write(file_name, prefix*"max/varsin$i", [string(v) for v in c.varsin])
    h5write(file_name, prefix*"max/varout$i", [string(c.varout)])
end

function write_constraint(file_name, c::IneqConstraint, i; prefix="")
    h5write(file_name, prefix*"ineq/varleft$i", [string(c.varleft)])
    h5write(file_name, prefix*"ineq/varright$i",[string(c.varrite)])
end

function write_constraint(file_name, c::NLConstraint, i; prefix="")
    h5write(file_name, prefix*"nl/left$i", [string(c.left)])
    h5write(file_name, prefix*"nl/R$i", [string(c.R)])
    h5write(file_name, prefix*"nl/right$i", [string(c.right)])
    h5write(file_name, prefix*"nl/indep_var$i", [string(c.indep_var)])
end

function write_overapproximateparser(bound_parser::OverApproximationParser,
                                     file_name::String,
                                     state_vars::Array{Symbol, 1},
                                     control_vars::Array{Symbol, 1},
                                     output_vars::Array{Symbol, 1})

    # write input and output variables to file
    h5write(file_name, "vars/states", [string(v) for v in state_vars])
    h5write(file_name, "vars/controls", [string(v) for v in control_vars])
    h5write(file_name, "vars/outputs", [string(v) for v in output_vars])

    # write constraints to file
    for (i, eq) in enumerate(bound_parser.eq_list)
        write_constraint(file_name, eq, i)
    end
    for (i, eq) in enumerate(bound_parser.relu_list)
        write_constraint(file_name, eq, i)
    end
    for (i, eq) in enumerate(bound_parser.max_list)
        write_constraint(file_name, eq, i)
    end
    for (i, eq) in enumerate(bound_parser.ineq_list)
        write_constraint(file_name, eq, i)
    end
    for (i, c) in enumerate(bound_parser.true_L_list)
        write_constraint(file_name, c, i, prefix="true_fun/")
    end
    for (i, c) in enumerate(bound_parser.true_NL_list)
        write_constraint(file_name, c, i, prefix="true_fun/")
    end
    for (v, vrange) in bound_parser.ranges
        h5write(file_name, "ranges/$v", [string(vrange)])
    end

end
#
#
"""
----------------------------------------------
Tests
----------------------------------------------
"""

function evaluate(oAP_test::OverApproximationParser, var_dict::Dict)
    println("Evaluate")
    """
    Given a OverApproximationParser object and a dictionary of input variables
    this function evaluate all other variables that are introduced. And the
        variable and its value to the var_dict.
    """

    for eq in oAP_test.relu_list
        if eq.varin in keys(var_dict)
            var_dict[eq.varout] = max(0, var_dict[eq.varin])
        end
    end

    for eq in oAP_test.max_list
        if eq.varsin ⊆ keys(var_dict)
            var_dict[eq.varout] = max(var_dict[eq.varsin[1]], var_dict[eq.varsin[2]])
        end
    end

    for c in oAP_test.true_NL_list
        if c.indep_var ∈ keys(var_dict)
            rcopy = deepcopy(c.right)
            @assert c.R == :(==) # only handles equality relations right now
            var_dict[c.left] = eval(substitute!(rcopy, c.indep_var, var_dict[c.indep_var]))
        end
    end

    for eq in oAP_test.eq_list
        v = setdiff(eq.vars, keys(var_dict))
        if length(v) == 1
            c = 0
            val = 0
            v = v[1]
            for i = 1:length(eq.vars)
                if eq.vars[i] == v
                    c = eq.coeffs[i]
                else
                    val += eq.coeffs[i]*var_dict[eq.vars[i]]
                end
            end
            val -= eq.scalar
            val /= -c
            var_dict[v] = val
        end
    end
end

function test_random_input(expr::Expr, oAP_test::OverApproximationParser)
    """
    for a given expr and its associated OverApproxmiationParser object, this
        function  computes the expr and its parser at some random values,
            and compares it together. Return true if they're almost equal (≈).
        At everystep, function evaluate replace all variables with their values
        given by dict_var. If a equality or a max or a relu can be solved
        with the available values, then the will be solved, and the value will be
        added to the dictionary. for example if you have
        x1 = x2 + x3
        x4 = max(x2, x1)
        you dictionary is supposed to have the values for x2 and x3. Then in the
        first round, x1 is solved. and then in the second round x4 is solved.
        note that the order of the equations is not necessarily in the order
        that you can solve them, so maybe the equations that can be reduced are
        at the bottom of your list. With this approach, you get rid of at least
        one equation every step of the for loop
    """

    vars = find_variables(expr.args[3])
    vars = setdiff(vars, [:min, :max])
    if expr.args[1] ∈ [:max, :min, :relu]
        values = rand(length(vars))*10 .- 5 # random number generated between -5 and 5
    else
        values = rand(length(vars))*10 # random number between 0 and 10, to avoid out of domain log
    end
    expr_eval = copy(expr)
    for (k, v) in zip(vars, values)
        substitute!(expr_eval, k, v)
    end
    expr_eval_rite = simplify(expr_eval.args[3]) # evaluate expression directly

    dict_var = Dict(zip(vars, values))
    while expr.args[2] ∉ keys(dict_var)
        println("expr: ", expr)
       evaluate(oAP_test, dict_var) # evaluate expression that has been parsed into bound parser
       println(dict_var)
    end

    # println(expr_eval_rite)
    # println(dict_var[expr.args[2]])

    return expr_eval_rite ≈ dict_var[expr.args[2]]
end


# these one should pass
# assert_expr(:(w == 2x + 3.4y +1z), "eq")
# assert_expr(:(w == max(0, x)), "eq")
# assert_expr(:(w == min(x, y)), "eq")
# assert_expr(:(x5 ≦ y1), "ineq")
# assert_expr(:(w == 2max(0, x) + 3max(0,y)), "max")
# assert_expr(:(w == 2max(0, x) + 3max(0, min(2z,t-1))), "max")
# assert_expr(:(w == max(0, x)), "single max")
# assert_expr(:(w == max(0, min(2z,t-1))), "single max")
# assert_expr(:(w == min(x,-y)), "single min")
# assert_expr(:(w == 2x-1), "linear")
# assert_expr(:(w == 2x+1y+1.5z), "linear")

# these one should NOT pass
# assert_expr(:(2x + 3.4y + 2z), "eq") # missing w == ...
# assert_expr(:(w==2x + 3.4y^2 -z), "eq") # squared term
# assert_expr(:(-x ≦ y), "ineq") # either sides should be symbols and not expressions
# assert_expr(:(w == 2max(y, x) + 3max(0,y)), "max") # max constraints are always relu.
# assert_expr(:(w == max(0, x)), "max") # this is a single max.
# assert_expr(:(w == max(x, 0)), "single max") # first max argument should be 0.
# assert_expr(:(w == 2max(0, min(2z,t-1))), "single max") # single max does not have a coefficient.
# assert_expr(:(w == min(x+y)), "linear")
# assert_expr(:(w == 2max(x, 3) + 3max(min(x,-y), min(2z,t-1))), "single max")
# assert_expr(:(w == max(x,y)), "single max")
# assert_expr(:(w == 2x-y+1.5z), "linear")

# """ other tests """
# oAP_test = OverApproximationParser()
# parse_ineq(:(x ≦ y), oAP_test)
# @assert oAP_test.ineq_list[1].varleft == :x
# @assert oAP_test.ineq_list[1].varrite == :y
#
# oAP_test = OverApproximationParser()
# parse_single_max_expr(:(y == max(0, x)), oAP_test)
# @assert oAP_test.relu_list[1].varin == :x
# @assert oAP_test.relu_list[1].varout == :y
# @assert oAP_test.max_list == []
#
# oAP_test = OverApproximationParser()
# parse_linear_expr(:(z == 2x+4y), oAP_test)
# @assert oAP_test.eq_list[1].vars == [:x, :y, :z]
# @assert oAP_test.eq_list[1].coeffs == [-2, -4, 1]
# @assert oAP_test.eq_list[1].scalar == 0
#
# oAP_test = OverApproximationParser()
# parse_linear_expr(:(z == -2x+1), oAP_test)
# @assert oAP_test.eq_list[1].vars == [:x, :z]
# @assert oAP_test.eq_list[1].coeffs == [2, 1]
# @assert oAP_test.eq_list[1].scalar == 1
#
# oAP_test = OverApproximationParser()
# parse_linear_expr(:(z == -2x-2y-2), oAP_test)
# @assert oAP_test.eq_list[1].vars == [:x, :y, :z]
# @assert oAP_test.eq_list[1].coeffs == [2, 2, 1]
# @assert oAP_test.eq_list[1].scalar == -2
#
# oAP_test = OverApproximationParser()
# parse_single_min_expr(:(z == min(y, x)), oAP_test)
# @assert length(oAP_test.relu_list) == 0
# @assert length(oAP_test.max_list) == 1
# @assert length(oAP_test.eq_list) == 3
#
# oAP_test = OverApproximationParser()
# parse_single_max_expr(:(z == max(0, min(3x-1, 2y))), oAP_test)
# @assert length(oAP_test.relu_list) == 1
# @assert length(oAP_test.max_list) == 1
# @assert length(oAP_test.eq_list) == 3
#
# oAP_test = OverApproximationParser()
# parse_max_expr(:(z == 3*max(0,y) + 2*max(0, min(3x-1, 2y))), oAP_test)
# @assert length(oAP_test.relu_list) == 2
# @assert length(oAP_test.max_list) == 1
# @assert length(oAP_test.eq_list) == 4

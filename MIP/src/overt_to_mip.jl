using JuMP
using Gurobi
using Crayons

"""
----------------------------------------------
main structure
----------------------------------------------
"""

"""
This structure is a mixed-integer-representation of Overt. The relu and max
operations are turned into mix integer programs following ideas from
MIPVerify paper https://arxiv.org/abs/1711.07356
"""
mutable struct OvertMIP
    overt_app::OverApproximation  # overt OverApproximation object
    model::JuMP.Model             # a JuMP model for MIP
    vars_dict::Dict{Symbol, JuMP.VariableRef}  # dictionary of Overt symbols and their associated variable in mip
    solver::String                # solver, default is Gurobi
end

# default constructor
function OvertMIP(overt_app::OverApproximation; threads=0)
    # for Gurobi, 0 threads is automatic (usually most of the cores in the machine)
     overt_mip_model = OvertMIP(overt_app,
                         Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0, Threads=threads)),
                         Dict{Symbol, JuMP.VariableRef}(),
                         "Gurobi")
    overt_2_mip(overt_mip_model)
    return overt_mip_model
end

#TODO: secondary constructor with specified solver


Base.print(overt_mip_model::OvertMIP) = println(overt_mip_model.model)
Base.display(overt_mip_model::OvertMIP) = println(overt_mip_model.model)


"""
----------------------------------------------
some utility functions
----------------------------------------------
"""

"""
This function returns mip variable associated with a Symbol in Overt.
if the Symbol does not have an mip variable yet, one is created.
With a new variable created, the range of that variable will be added to the
mip model. The ranges are obtained from range attribute of OverApproximation object.
"""
function get_mip_var(var::Symbol, overt_mip_model::OvertMIP)
    if var in keys(overt_mip_model.vars_dict)
        mip_var = overt_mip_model.vars_dict[var]
    else
        l, u = overt_mip_model.overt_app.ranges[var]
        mip_var = @variable(overt_mip_model.model, lower_bound=l, upper_bound=u, base_name="$var")
        overt_mip_model.vars_dict[var] = mip_var
    end
    return mip_var
end

"""
This function creates an auxillary mip variable. These are the intermediate defined
    when relu or max are turned into mip.
"""
aux_var_counter = 1
function get_mip_aux_var(overt_mip_model::OvertMIP; binary::Bool = false)
    global aux_var_counter;
    base_name = "aux_var_$aux_var_counter"
    mip_aux_var = @variable(overt_mip_model.model, binary=binary, base_name=base_name)
    aux_var_counter += 1
    return mip_aux_var
end

"""
----------------------------------------------
main loop
----------------------------------------------
"""

"""
This function reads an OverApproximation object and translate all constraints
    in terms of a Mixed Integer Program.
    input: overt_mip_model, an object for mip model of overt.

    Equality constraints in Overt are of two types:
        - linear equalities are an affine relation like: like z == 2x+3y
        - max equalities contain a max operation. They are of two kinds themselves:
                -- max-alone: like z = max(x, 0) which is basically a relu
                -- max-min:   like z = max(min(x,y), 0)
        linear equalities are processed in linear_2_mip as an additional constraints
        added to the model.
        max-alone equalities are processed in max_alone_2_mip. they are turned to a
        mip representation of a relu (following MIPVerify algorithm) and then
        added to the model
        max-min equalities are processed in max_min_2_mip. they are turned to a
        mip representation of a max-min, which is similar to the what is derived
        in MIPVerify algorithms for relu and max.
"""
function overt_2_mip(overt_mip_model::OvertMIP)
    # parsing equalities
    for eq in overt_mip_model.overt_app.approx_eq
        eq_2_mip(eq, overt_mip_model)
    end

    # parsing inequalities
    for ineq in overt_mip_model.overt_app.approx_ineq
        ineq_2_mip(ineq, overt_mip_model)
    end
end

"""
turn equality constraints to mip
"""
function eq_2_mip(expr::Expr, overt_mip_model::OvertMIP)
    #println(expr)
    fix_negation!(expr)
    if is_max_expr(expr)
        max_2_mip(expr, overt_mip_model)
    else
        affine_to_mip(expr, overt_mip_model)
    end
end

"""
turn inequality constraints to mip
"""
function ineq_2_mip(expr::Expr, overt_mip_model::OvertMIP)
    leftvar = get_mip_var(expr.args[2], overt_mip_model)
    ritevar = get_mip_var(expr.args[3], overt_mip_model)
    @constraint(overt_mip_model.model, leftvar <= ritevar )
end

"""
----------------------------------------------
low level functions: affine equalities
----------------------------------------------
"""

""" parse affine equalities z == 2*x + 3*y -1 """
function affine_to_mip(expr::Expr, overt_mip_model::OvertMIP)
    lhs = get_mip_var(expr.args[2], overt_mip_model)
    var_list, coeff_list, scalar = get_linear_coeffs(expr.args[3])
    for (v, c) in zip(var_list, coeff_list)
        mip_var = get_mip_var(v, overt_mip_model)
        lhs -= mip_var*c
    end
    lhs -= scalar
    @constraint(overt_mip_model.model, lhs == 0)
end


"""
----------------------------------------------
low level functions: max equalities
----------------------------------------------
"""

""" parse a big max into some of max-only or max-min expressions """
function max_2_mip(expr::Expr, overt_mip_model::OvertMIP)
    lhs = get_mip_var(expr.args[2], overt_mip_model)
    max_list = expr.args[3].args[2:end]
    for e in max_list
        @assert e.args[1] == :*
        c = e.args[2]
        tmp_var = get_mip_aux_var(overt_mip_model)
        lhs -= c*tmp_var
        if is_min_expr(e)
            max_min_2_mip(e.args[3], tmp_var, overt_mip_model)
        else
            max_alone_2_mip(e.args[3], tmp_var, overt_mip_model)
        end
    end
    @constraint(overt_mip_model.model, lhs == 0)
end

""" parse expr like y == max(0, x) """
function max_alone_2_mip(expr::Expr, outvar::JuMP.VariableRef, overt_mip_model::OvertMIP)
    var, coef, scalar = get_linear_coeffs(expr.args[3])
    var, coef = var[1], coef[1]
    l, u = overt_mip_model.overt_app.ranges[var]
    if coef > 0
        L = l * coef + scalar
        U = u * coef + scalar
    else
        L = u * coef + scalar
        U = l * coef + scalar
    end
    var_mip = get_mip_var(var, overt_mip_model)
    a = get_mip_aux_var(overt_mip_model; binary=true)

    @constraint(overt_mip_model.model, outvar >= 0)
    @constraint(overt_mip_model.model, outvar >= var_mip*coef + scalar)
    @constraint(overt_mip_model.model, outvar <= a*U)
    @constraint(overt_mip_model.model, outvar <= var_mip*coef + scalar - L*(1-a))
end


""" parse expr like z == max(0, min(x, y)) """
function max_min_2_mip(expr::Expr, outvar::JuMP.VariableRef, overt_mip_model::OvertMIP)
    @assert expr.args[1] == :max
    @assert expr.args[2] == 0
    min_arg = expr.args[3]
    @assert min_arg.args[1] == :min

    var1, coef1, scalar1 = get_linear_coeffs(min_arg.args[2])
    var1, coef1 = var1[1], coef1[1]
    l1, u1 = overt_mip_model.overt_app.ranges[var1]
    if coef1 > 0
        L1 = l1 * coef1 + scalar1
        U1 = u1 * coef1 + scalar1
    else
        L1 = u1 * coef1 + scalar1
        U1 = l1 * coef1 + scalar1
    end

    var2, coef2, scalar2 = get_linear_coeffs(min_arg.args[3])
    var2, coef2 = var2[1], coef2[1]
    l2, u2 = overt_mip_model.overt_app.ranges[var2]
    if coef2 > 0
        L2 = l2 * coef2 + scalar2
        U2 = u2 * coef2 + scalar2
    else
        L2 = u2 * coef2 + scalar2
        U2 = l2 * coef2 + scalar2
    end

    L3 = min(L1, L2)
    U3 = max(U1, U2)

    var1_mip = get_mip_var(var1, overt_mip_model)
    var2_mip = get_mip_var(var2, overt_mip_model)

    a  = get_mip_aux_var(overt_mip_model; binary=true)
    b  = get_mip_aux_var(overt_mip_model; binary=true)
    x3 = get_mip_aux_var(overt_mip_model)

    @constraint(overt_mip_model.model, x3 >= coef1*var1_mip + scalar1 - (1 - b) * (U1 - L2))
    @constraint(overt_mip_model.model, x3 >= coef2*var2_mip + scalar2 - b * (U2 - L1))
    @constraint(overt_mip_model.model, x3 <= coef1*var1_mip + scalar1)
    @constraint(overt_mip_model.model, x3 <= coef2*var2_mip + scalar2)

    @constraint(overt_mip_model.model, outvar >= 0)
    @constraint(overt_mip_model.model, outvar >= x3)
    @constraint(overt_mip_model.model, outvar <= a*U3)
    @constraint(overt_mip_model.model, outvar <= x3 - L3*(1-a))
end

function mip_summary(model)
    MathOptInterface = MOI
    const_types = list_of_constraint_types(model)
    l_lin = 0
    l_bin = 0
	println(Crayon(foreground = :yellow), "="^50)
	println(Crayon(foreground = :yellow), "="^18 * " mip summary " * "="^19)
    for i = 1:length(const_types)
        var = const_types[i][1]
        const_type = const_types[i][2]
        l = length(all_constraints(model, var, const_type))
        #println("there are $l constraints of type $const_type with variables type $var.")
        if const_type != MathOptInterface.ZeroOne
            l_lin += l
        else
            l_bin += l
        end
    end
	println(Crayon(foreground = :yellow), "there are $l_lin linear constraints and $l_bin binary constraints.")
	println(Crayon(foreground = :yellow), "="^50)
    println(Crayon(foreground = :white), " ")
end

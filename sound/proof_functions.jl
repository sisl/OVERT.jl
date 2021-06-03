# checking the soundness of overapproximations for the benchmark problems

include("../OverApprox/src/overapprox_nd_relational.jl")
include("soundness.jl")

# Checking sample problem

function sample1()
    # sample 1
    input_vars = [:x]
    domain = Dict(:x => [-π/4, π/4])
    # construct overapproximation
    example_dynamics = :(sin(cos(x)) + 5) 
    oa = overapprox_nd(example_dynamics, domain, N=-1, ϵ=.1)
end

function sample2()
    # sample 2
    input_vars = [:x, :y]
    domain = Dict(:x => [-π/4, π/4], :y => [-.1,.1])
    # construct overapproximation
    example_dynamics = :(y*sin(x) + 5)  
    oa = overapprox_nd(example_dynamics, domain, N=-1, ϵ=.1)
end

##################### MODE 3: BUILD UP PROOFS OF EACH LEVEL OF THE APPROX

# # get dependencies of variables
# function get_dependency(var, list, input_vars, exclude_vars)
#     # for this var, retrieve all constraints that include this var but do NOT include exclude_vars AND exclude all constraints with vars that are "greater" than var
#     function map_fun(list_item)
#         vars_in_c = Expr.(free_symbols(Basic(list_item)))
#         vars_in_c_excluding_inputs = setdiff(vars_in_c, input_vars)
#         return (var in vars_in_c) && !any(exclude_vars .∈ Ref(vars_in_c)) && all(special_compare.(vars_in_c_excluding_inputs, Ref(var)))
#     end
#     idxs = map(map_fun, list)
#     new_free_vars = setdiff(Expr.(free_symbols(Basic.(list[idxs]))), [exclude_vars..., var])
#     return idxs, new_free_vars # return boolean indices and free variables
# end
# function get_all_dependecies(vars, constraint_list, input_vars, exclude_vars)
#     free_vars = Set(vars) 
#     all_free_vars = Set(vars) 
#     idxs = Bool.(zeros(length(constraint_list)))
#     # get all dependent constraint indices for the vars in free_vars
#     for v in free_vars
#         idxs_i, new_free_vars_i = get_dependency(v, constraint_list, input_vars, exclude_vars)
#         idxs .|= idxs_i
#         if length(new_free_vars_i) > 0
#             push!(all_free_vars, new_free_vars_i...)
#         end
#     end
#     new_free_vars = setdiff(all_free_vars, free_vars)
#     old_free_vars = setdiff(all_free_vars, new_free_vars)
#     if length(new_free_vars) == 0
#         return idxs
#     else
#         if length(old_free_vars) > 0
#             push!(exclude_vars, old_free_vars...)
#         end
#         idxs .|= get_all_dependecies(new_free_vars, constraint_list, input_vars, exclude_vars) 
#         return idxs
#     end
# end

function special_compare(v1, v2)
    # returns TRUE is v1 <= v2
    # for variables of the form: :v_1, :v_2, etc. 
    @debug("special_compare: ", v1, v2)
    v1num = Meta.parse(match(r"(?<=_)(.*)", string(v1)).captures[1])
    v2num = Meta.parse(match(r"(?<=_)(.*)", string(v2)).captures[1])
    return v1num <= v2num 
end

# # iterate through pieces of og function
# # (imagine sorted now but doesn't have to be i think)
# # e.g. take v1 = f(x)
# function check_overapprox(oa, domain, input_vars, problem_name; jobs=1, delta_sat=0.001)
#     # NOTE THIS CODE RELIES ON THE VARIABLE ORDERING!
#     vars = sort!(collect(keys(oa.fun_eq))) # sort just there for debugging
#     R = true
#     for vari in vars
#         ϕ = [:($vari == $(oa.fun_eq[vari]) )]
#         # for each variable vi in og func, consider all approx constraints that have ALL vars leq to it (plus input var(s))
#         all_approx_constraints = [oa.approx_eq..., oa.approx_ineq...]
#         all_approx_constraints = [((constraint.args[1] == :≤) || (constraint.args[1] == :≦) ) ? :($(constraint.args[2]) <= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]
#         all_approx_constraints = [((constraint.args[1] == :≥) || (constraint.args[1] ==  :≧)) ? :($(constraint.args[2]) >= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]

#         all_dependencies = all_approx_constraints[get_all_dependecies([vari], all_approx_constraints, input_vars, [])]

#         defs = []
#         ϕ̂ = []
#         for c in all_dependencies
#             vars_in_c = Expr.(free_symbols(Basic(c))) # use SymEngine free_symbols function
#             if vari ∉ vars_in_c
#                 push!(defs, c)
#             else
#                 push!(ϕ̂, c)
#             # things that go in phi hat are those with the output variable in them? e.g. v6 ? (everything else in defs?) [kind of like mode 2 but only 1 piece at a time for the og func?]
#             end
#         end

#         sq = SoundnessQuery(ϕ, # ϕ
#                             defs,
#                             ϕ̂, # definitions for ϕ̂
#                             domain)
#         # println("Soundness Query: $sq")

#         result = check("dreal", sq, problem_name*"_soundness_query_"*string(vari)*".smt2", δ=delta_sat, jobs=jobs) # TODO: pass dreal delta
#         println("result for var ",vari," is: ", result)
#         R &= occursin("unsat", result)
#         # readline() # to make things interactive for debugging
#     end
#     R ? println("all checks pass for "*problem_name*"!") : println("Some checks fail :( for "*problem_name)
#     return R
# end
#check_overapprox(oa, domain, input_vars)

function get_free_vars(e::Expr)
    return Symbol.(free_symbols(Basic(e)))
end

function run_soundness_query(ϕ, ϕ̂, defs, domain, problem_name, delta_sat, jobs)
    println("ϕ = $ϕ, ϕ̂ = $ϕ̂, all defs: $defs")
    sq = SoundnessQuery(ϕ, # ϕ
                            defs,
                            ϕ̂, # definitions for ϕ̂
                            domain)
        # println("Soundness Query: $sq")

        result = check("dreal", sq, problem_name*"_soundness_query.smt2", δ=delta_sat, jobs=jobs) # TODO: pass dreal delta
        println("result for var ",problem_name," is: ", result)
        return occursin("unsat", result)
end

import Base.setdiff
function setdiff(s::Set{Symbol}, e::Symbol)
    elements = [s...]
    return Set(filter(i -> i != e, elements))
end

function get_approx_equality_def(var, approx_equality_list)
    defs = filter(c -> c.args[2] == var, approx_equality_list)
    @assert(length(defs) == 1)
    return defs[1]
end

function assert_ineq_bounded(cur_var_bounds, current_var)
    println("current_var=$(current_var)")
    println("cur_var_bounds = $(cur_var_bounds)")
    # assert that the bounds sandwhich the current_var e.g. v1 <= current_var <= v2
    @assert length(cur_var_bounds) == 2
    b1 = cur_var_bounds[1]
    b2 = cur_var_bounds[2]
    mixed_set = Set([:(>=),:(<=)])
    if Set([b1.args[1], b2.args[1]]) == mixed_set # if opposite signs
        # current_var must be on same side of ineqs to be sandwhiched
        @assert (b1.args[2] == current_var && b2.args[2] == current_var) || (b1.args[3] == current_var && b2.args[3] == current_var)
    elseif (b1.args[1] ∈ mixed_set) && (b2.args[1] ∈ mixed_set) && (b1.args[1] == b2.args[1]) # share same ineq sense
        # current var must be on dif sides of ineq to be sandwhiched
        @assert (b1.args[2] == current_var && b2.args[3] == current_var) || (b1.args[3] == current_var && b2.args[2] == current_var)
    else
        @assert false
    end
end

# Overall plan: do this in a more structured way.
# recurse down the fun_eq tree 
# and then collect the approx pieces one by one
# e.g. first check if a variable definition exists in approx_eq. if not, then look in approx_ineq 
# make sure all variables are defined before generating smt2 file
    # begin the recursion from the output variable
    # output = oa.output 
function __check_overapprox(current_var, defs, oa, domain, input_vars, problem_name; jobs=1, delta_sat=0.001)
    println("\n current_var = $(current_var), \n defs = $defs") 
    # base case: IS an input var, not v_i variable
    if current_var ∈ input_vars
        return [], true
    # check if v1 base case: RHS only contains inputs 
        # if v1 def is affine add to defs and return
        # if v1 not affine, check for bounds for v1 in oa.approx_ineq, put those into phi hat and v1 into phi and gen proof goal 
            # then, move ineqs to defs and return 
    # if not base case
    else
        # check_overapprox of free_vars in RHS expression --> should return defs for each one, merge these into existing defs 
        free_vars = Set(get_free_vars(oa.fun_eq[current_var])) # get free vars in exact expression
        println("free_vars = $(free_vars)")
        # get dependencies of each free var (and also check proof goals of dependencies)
        data = [__check_overapprox(v, [], oa, domain, input_vars, problem_name; jobs=jobs, delta_sat=delta_sat) for v in free_vars]
        # check that each free var is either bounded by two inequalities of the right signs or one equality
        println("current_var again= $(current_var)")
        #println("data = $data")
        dependencies = vcat([d[1] for d in data]...)
        #println("dependencies = $dependencies")
        defs = vcat(defs, dependencies)
        #println("defs after recursing are: $defs")
        results = [d[2] for d in data]
        is_unsat = all(results)
        # if v_2 affine, add to defs and return
        if is_affine(oa.fun_eq[current_var])
            defs = vcat(defs, :($(current_var) == $(oa.fun_eq[current_var])))
        else
        # if v_2 not affine, check for bounds for v2 in oa.approx_ineq, gen proof goal...
            cur_var_bounds = filter(e -> current_var ∈ get_free_vars(e), oa.approx_ineq) # get the two inequalities: v2 <= v3 and v3 <= v4
            assert_ineq_bounded(cur_var_bounds, current_var)
            println("cur_var_bounds = $(cur_var_bounds)")
            bound_vars = setdiff(Set(vcat(get_free_vars.(cur_var_bounds)...)), current_var) # get the new vars, e.g. v2, v4
            println("bound_vars = $(bound_vars)")
            bound_var_defs = [get_approx_equality_def(v, oa.approx_eq) for v in bound_vars] # get the definitions for the new vars v2 = ... and v4 = ...
            #println("bound_var_defs = $(bound_var_defs)")
            defs = vcat(defs, bound_var_defs) # add to defs
            # construct proof goal and run 
            ϕ = [:($(current_var) == $(oa.fun_eq[current_var]))]
            ϕ̂ = cur_var_bounds
            problem_specific_name = problem_name*"_$(current_var)_"
            is_unsat &= run_soundness_query(ϕ, ϕ̂, defs, domain, problem_specific_name, delta_sat, jobs)
            # then move ineq bounds to defs and return 
            defs = vcat(defs, ϕ̂)
        end
    end

    # I think actually cases 2 and 3 can be combined so long as case 1 is added where it just returns nothing if variable is an input 
    return defs, is_unsat
end

function replace_unicode(oa::OverApproximation)
    # why did we use ≦ in the oa ???
    oa_approx_ineq_fixed = [( (c.args[1] == :≤) || (c.args[1] == :≦) ) ? :($(c.args[2]) <= $(c.args[3]))  : c for c in oa.approx_ineq ]
    oa_approx_ineq_double_fixed = [( (c.args[1] == :≥) || (c.args[1] == :≧) ) ? :($(c.args[2]) >= $(c.args[3]))  : c for c in oa_approx_ineq_fixed]
    oa.approx_ineq = oa_approx_ineq_double_fixed
    return oa
end

function check_overapprox(oa, domain, input_vars, problem_name; jobs=1, delta_sat=0.001)
    defs = []
    oa_clean = replace_unicode(oa)
    defs, result = __check_overapprox(oa.output, defs, oa_clean, domain, input_vars, problem_name; jobs=jobs, delta_sat=delta_sat)
    result ? println("All checks pass!") : println("Some checks fail :( ")
    return result
end
# checking the soundness of overapproximations for the benchmark problems

include("../models/problems.jl")
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

# get dependencies of variables
function get_dependency(var, list, exclude_vars)
    # for this var, retrieve all constraints that include this var but do NOT include exclude_vars AND exclude all constraints with vars that are "greater" than var
    function map_fun(list_item)
        vars_in_c = Expr.(free_symbols(Basic(list_item)))
        vars_in_c_excluding_inputs = setdiff(vars_in_c, input_vars)
        return (var in vars_in_c) && !any(exclude_vars .∈ Ref(vars_in_c)) && all(special_compare.(vars_in_c_excluding_inputs, Ref(var)))
    end
    idxs = map(map_fun, list)
    new_free_vars = setdiff(Expr.(free_symbols(Basic.(list[idxs]))), [exclude_vars..., var])
    return idxs, new_free_vars # return boolean indices and free variables
end
function get_all_dependecies(vars, list, exclude_vars)
    free_vars = Set(vars) 
    all_free_vars = Set(vars) 
    idxs = Bool.(zeros(length(list)))
    # get all dependent constraint indices for the vars in free_vars
    for v in free_vars
        idxs_i, new_free_vars_i = get_dependency(v, list, exclude_vars)
        idxs .|= idxs_i
        if length(new_free_vars_i) > 0
            push!(all_free_vars, new_free_vars_i...)
        end
    end
    new_free_vars = setdiff(all_free_vars, free_vars)
    old_free_vars = setdiff(all_free_vars, new_free_vars)
    if length(new_free_vars) == 0
        return idxs
    else
        if length(old_free_vars) > 0
            push!(exclude_vars, old_free_vars...)
        end
        idxs .|= get_all_dependecies(new_free_vars, list, exclude_vars) 
        return idxs
    end
end

function special_compare(v1, v2)
    # returns TRUE is v1 <= v2
    # for variables of the form: :v_1, :v_2, etc. 
    v1num = Meta.parse(match(r"(?<=_)(.*)", string(v1)).captures[1])
    v2num = Meta.parse(match(r"(?<=_)(.*)", string(v2)).captures[1])
    return v1num <= v2num 
end

# iterate through pieces of og function
# (imagine sorted now but doesn't have to be i think)
# e.g. take v1 = f(x)
function check_overapprox(oa, domain, input_vars, problem_name, jobs=1)
    # NOTE THIS CODE RELIES ON THE VARIABLE ORDERING!
    vars = sort!(collect(keys(oa.fun_eq))) # sort just there for debugging
    R = true
    for vari in vars
        ϕ = [:($vari == $(oa.fun_eq[vari]) )]
        # for each variable vi in og func, consider all approx constraints that have ALL vars leq to it (plus input var(s))
        all_approx_constraints = [oa.approx_eq..., oa.approx_ineq...]
        all_approx_constraints = [((constraint.args[1] == :≤) || (constraint.args[1] == :≦) ) ? :($(constraint.args[2]) <= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]
        all_approx_constraints = [((constraint.args[1] == :≥) || (constraint.args[1] ==  :≧)) ? :($(constraint.args[2]) >= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]

        all_dependencies = all_approx_constraints[get_all_dependecies([vari], all_approx_constraints, [])]

        defs = []
        ϕ̂ = []
        for c in all_dependencies
            vars_in_c = Expr.(free_symbols(Basic(c))) # use SymEngine free_symbols function
            if vari ∉ vars_in_c
                push!(defs, c)
            else
                push!(ϕ̂, c)
            # things that go in phi hat are those with the output variable in them? e.g. v6 ? (everything else in defs?) [kind of like mode 2 but only 1 piece at a time for the og func?]
            end
        end

        sq = SoundnessQuery(ϕ, # ϕ
                            defs,
                            ϕ̂, # definitions for ϕ̂
                            domain)

        result = check("dreal", sq, problem_name*"_soundness_query_"*string(vari)*".smt2", δ=0.001, jobs=jobs) # TODO: pass dreal delta
        println("result for var ",vari," is: ", result)
        R &= occursin("unsat", result)
        readline() # to make things interactive
    end
    R ? println("all checks pass for "*problem_name*"!") : println("Some checks fail :( for "*problem_name)
    return R
end
#check_overapprox(oa, domain, input_vars)
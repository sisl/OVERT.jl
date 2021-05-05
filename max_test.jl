using JuMP
using LazySets
using Gurobi
using LinearAlgebra: dot
using IntervalArithmetic
N = 1
aux_var_counter = 1

function get_jump_var(model::JuMP.Model, var::Symbol, var2JuMPmap::Dict, var_ranges::Dict)
    if var in keys(var2JuMPmap)
        mip_var = var2JuMPmap[var]
    else
        l, u = var_ranges[var]
        name = "$var"
        mip_var = @variable(model, lower_bound=l, upper_bound=u, base_name=name)
        var2JuMPmap[var] = mip_var
    end
    return mip_var
end

function get_bound(h::HalfSpace, ranges)
    # ranges is a list of pairs like: [[l1, u1], [l2, u2]]
    intervals = [r[1]..r[2] for r in ranges] # using IntervalArithmetic.Interval type with .. syntax
    output = h.b - dot(h.a, intervals) # b - ax
    return [output.lo, output.hi]
end

# note: may have to solve non-trivial mixed integer program to get bounds on args to max...

function add_max(model, args::Array, arg_bounds)
    # takes JuMP expressions, returns JuMP variable 
    # arg bounds is an array [[l1, l2, l3...], [u1, u2, u3...]] of lower and upper bounds 
    lower_bounds, upper_bounds = arg_bounds
    # eliminate those variables where u_i ≤ l_max 
    l_max = maximum(lower_bounds)
    x_idx_to_elim = findall(upper_bounds .< l_max)
    println("eliminating the following: $x_idx_to_elim")
    good_idx = filter( x-> x ∉ x_idx_to_elim, 1:length(args))
    if length(good_idx) > 1
        return add_max_to_JuMP_helper(model, args[good_idx], [lower_bounds[good_idx], upper_bounds[good_idx]])
    else
        return args[good_idx]
    end
end
function get_u_max_i(us::Array, i)
    # get max excluding ith value
    return maximum(us[[1:(i-1)...,(i+1):end...]])
end
function add_max_to_JuMP_helper(model, x::Array, x_bounds)
    # max encoding from MNIPverify paper by Tjeng, Xiao, and Tedrake
    # x is a vector of variables
    # only considering "good" indices
    # unpack
    lower_bounds, upper_bounds = x_bounds
    # generate output variable 
    name = "max_ouput_$N"
    y = @variable(model, lower_bound=minimum(lower_bounds), upper_bound=maximum(upper_bounds), base_name=name)
    deltas = []
    # add constraints 
    for i = 1:length(x)
        # if this is an index we want to consider
        u_max_i = get_u_max_i(upper_bounds, i)
        # use get_aux_var in OVERT version
        δᵢ = @variable(model, binary=true, base_name="δ_$aux_var_counter")
        push!(deltas, δᵢ)
        lᵢ = lower_bounds[i]
        @constraint(model, y <= x[i] + (1 - δᵢ)*(u_max_i - lᵢ))
        @constraint(model, y >= x[i])
    end
    @constraint(model, sum(deltas) == 1)
    return y
end

#####################################################
# We have HalfSpace constraints of the form: ax - b <= 0
# if this constraint holds, we are in an "avoid set"
# we can rewrite as: b - ax >=0
# we then take multiple expressions and look at their max:
# max(b₁ - a₁x, b₂ - a₂x, ...)
# if max(b₁ - a₁x, b₂ - a₂x, ...) > 0 
# then at least one of
# the "avoid sets" has been reached :/
# but if the problem is unsat -- no avoid set has been reached!
# What's nice about this formulation is that the avoid sets can be 
# mutually exclusive, the disjunction in the max
# handles it
#####################################################
vars = [:x, :y]
var_ranges = Dict(:x => [0.,1.], :y=> [-2. , 1.2])
var2JuMPmap = Dict() # maps symbols -> JuMP variables
# we have halfspace constraints of the form: [c1, c2]' * [x, y] <= b where b is a scalar
h1 = HalfSpace([100., -1.], -15.)
h2 = HalfSpace([20., -0.1], -5.)
h3 = HalfSpace([120., 1.], -5.)
constraints = [h1, h2, h3]

model = JuMP.Model(Gurobi.Optimizer)

jump_vars = [get_jump_var(model, v, var2JuMPmap, var_ranges) for v in vars]
exprs = [c.b - dot(c.a, jump_vars) for c in constraints] 
expr_bounds = [get_bound(c, [var_ranges[v] for v in vars]) for c in constraints]
expr_bounds = hcat(expr_bounds...)[1,:], hcat(expr_bounds...)[2,:] 
max_out = add_max(model, exprs, expr_bounds)
@constraint(model, max_out >= 0)
optimize!(model)
termination_status(model)

vals = value.(jump_vars)

h1.b - dot(h1.a, vals)
h2.b - dot(h2.a, vals)
h3.b - dot(h3.a, vals)



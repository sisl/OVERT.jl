using JuMP
using LazySets
using Gurobi
using LinearAlgebra: dot
using IntervalArithmetic

vars = [:x, :y]
var_ranges = Dict(:x => [0.,1.], :y=> [-2. , 1.2])
var2JuMPmap = Dict() # maps symbols -> JuMP variables
# we have halfspace constraints of the form: [c1, c2]' * [x, y] <= b where b is a scalar
h1 = HalfSpace([1., -1.], 20.)
h2 = HalfSpace([-0.2, -0.1], 10.)
h3 = HalfSpace([-12., 1.], 6.)
constraints = [h1, h2, h3]

model = JuMP.Model(Gurobi.Optimizer)

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
    output = dot(h.a, intervals) - h.b # ax - b <= 0
    return [output.lo, output.hi]
end

jump_vars = [get_jump_var(model, v, var2JuMPmap, var_ranges) for v in vars]
exprs = [dot(c.a, jump_vars) - c.b for c in constraints] 
expr_bounds = [get_bound(c, [var_ranges[v] for v in vars]) for c in constraints]
expr_bounds = hcat(expr_bounds...)[1,:], hcat(expr_bounds...)[2,:] 
max_out = add_max(model, exprs, expr_bounds)
@constraint(model, max_out >= 0)
optimize!(model)
termination_status(model)

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



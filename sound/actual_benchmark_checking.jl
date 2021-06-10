include("sound/proof_functions.jl")
include("models/acc/acc.jl")
include("models/car/simple_car.jl")
include("models/single_pendulum/single_pend.jl")
include("models/tora/tora.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
using JLD2
using LaTeXStrings
file_dir = "sound"
file_dir = join(split(@__FILE__, "/")[1:end-1], "/") # get directory of this script
println("Writing results to $file_dir directory.")
ϵ = 0.1
δ = 0.001

function sincos(ϵ, δ; N=-1)
    dyn = :(sin(cos(x)))
    x = [0, π]
    domain = Dict(zip([:x], [x]))
    oa = overapprox_nd(dyn, domain, N=N, ϵ=ϵ)
    t = time()
    result = check_overapprox(oa, domain, [:x], "sincosexample", jobs=2, delta_sat=δ)
    Δt = time() - t
    return result, Δt
end

function sincosysq(ϵ, δ; N=-1, jobs=4)
    dyn = :(sin(cos(x + y^2)))
    x_domain = [0, π]
    y_domain = [-0.25, 1]
    domain = Dict(zip([:x, :y], [x_domain, y_domain]))
    oa = overapprox_nd(dyn, domain, N=N, ϵ=ϵ)
    t = time()
    result = check_overapprox(oa, domain, [:x, :y], "sincos_ysq_example", jobs=jobs, delta_sat=δ)
    Δt = time() - t

    # write to file
    file = open(file_dir*"/sinccosysq_time.txt", "w")
    write(file, "Time to check validity of overapproximation using dreal: ϵ= "*string(ϵ)*", δ="*string(δ)*"\n")
    write(file, "sincosysq check time (sec): "*string(Δt)*"\n")
    write(file, "jobs = "*string(jobs)*"\n")
    write(file, "N=$N\n")
    write(file, "All unsat? $result\n")
    close(file)

    return result, Δt
end

function xy(ϵ, δ; N=-1, jobs=2)
    dyn = :(x*y)
    x = [-1.5, -1.4] # x = [-1.5, 3.5]
    y = [-0.738, -0.6663] # y = [-1.2, 2.2]
    domain = Dict(zip([:x, :y], [x, y]))
    oa = overapprox_nd(dyn, domain, N=N, ϵ=ϵ)
    t = time()
    result = check_overapprox(oa, domain, [:x, :y], "xy_example", jobs=jobs, delta_sat=δ)
    Δt = time() - t
    return result, Δt
end 

function xy_SAT(ϵ, δ; N=-1, jobs=2)
    # expect this to produce bad bounds and return SAT
    dyn = :(x*y)
    x = [-1.5, -1.4] # x = [-1.5, 3.5]
    y = [-0.738, -0.6663] # y = [-1.2, 2.2]
    domain = Dict(zip([:x, :y], [x, y]))
    oa = overapprox_nd(dyn, domain, N=N, ϵ=-ϵ) # < note how I am NEGATING ϵ
    t = time()
    result = check_overapprox(oa, domain, [:x, :y], "xy_SAT_example", jobs=jobs, delta_sat=δ)
    Δt = time() - t
    return result, Δt
end 

function check_sat_xy(δ; jobs=2)
    # expect result to be sat
    ϕ = [:(y == log(x))]
    ϕ̂ = [:(y == x)]
    defs = []
    x = [1, 2.2]
    domain = Dict(zip([:x], [x]))
    sq = SoundnessQuery(ϕ, # ϕ
                        defs,
                        ϕ̂, # definitions for ϕ̂
                        domain)

    result = check("dreal", sq, "expect_sat_soundness_query.smt2", δ=δ, jobs=jobs) # TODO: pass dreal delta
    println("result is: ", result)
    R = occursin("unsat", result)

    R ? println("all checks pass for check_sat_xy!") : println("Some checks fail :( for check_sat_xy")
    return R
end

function check_xy_whole(ϵ, δ; N=-1, jobs=56)
    # not the right way to do this. expect sat.
    expr = :(x*y)
    x_range = [-1.5, -1.4] # x = [-1.5, 3.5]
    y_range = [-0.738, -0.6663] # y = [-1.2, 2.2]
    domain = Dict(zip([:x, :y], [x_range, y_range]))
    oa = overapprox_nd(expr, domain, N=N, ϵ=ϵ) 

    ϕ = [:($(oa.output) == $expr)]
    ####
    all_approx_constraints = [oa.approx_eq..., oa.approx_ineq...]
    all_approx_constraints = [((constraint.args[1] == :≤) || (constraint.args[1] == :≦) ) ? :($(constraint.args[2]) <= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]
    all_approx_constraints = [((constraint.args[1] == :≥) || (constraint.args[1] ==  :≧)) ? :($(constraint.args[2]) >= $(constraint.args[3])) : constraint for constraint in all_approx_constraints]
    ####
    # all_dependencies = all_approx_constraints[get_all_dependecies([oa.output], all_approx_constraints, [:x, :y], [])]
    ## ^ should be all constraints :// but isn't...
    ϕ̂_idx = map( x-> oa.output ∈ Expr.(free_symbols(Basic(x))), all_approx_constraints)
    ϕ̂ = all_approx_constraints[ϕ̂_idx]
    defs = all_approx_constraints[.!ϕ̂_idx]

    sq = SoundnessQuery(ϕ, # ϕ
                        defs,
                        ϕ̂, # definitions for ϕ̂
                        domain)

    result = check("dreal", sq, "mult_soundness_query.smt2", δ=δ, jobs=jobs) # TODO: pass dreal delta
    println("result is: ", result)
    R = occursin("unsat", result)

    R ? println("all checks pass for mult_xy!") : println("Some checks fail :( for mult_xy")
    return R
    # why is this sat?? :(( Because phihat isn't constrained to take on any particular value in it's range at any particular time. the correct way to do this is to propagate bounds. And even then only 1 nesting level of the overapprox can be checked at once without teh use of quantifiers.
end

# slows down...
function xcosy(ϵ, δ; N=-1, jobs=56)
    dyn = :(x*cos(y))
    x = [-1., 1.] #[-1.5, -1.4]
    y = [2., 3.0]#[2.3, 2.4]
    domain = Dict(zip([:x, :y], [x, y]))
    oa = overapprox_nd(dyn, domain, N=N, ϵ=ϵ)
    t = time()
    result = check_overapprox(oa, domain, [:x, :y], "xcosy_example", jobs=jobs, delta_sat=δ)
    Δt = time() - t
    return result, Δt
end 

# runs on laptop!
function acc(ϵ, δ; jobs=4)
    # Checking acc
    # In ACC, the derivaties of dimensions 3 and 6 are overapproximated
    x_lead = [90,110]
    v_lead = [32, 32.2]
    gamma_lead = [0,0]
    x_ego = [10,11]
    v_ego = [0, 20]#[30, 30.2]
    gamma_ego = [0, 0]
    input_domains = [x_lead, v_lead, gamma_lead, x_ego, v_ego, gamma_ego]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    bounds = find_controller_bound(file_dir*"/../nnet_files/jair/acc_controller.nnet", input_set, Id()) # returns 1D Hyperrectangle (interval)

    domain = Dict(zip(acc_input_vars, input_domains))
    domain[acc_control_vars[1]] = [low(bounds)..., high(bounds)...] 
    # construct overapproximation   
    oa_dim3 = overapprox_nd(acc_ẋ₃, domain, N=-1, ϵ=ϵ)
    oa_dim6 = overapprox_nd(acc_ẋ₆, domain, N=-1, ϵ=ϵ)

    println("Using $jobs jobs")
    result_dim3 = check_overapprox(oa_dim3, domain, [acc_input_vars..., acc_control_vars...], "acc_dim3", jobs=jobs, delta_sat=δ)
    result_dim6 = check_overapprox(oa_dim6, domain, [acc_input_vars..., acc_control_vars...], "acc_dim6", jobs=jobs, delta_sat=δ)
    return result_dim3 && result_dim6
end

# first several checks run on laptop, then starts to slow down. 
# ran for a month on 25 cores on jodhpur and didn't finish :///
function simple_car(ϵ, δ; N=-1, jobs=4)
    posx = [9.5, 9.55]
    posy = [-4.5, -4.45]
    yaw = [2.1, 2.11]
    vel = [1.5, 1.51]
    input_domains =  [posx, posy, yaw, vel]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    c_bounds = find_controller_bound(file_dir*"/../nnet_files/jair/car_smallest_controller.nnet", input_set, Id()) # returns 2D
    println("Found controller bound.")
    domain = Dict(zip(simple_car_input_vars, input_domains))
    domain[simple_car_control_vars[1]] = [low(c_bounds)[1]..., high(c_bounds)[1]...]  # u is 2d
    domain[simple_car_control_vars[2]] = [low(c_bounds)[2]..., high(c_bounds)[2]...]  # u is 2d

    oa_dim1 = overapprox_nd(simple_car_ẋ, domain, N=N, ϵ=ϵ)
    oa_dim2 = overapprox_nd(simple_car_ẏ, domain, N=N, ϵ=ϵ)

    result_dim1 = check_overapprox(oa_dim1, domain, [simple_car_input_vars..., simple_car_control_vars...], "simple_car_dim1", jobs=56, delta_sat=δ)
    println("Finished checking dim 1")
    result_dim2 = check_overapprox(oa_dim2, domain, [simple_car_input_vars..., simple_car_control_vars...], "simple_car_dim2", jobs=56, delta_sat=δ)
    return result_dim1 && result_dim2
end

# runs on laptop!
function single_pendulum(ϵ, δ; jobs=4)
    θ = [1., 1.2]
    θ_dot = [0., 0.2]
    input_domains = [θ, θ_dot]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    # get controller bounds
    bounds = find_controller_bound(file_dir*"/../nnet_files/jair/single_pendulum_small_controller.nnet", input_set, Id()) # returns 1D
    domain = Dict(zip(single_pend_input_vars, input_domains))
    domain[single_pend_control_vars[1]] = [low(bounds)..., high(bounds)...]  # u is 1d
    oa = overapprox_nd(single_pend_θ_doubledot, domain, N=-1, ϵ=ϵ)

    println("Using $jobs jobs")
    result = check_overapprox(oa, domain, [single_pend_input_vars..., single_pend_control_vars...], "single_pend", jobs=jobs, delta_sat=δ)
    return result
end

# runs on laptop!!!
function tora(ϵ, δ; jobs=4)
    x1 = [0.6, 0.7]
    x2 = [-0.7, -0.6]
    x3 = [-0.4, -0.3]
    x4 = [0.5, 0.6]
    input_domains = [x1, x2, x3, x4]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    # get controller bounds
    bounds = find_controller_bound(file_dir*"/../nnet_files/jair/tora_smallest_controller.nnet", input_set, Id()) # returns 1D
    domain = Dict(zip(tora_input_vars, input_domains))
    domain[tora_control_vars[1]] = [low(bounds)..., high(bounds)...]  # u is 1d
    oa = overapprox_nd(tora_dim2, domain, N=-1, ϵ=ϵ)

    println("Using $jobs jobs")
    result = check_overapprox(oa, domain, [tora_input_vars..., tora_control_vars...], "tora", jobs=jobs, delta_sat=δ)
    return result
end

function run_benchmarks_and_time(ϵ, δ; jobs=4)
    t = time()
    acc(ϵ, δ; jobs=jobs)
    acc_Δt = time() - t
    println("acc done")
    #
    # simple_car(ϵ, δ)
    # println("simple car done")
    #
    t = time()
    single_pendulum(ϵ, δ; jobs=jobs)
    sing_pend_Δt = time() - t
    println("single pendulum done")
    #
    t = time()
    tora(ϵ, δ; jobs=jobs)
    tora_Δt = time() - t
    println("tora done")

    # write to file
    file = open(file_dir*"/benchmark_times.txt", "w")
    write(file, "Time to check validity of overapproximation using dreal: ϵ= "*string(ϵ)*", δ="*string(δ)*"\n")
    write(file, "acc check time (sec): "*string(acc_Δt)*"\n")
    write(file, "single pendulum check time (sec): "*string(sing_pend_Δt)*"\n")
    write(file, "tora check time (sec): "*string(tora_Δt)*"\n")
    write(file, "jobs = "*string(jobs)*"\n")
    close(file)
end

function run_sin_cos(ϵ, δ; N=-1)
    result, Δt = sincos(ϵ, δ, N=N)
    file = open(file_dir*"/sincos_time_Nis"*string(N)*".txt", "w")
    write(file, "Time to check validity of overapproximation using dreal: ϵ= "*string(ϵ)*", δ="*string(δ)*"\n")
    write(file, "Result = "*string(result)*"\n")
    write(file, "sincos time (sec): "*string(Δt)*"\n")
    write(file, "N="*string(N))
    close(file)
end

function run_xy(ϵ, δ; N=-1)
    result, Δt = xy(ϵ, δ, N=N)
    file = open(file_dir*"/xy_time_Nis"*string(N)*".txt", "w")
    write(file, "Time to check validity of overapproximation using dreal: ϵ= "*string(ϵ)*", δ="*string(δ)*"\n")
    write(file, "Result = "*string(result)*"\n")
    write(file, "xy time (sec): "*string(Δt)*"\n")
    write(file, "N="*string(N))
    close(file)
end

function run_simple_car(ϵ, δ; N=-1)
    simple_car(ϵ, δ)
    println("simple car done")
end

# run_sin_cos(ϵ, δ, N=1)
# run_xy(ϵ, δ; N=1)
# xy_SAT(ϵ, δ)
# check_sat_xy(δ)
# xcosy(ϵ, δ)
# run_benchmarks_and_time(ϵ, δ; jobs=1)
#run_simple_car(ϵ, δ, N=1)
sincosysq(ϵ, δ; N=1, jobs=1)


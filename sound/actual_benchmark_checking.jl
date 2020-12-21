include("proof_functions.jl")
include("../models/acc/acc.jl")
include("../models/car/simple_car.jl")
include("../models/single_pendulum/single_pend.jl")
include("../models/tora/tora.jl")
include("../MIP/src/overt_to_mip.jl")
include("../MIP/src/mip_utils.jl")

# runs on laptop!
function acc()
    # Checking acc
    # In ACC, the derivaties of dimensions 3 and 6 are overapproximated
    x_lead = [90,110]
    v_lead = [32, 32.2]
    gamma_lead = [0,0]
    x_ego = [10,11]
    v_ego = [30, 30.2]
    gamma_ego = [0, 0]
    input_domains = [x_lead, v_lead, gamma_lead, x_ego, v_ego, gamma_ego]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    bounds = find_controller_bound(pwd()*"/../nnet_files/jair/acc_controller.nnet", input_set, Id()) # returns 1D Hyperrectangle (interval)

    domain = Dict(zip(acc_input_vars, input_domains))
    domain[acc_control_vars[1]] = [low(bounds)..., high(bounds)...] 
    # construct overapproximation   
    oa_dim3 = overapprox_nd(acc_ẋ₃, domain, N=-1, ϵ=.1)
    oa_dim6 = overapprox_nd(acc_ẋ₆, domain, N=-1, ϵ=.1)

    result_dim3 = check_overapprox(oa_dim3, domain, [acc_input_vars..., acc_control_vars...], "acc_dim3", jobs=2, delta_sat=0.001)
    result_dim6 = check_overapprox(oa_dim6, domain, [acc_input_vars..., acc_control_vars...], "acc_dim6", jobs=2, delta_sat=0.001)
    return result_dim3 && result_dim6
end

# first several checks run on laptop, then starts to slow down. TODO: test on server
function simple_car()
    posx = [9.5, 9.55]
    posy = [-4.5, -4.45]
    yaw = [2.1, 2.11]
    vel = [1.5, 1.51]
    input_domains =  [posx, posy, yaw, vel]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    bounds = find_controller_bound(pwd()*"/../nnet_files/jair/car_smallest_controller.nnet", input_set, Id()) # returns 2D
    domain = Dict(zip(simple_car_input_vars, input_domains))
    domain[simple_car_control_vars[1]] = [low(bounds)[1]..., high(bounds)[1]...]  # u is 2d
    domain[simple_car_control_vars[2]] = [low(bounds)[2]..., high(bounds)[2]...]  # u is 2d

    oa_dim1 = overapprox_nd(simple_car_ẋ, domain, N=-1, ϵ=.1)
    oa_dim2 = overapprox_nd(simple_car_ẏ, domain, N=-1, ϵ=.1)

    result_dim1 = check_overapprox(oa_dim1, domain, [simple_car_input_vars..., simple_car_control_vars...], "simple_car_dim1", jobs=10, delta_sat=0.001)
    result_dim2 = check_overapprox(oa_dim2, domain, [simple_car_input_vars..., simple_car_control_vars...], "simple_car_dim2", jobs=10, delta_sat=0.001)
    return result_dim1 && result_dim2
end

# runs on laptop!
function single_pendulum()
    θ = [1., 1.2]
    θ_dot = [0., 0.2]
    input_domains = [θ, θ_dot]
    input_set = Hyperrectangle(low=[i[1] for i in input_domains], high=[i[2] for i in input_domains])
    # get controller bounds
    bounds = find_controller_bound(pwd()*"/../nnet_files/jair/single_pendulum_small_controller.nnet", input_set, Id()) # returns 1D
    domain = Dict(zip(single_pend_input_vars, input_domains))
    domain[single_pend_control_vars[1]] = [low(bounds)..., high(bounds)...]  # u is 1d
    oa = overapprox_nd(single_pend_θ_doubledot, domain, N=-1, ϵ=.1)

    result = check_overapprox(oa, domain, [single_pend_input_vars..., single_pend_control_vars...], "single_pend", jobs=1, delta_sat=0.001)
    return result
end

function tora()
end

#acc()
#simple_car()
#single_pendulum()
#tora()
include("proof_functions.jl")
include("../models/acc/acc.jl")
include("../models/car/simple_car.jl")
include("../models/single_pendulum/single_pend.jl")
include("../models/tora/tora.jl")

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
    domain = Dict(zip(acc_input_vars, input_domains))
    # construct overapproximation   
    oa_dim3 = overapprox_nd(acc_ẋ₃, domain, N=-1, ϵ=.1)
    oa_dim6 = overapprox_nd(acc_ẋ₆, domain, N=-1, ϵ=.1)

    result_dim3 = check_overapprox(oa_dim3, domain, acc_input_vars, "acc_dim3", jobs=10)
    result_dim6 = check_overapprox(oa_dim6, domain, acc_input_vars, "acc_dim6", jobs=10)
    return result_dim3 && result_dim6
end

function simple_car()
    posx = [9.5, 9.55]
    posy = [-4.5, -4.45]
    yaw = [2.1, 2.11]
    vel = [1.5, 1.51]
    domain = Dict(zip(simple_car_input_vars, [posx, posy, yaw, vel]))
    oa_dim1 = overapprox_nd(simple_car_ẋ, domain, N=-1, ϵ=.1)
    oa_dim2 = overapprox_nd(simple_car_ẏ, domain, N=-1, ϵ=.1)

    result_dim1 = check_overapprox(oa_dim1, domain, simple_car_input_vars, "simple_car_dim1", jobs=10)
    result_dim2 = check_overapprox(oa_dim2, domain, simple_car_input_vars, "simple_car_dim2", jobs=10)
    return result_dim1 && result_dim2
end

acc()
simple_car()
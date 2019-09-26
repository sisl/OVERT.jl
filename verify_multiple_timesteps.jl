using NeuralVerification, LazySets
using BSON: @load
using Flux, JuMP

@load "closed_loop_controller.bson" total_network
controller = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))
@load "controller_weights.bson" weights_c
Flux.loadparams!(controller, weights_c)

NV = NeuralVerification

# load total_network, controller, etc.
init = Float64.(Tracker.data(controller[2].init))
nvnet = NV.network(total_network)

bignum = 1e7

# s := [θ, θ_dot_LB, θ_dot_UB, ℓ1, ℓ2, ℓ3, ℓ4]
input_set = Hyperrectangle(low = [-deg2rad(5); -deg2rad(1); init],
                           high = [deg2rad(5); deg2rad(1); init])

output_set = Hyperrectangle(low = [-deg2rad(6); -deg2rad(45); -deg2rad(45); fill(-bignum, size(init))],
                            high = [deg2rad(6); deg2rad(45); deg2rad(45); fill(bignum, size(init))])


prob = Problem(nvnet, input_set, output_set)

# solve(ReluVal(max_iter = 1000), prob)

#
function convex_nv(solver, net, input, output)
    res = nothing
    for hs in constraints_list(output)
        prob = Problem(net, input, hs)
        res = solve(solver, prob)
        @show res
        res.status == :violated && return res
    end
    res
end


import NeuralVerification: init_neurons, init_deltas, add_complementary_set_constraint!,
get_bounds, encode_network!, BoundedMixedIntegerLP, max_disturbance!

function chain_convex_nv(solver, net, input, output, steps)
    res = nothing
    for hs in constraints_list(output)
        prob = Problem(net, input, hs)
        res = chain_solve(solver, prob, steps)
        @time @show res
        res.status == :violated && return res
    end
    res
end


function chain_solve(solver, problem, steps,
                     io_relation = Dict(:eq => [1=>1; 4:7 .=> 3:6],
                                        :leq => [2 => 2],
                                        :geq => [3 => 2]))

    model, neurons = chain_encode(solver, problem, steps, io_relation)
    o = max_disturbance!(model, first(neurons) - problem.input.center)
    println("initial optimize call")
    optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
        return AdversarialResult(:holds)
    elseif termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED
        println("we're maybe infeasible")
        model, _ = chain_encode(problem, steps, io_relation)
        println("second optimize call")
        optimize!(model)
        if termination_status(model) ∈ (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            return AdversarialResult(:holds)
        end
    end
    println("we're not infeasible, or we were and we're not anymore")
    if termination_status(model) == MOI.OPTIMAL
        if value(o) >= maximum(problem.input.radius)
            return AdversarialResult(:holds)
        else
            return AdversarialResult(:violated, value(o))
        end
    end
    error("WTF. return code $(termination_status(model))")
end


function chain_encode(solver::MIPVerify, problem, steps,
                      io_relation = Dict(:eq => [1=>1; 4:7 .=> 3:6],
                                         :leq => [2 => 2],
                                         :geq => [3 => 2]))

    solver = MIPVerify()
    model = Model(solver)
    first_T_neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    bounds = get_bounds(problem)
    encode_network!(model, problem.network, first_T_neurons, deltas, bounds, BoundedMixedIntegerLP())

    neurons = first_T_neurons
    for _ in 1:steps-1

        OUT = last(neurons)

        # Use output of old bounds to get new bounds
        B = last(bounds)
        lb, ub = low(B), high(B)
        l, h = extrema([lb[2:3]; ub[2:3]])
        B = Hyperrectangle(low = [lb[1]; l; lb[4:end]], high = [ub[1]; h; ub[4:end]])
        bounds = get_bounds(problem.network, B)

        neurons = init_neurons(model, problem.network)
        deltas = init_deltas(model, problem.network)
        encode_network!(model, problem.network, neurons, deltas, bounds, BoundedMixedIntegerLP())

        IN = first(neurons)
        @constraint(model, OUT[1] == IN[1])
        @constraint(model, OUT[2] <= IN[2])
        @constraint(model, OUT[3] >= IN[2])
        @constraint(model, OUT[4:end] .== IN[3:end])
    end
    add_complementary_set_constraint!(model, problem.output, last(neurons))

    return model, first_T_neurons
end

import NeuralVerification: solve, find_relu_to_fix, type_one_broken, type_two_broken, type_one_repair,
type_two_repair, activation_constraint, activation_constraint, encode, enforce_repairs, reluplex_step


function chain_solve(solver::Reluplex, problem, steps, model = Model(with_optimizer(GLPK.Optimizer));
                      io_relation = Dict(:eq => [1=>1; 4:7 .=> 3:6],
                                         :leq => [2 => 2],
                                         :geq => [3 => 2]))

    optimize!(model)

    ẑ = model[:ẑ]
    z = model[:z]

    if isfeasible(model)
        i, j = find_relu_to_fix(ẑ, z)

        # In case no broken relus could be found, return the "input" as a counterexample
        i == 0 && return CounterExampleResult(:violated, value.(first(ẑ)))

        for repair_type in 1:2
            # Set the relu status to the current fix.
            relu_status[i][j] = repair_type
            new_m  = Model(solver)
            bs, fs = encode_all(new_m, problem, steps)
            enforce_repairs!(new_m, bs, fs, relu_status)

            result = chain_solve(solver, problem, steps, new_m)

            # Reset the relu when we're done with it.
            relu_status[i][j] = 0

            result.status == :violated && return result
        end
    else
        return CounterExampleResult(:holds)
    end
end

function get_next_bounds(problem, bounds = get_bounds(problem))
    B = last(bounds)
    lb, ub = low(B), high(B)
    l, h = extrema([lb[2:3]; ub[2:3]])
    B = Hyperrectangle(low = [lb[1]; l; lb[4:end]], high = [ub[1]; h; ub[4:end]])
    bounds = get_bounds(problem.network, B)
end

# function encode_i(model, problem, bounds = get_bounds(problem))
#     layers = problem.network.layers
#     ẑ = init_neurons(model, layers) # before activation
#     z = init_neurons(model, layers) # after activation
#     activation_constraint!(model, ẑ[1], z[1], Id())

#     for (i, L) in enumerate(layers)
#         @constraint(model, affine_map(L, z[i]) .== ẑ[i+1])
#         add_set_constraint!(model, bounds[i], ẑ[i])
#         activation_constraint!(model, ẑ[i+1], z[i+1], L.activation)
#     end

#     bounds = get_next_bounds(problem, bounds)

#     return ẑ, z, bounds
# end
# function encode_all(model, problem, steps)
#     ẑ, z, bounds = encode_i(model, problem)
#     for null in 1:steps-1
#         ẑᵢ, zᵢ, bounds = encode_i(model, problem, bounds)

#         OUT = last(z)
#         IN = first(ẑᵢ)
#         @constraint(model, OUT[1] == IN[1])
#         @constraint(model, OUT[2] <= IN[2])
#         @constraint(model, OUT[3] >= IN[2])
#         @constraint(model, OUT[4:end] .== IN[3:end])

#         append!(ẑ, ẑᵢ)
#         append!(z, zᵢ)
#     end
#     add_complementary_set_constraint!(model, problem.output, last(z))

#     model[:ẑ] = ẑ
#     model[:z] = z
#     ẑ, z
# end


# let solver = MIPVerify(optimizer = ECOS.Optimizer), output = constraints_list(output_set)[2]
#     problem = Problem(nvnet, input_set, output)

#     model = Model(solver)
#     neurons1 = init_neurons(model, problem.network)
#     deltas1 = init_deltas(model, problem.network)
#     bounds = get_bounds(problem)
#     encode_network!(model, problem.network, neurons1, deltas1, bounds, BoundedMixedIntegerLP())

#     B = last(bounds)
#     lb, ub = low(B), high(B)
#     l, h = extrema([lb[2:3]; ub[2:3]])
#     B = Hyperrectangle(low = [lb[1]; l; lb[4:end]], high = [ub[1]; h; ub[4:end]])
#     bounds = get_bounds(nvnet, B) # output is irrelevant
#     neurons2 = init_neurons(model, problem.network)
#     deltas2 = init_deltas(model, problem.network)

#     encode_network!(model, problem.network, neurons2, deltas2, bounds, BoundedMixedIntegerLP())

#     OUT = last(neurons1)
#     IN = first(neurons2)

#     @constraint(model, OUT[1] == IN[1])
#     @constraint(model, OUT[2] <= IN[2])
#     @constraint(model, OUT[3] >= IN[2])
#     @constraint(model, OUT[4:end] .== IN[3:end])

#     add_complementary_set_constraint!(model, problem.output, last(neurons2))
#     o = max_disturbance!(model, first(neurons1) - problem.input.center)

#     optimize!(model)
#     model, [neurons1; neurons2]
# end
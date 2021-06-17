include("compare_to_dreal.jl")
include("../../../models/tora/tora.jl")

state_vars = tora_input_vars
control_vars = [:u1]
input_set = Dict(:x1=>[1., 1.2], :x2=>[0., 0.2])
controller_file = "nnet_files/jair/single_pendulum_small_controller.nnet"
dynamics_map = Dict(single_pend_input_vars[1]=>single_pend_input_vars[2],
                    single_pend_input_vars[2]=>single_pend_θ_doubledot)
dt = 0.1
output_constraints = [:(x1 <= -0.2167)] # (avoid set)
N_steps=25
experiment_name = "single_pendulum_small_controller"
dirname="examples/jmlr/comparisons/"

ΔT = compare_to_dreal(state_vars, control_vars, input_set, controller_file, dynamics_map, dt, output_constraints, dirname, experiment_name, N_steps; jobs=28)include("compare_to_dreal.jl")



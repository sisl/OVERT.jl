include("compare_to_dreal.jl")
include("../../../models/tora/tora.jl")

state_vars = tora_input_vars
control_vars = tora_control_vars
# input_set_hyperrect = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
ranges = [[z...] for z in zip([0.6, -0.7, -0.4, 0.5], [0.7, -0.6, -0.3, 0.6])]
input_dict = Dict(zip(state_vars, ranges))
controller_file = "nnet_files/jair/tora_smallest_controller.nnet"
dynamics_map = Dict(state_vars[1]=>state_vars[2],
                    state_vars[2]=>tora_dim2,
                    state_vars[3]=>state_vars[4],
                    state_vars[4]=>tora_control_vars[1]
                    )
dt = 0.1
output_constraints = [:(x1 <= -2.0), :(x1 >= 2.0)] # (avoid set)
N_steps=15
experiment_name = "tora_smallest_controller"
dirname="examples/jmlr/comparisons/"

ΔT = compare_to_dreal(state_vars, control_vars, input_dict, controller_file, dynamics_map, dt, output_constraints, dirname, experiment_name, N_steps; jobs=28, dreal_path="/opt/dreal/4.20.12.1/bin/dreal")



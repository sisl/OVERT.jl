
## File description

1. julia files
    1. single_pendulum.jl: file to generate single pendulum overt dynamics. Assuming g=m=L=1 and c= 0.2
    2. double_penduluml.j: file to generate double pendulum overt dynamics. Assuming g=m=L=1 and c= 0.

2. trained models
    1. single_pend_controller_nn.h5: untrained model with 2 inputs, 1 output, (compatible with single pendulum) only for testing. shall be deleted.
    2. single_pend_nn_ilqr_data.h5: a 3 layer nn model trained for single pendulum. Layers are fully connected with 50x10x1 neuron sizes. Input data are angle and angular velocity. training was done using behavior clonning. The clonning data included 200 simulation examples generated using iterative linear quadratic regulator (ilqr). The initial state of pendulum is randomly choosen in range `[0, 2pi]x[-4pi, 4pi]`. The pendulum parameters are `m=1., L=1., g=1., c=0.2, dt=0.1, ntimesteps=25`.
    3. single_pend_nn_lqr_data.h5: same as single_pend_nn_ilqr_data.h5, except the data are generated using lqr and the initial points are the endpoint of the ilqr simulations.
    
3. overt dynamics model
    1. single_pend_acceleration_overt.h5: overt dynamics of a single pendulum (expression for angular acceleration).

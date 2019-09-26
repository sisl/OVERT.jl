using Flux, LinearAlgebra, Plots
using BSON: @save

include("utils.jl")
include("relubypass.jl")
include("autoline.jl")

"""
The assumed dynamics are
θ := θ + dt*θ_dot
θ_dot_lb := θ_dot + dt*(A*u + B*LBsin)
θ_dot_ub := θ_dot + dt*(A*u + B*UBsin)
"""

### FUNCTION DEFINITIONS ###
function mixing_layer(g=9.81, m=1, ℓ=0.85, dt=0.005)
    A = dt / (m*ℓ^2)
    B = g*dt / ℓ
    W_mixing = [0 0 1 dt 0;
                B 0 0 1 A;
                0 B 0 1 A]
    return Dense(W_mixing, zeros(3), identity)
end

### IMPLEMENTATION ###
# just can't be literal π because of SymEngine
mypi = round(pi, digits = 100)

upper_bound_pts = [(-2mypi, 0),
                   (-3mypi/2, mypi/2),
                   (-mypi, 0),
                   (-mypi/2, -1),
                   (0, 0),
                   (mypi/2, mypi/2),
                   (mypi, 0),
                   (3mypi/2, -1),
                   (2mypi, 0)]

lower_bound_pts = [(-2mypi, 0),
                   (-3mypi/2, 1),
                   (-mypi, 0),
                   (-mypi/2, -mypi/2),
                   (0, 0),
                   (mypi/2, 1),
                   (mypi, 0),
                   (3mypi/2, -mypi/2),
                   (2mypi, 0)]

upper_bound_sin = relu_bypass(to_network(upper_bound_pts))
lower_bound_sin = relu_bypass(to_network(lower_bound_pts))

# Stack into one net and plot to verify
sin_over = Base.vcat(lower_bound_sin, upper_bound_sin)
@save "sin_net.bson" sin_over

# first add 3 bypassed variables to the sin network. θ, θ_dot, and u
dynamics = add_bypass_variables(sin_over, 3)
# now add the mixing layer:
dynamics = Chain(dynamics..., mixing_layer())
dynamics = relu_bypass(dynamics)
@save "dynamics_model.bson" dynamics

### TESTING ###
x = -2pi:0.01:2pi
plot(x, hcat(sin_over.(x)...)'[:,1])
plot!(x, hcat(sin_over.(x)...)'[:,2])
plot!(x, sin.(x))
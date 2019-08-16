using Flux
using Plots, UnicodePlots
using Base.Iterators: partition
using LinearAlgebra
using BSON: @load

Base.:-(x::AbstractArray, y::Number) = x .- y
Base.:+(x::AbstractArray, y::Number) = x .+ y
Base.:-(x::Number, y::AbstractArray) = x .- y
Base.:+(x::Number, y::AbstractArray) = x .+ y

controller = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))
@load "controller_weights.bson" weights_c
Flux.loadparams!(controller, weights_c)

# function to keep angles in [-π, π]
fixrad(x) = (y = mod2pi(x); y > pi ? y-2pi : y)
fixdeg(x) = (y = mod(x, 360); y > 180 ? y-360 : y)

# Inverted pendulum dynamics parameters #
g = 9.81;  L = 0.85;  m = 1;
I = 8;     damp = 1;  w_n = 10;
dt = 0.005; T = 100  # length of episode

function desired_control(x)
   u = -(m*L^2 + I)*((w_n^2 + m*g*L/(m*L^2 + I))*x[1] + (damp*w_n)*x[2])
   return u
end

# simulate policy after training #
function sim(f, x, T, dt = 0.005)
   u = 0
   X = zeros(T, 2)
   for t in 1:T
      θ, θ_dot = x
      θ_ddot = u/(m*L^2) + (g/L)*sin(θ)
      u = f(x)
      x .+= [θ_dot, θ_ddot].*dt
      x[1] = fixrad(x[1])
      X[t, :] .= x
   end
   return X
end

function plotsim!(p, f = NN; dt = dt, T = 2T, s0 = [deg2rad(2), deg2rad(0)])
   s = sim(f, s0, T, dt)
   label = f == NN ? "NN" : "desired"
   plot!(p, 1:T, fixdeg.(rad2deg.(s[:, 1])), label = label, xlabel="Time Step", ylabel="Degrees")
end

plotsim(args...) = plotsim!(plot(), args...)
# plot a trace of time 'T' #
NN(x) = Tracker.data(controller(x))[1]
p = plotsim(NN)
plotsim!(p, desired_control)
[plotsim!(p, NN, s0 = [0.1*(rand() - 0.5), 0.05*(rand() - 0.5)]) for i in 1:500]
p



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
(D::Dense)(x::Number) = D.σ.(D.W*x + D.b)
FluxArr = AbstractArray{<:Union{Float32, Float64}, N} where N

@load "dynamics_model.bson" total
x = collect(-pi:0.01:pi)
p2 = plot(x, Tracker.data(hcat(total.(x)...)'[:,1]))
plot!(x, Tracker.data(hcat(total.(x)...)'[:,2]))
plot!(x, sin.(x))

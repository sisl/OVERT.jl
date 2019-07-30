using Flux
using Plots, UnicodePlots
using Base.Iterators: partition
using LinearAlgebra

function θ_ddot_desired(x, m, L, g)
   u = desired_control(x)
   return u/(m*L^2) + (g/L)*sin(x[1])
end

# function to keep angles in [-π, π]
fixrad(x) = (y = mod2pi(x); y > pi ? y-2pi : y)
fixdeg(x) = (y = mod(x, 360); y > 180 ? y-360 : y)

# function to generate traces at specified dt for training data #
# put Flux.train! in a for loop with different data each time to simulate epochs but with different data
function generate_x_train(N, dt, m, L, g)
   X_train = [zeros(2) for i in 1:N]
   X_train[1] = [0.1*(rand() - 0.5), 0.05*(rand() - 0.5)]
   for i in 2:N
      θ, θ_dot = x = X_train[i-1]
      X_train[i] = [fixrad(θ + θ_dot*dt),   θ_dot + θ_ddot_desired(x, m, L, g)*dt]
   end
   return X_train
end

# Overload operations for our purposes #
Base.:-(x::AbstractArray, y::Number) = x.-y
Base.:+(x::AbstractArray, y::Number) = x.+y
Base.:-(x::Number, y::AbstractArray) = x.-y
Base.:+(x::Number, y::AbstractArray) = x.+y


# Define network #
model = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 2, identity))

# Inverted pendulum dynamics parameters #
g = 9.81;  L = 0.85;  m = 1;
I = 8;     damp = 1;  w_n = 10;

# control output solved by linearizing around unstable equilibrium #
function desired_control(x)
   u = -(m*L^2 + I)*((w_n^2 + m*g*L/(m*L^2 + I))*x[1] + (damp*w_n)*x[2])
   return u
end

# squared loss #
function loss(x, y)
   # z = (model(x) - desired_control(x))[1]^2
   z = sum(Flux.mse.(model.(x), desired_control.(x)))
   s =  sim(x-> Tracker.data(model(x))[1], [deg2rad(10), deg2rad(0)], T, dt)[end, :]
   Flux.reset!(model)
   return z + 100*norm(s)
end


# training setup and execution #
dt = 0.005
opt = ADAM(0.001, (0.9, 0.999))
T = 100  # length of episode
num_eps = 2500
N = fill(T, num_eps)
Xs = generate_x_train.(N, dt, 1, 0.85, 9.81)
Ys = zeros.(length.(Xs))
data = collect(zip(Xs, Ys))
Flux.train!(loss,  Flux.params(model), data,  opt, cb = Flux.throttle(() -> @show(loss([[0.0, 0.1]], 0)), 1))

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
   plot!(p, 1:T, fixdeg.(rad2deg.(s[:, 1])), label = label)
end
plotsim(args...) = plotsim!(plot(), args...)
# plot a trace of time 'T' #
NN(x) = Tracker.data(model(x))[1]
p = plotsim(NN)
plotsim!(p, desired_control)
[plotsim!(p, NN, s0 = [0.1*(rand() - 0.5), 0.05*(rand() - 0.5)]) for i in 1:10]
p
# plot(1:T, fixdeg.(rad2deg.(X[:, 2])))
# lineplot(1:T, fixdeg.(rad2deg.(X[:, 1])))

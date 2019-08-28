using Flux, Plots, LinearAlgebra
using Base.Iterators: partition
using BSON: @save
include("utils.jl")
# To train a model first define the model. Ex: model = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))
# Second, run the function train_pendulum(model, num_eps, θ_mean, θ_dot_mean)
# You can view sim traces of the result by calling plotting(model)
# Model weights can be saved using save_weights(model)

const PARAMS = (m=1, L=0.85, I=8, wₙ=10, damp=1, g=9.81, T=100, dt=0.005)  # Inverted pendulum parameters

# function to keep angles in [-π, π]
fixrad(x) = (y = mod2pi(x); y > pi ? y-2pi : y)
fixdeg(x) = (y = mod(x, 360); y > 180 ? y-360 : y)

θ_ddot_desired(x) = desired_control(x)/(PARAMS.m*PARAMS.L^2) + (PARAMS.g/PARAMS.L)*sin(x[1])
desired_control(x) = -(PARAMS.m*PARAMS.L^2 + PARAMS.I)*((PARAMS.wₙ^2 + PARAMS.m*PARAMS.g*PARAMS.L/(PARAMS.m*PARAMS.L^2 + PARAMS.I))*x[1] + (PARAMS.damp*PARAMS.wₙ)*x[2])

function generate_x_train(N, θ_mean, θ_dot_mean)
   X_train = [zeros(2) for i in 1:N]
   X_train[1] = [θ_mean*(rand() - 0.5), θ_dot_mean*(rand() - 0.5)]
   for i in 2:N
      θ, θ_dot = x = X_train[i-1]
      X_train[i] = [fixrad(θ + θ_dot*PARAMS.dt),   θ_dot + θ_ddot_desired(x)*PARAMS.dt]
   end
   return X_train
end

function loss(x, y)
   z = sum(Flux.mse.(model.(x), desired_control.(x)))
   s = sim(x-> Tracker.data(model(x))[1], [deg2rad(10), deg2rad(0)], PARAMS.T)[end, :]
   Flux.reset!(model)
   return z + 100*norm(s)
end

function sim(f, x, T)
   u = 0
   X = zeros(T, 2)
   for t in 1:T
      θ, θ_dot = x
      θ_ddot = u/(PARAMS.m*PARAMS.L^2) + (PARAMS.g/PARAMS.L)*sin(θ)
      u = f(x)
      x .+= [θ_dot, θ_ddot].*PARAMS.dt
      x[1] = fixrad(x[1])
      X[t, :] .= x
   end
   return X
end

function plotsim!(p, f = NN; T = 2*PARAMS.T, s0 = [deg2rad(2), deg2rad(0)])
   plot!(p, 1:T, fixdeg.(rad2deg.(sim(f, s0, T)[:, 1])), legend = false, xlabel="Time Step", ylabel="Degrees")
end

function train_pendulum(model, num_eps, θ_mean, θ_dot_mean)
   N  = fill(PARAMS.T, num_eps)
   Xs = generate_x_train.(N, θ_mean, θ_dot_mean)
   Ys = zeros.(length.(Xs))
   data = collect(zip(Xs, Ys))
   opt  = ADAM(0.001, (0.9, 0.999))
   Flux.train!(loss,  Flux.params(model), data,  opt, cb = Flux.throttle(() -> @show(loss([[0.0, 0.1]], 0)), 1))
   return model
end

function plotting(model)
   # Plot resulting trajectories
   plotsim(args...) = plotsim!(plot(), args...)
   NN(x) = Tracker.data(model(x))[1]
   p = plotsim(NN)
   [plotsim!(p, NN, s0 = [deg2rad(10)*(rand() - 0.5), deg2rad(5)*(rand() - 0.5)]) for i in 1:100]; p
end

function save_weights(model)
   weights_c = Tracker.data.(Flux.params(model));
   @save "controller_weights.bson" weights_c
end

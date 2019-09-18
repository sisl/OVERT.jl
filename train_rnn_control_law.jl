using Flux, Plots, LinearAlgebra
using Base.Iterators: partition
using BSON: @save
include("utils.jl")
# To train a model first define the model. Ex: model = Chain(Dense(2, 4, relu), RNN(4, 4, relu), Dense(4, 1, identity))
# Second, run the function train_pendulum(model, num_eps, θ_var, θ_dot_var)
# You can view sim traces of the result by calling plotting(model)
# Model weights can be saved using save_weights(model)

const PARAMS = (m=1, L=0.85, I=8, wₙ=10, damp=1, g=9.81, T=100, dt=0.005)  # Inverted pendulum parameters

# function to keep angles in [-π, π]
fixrad(x) = (y = mod2pi(x); y > pi ? y-2pi : y)
fixdeg(x) = (y = mod(x, 360); y > 180 ? y-360 : y)

θ_ddot_desired(x) = desired_control(x)/(PARAMS.m*PARAMS.L^2) + (PARAMS.g/PARAMS.L)*sin(x[1])
desired_control(x) = -(PARAMS.m*PARAMS.L^2 + PARAMS.I)*((PARAMS.wₙ^2 + PARAMS.m*PARAMS.g*PARAMS.L/(PARAMS.m*PARAMS.L^2 + PARAMS.I))*x[1] + (PARAMS.damp*PARAMS.wₙ)*x[2])

random_start(θ_var, θ_dot_var) = [θ_var*(rand() - 0.5), θ_dot_var*(rand() - 0.5)]

function generate_x_train(N, θ_var, θ_dot_var)
   X_train = [zeros(2) for i in 1:N]
   X_train[1] = random_start(θ_var, θ_dot_var)
   for i in 2:N
      θ, θ_dot = x = X_train[i-1]
      X_train[i] = [fixrad(θ + θ_dot*PARAMS.dt),   θ_dot + θ_ddot_desired(x)*PARAMS.dt]
   end
   return X_train
end

function loss(x, y)
    u = model.(x)
    z = sum(Flux.mse.(u, desired_control.(x)))

    Flux.reset!(model)

    f = x′ -> Tracker.data(model(x′))[1] # gets control output for x0 input
    x0 = random_start(deg2rad(10), deg2rad(10))
    s = sim(f, x0, PARAMS.T)[end, :] # get the T timestep state for x0

    Flux.reset!(model)

    return z + 100*norm(s) + 5*sum(abs, vcat(u...))
end

function sim(f, x, T)
   X = zeros(T, 2)
   for t in 1:T
      u = f(x)
      θ, θ_dot = x
      θ_ddot = u/(PARAMS.m*PARAMS.L^2) + (PARAMS.g/PARAMS.L)*sin(θ)
      x .+= [θ_dot, θ_ddot].*PARAMS.dt
      x[1] = fixrad(x[1])
      X[t, :] .= x
   end
   return X
end

function train_pendulum(model, num_eps, θ_var, θ_dot_var)
   N  = fill(PARAMS.T, num_eps)
   Xs = generate_x_train.(N, θ_var, θ_dot_var)
   Ys = zeros.(length.(Xs))
   data = collect(zip(Xs, Ys))
   opt  = ADAM(0.001, (0.9, 0.999))
   Flux.train!(loss,  Flux.params(model), data,  opt, cb = Flux.throttle(() -> @show(loss([[0.0, 0.1]], 0)), 1))
   return model
end

function plotsim!(p, f = NN; T = 2*PARAMS.T, s0 = [deg2rad(2), deg2rad(0)], state_var = 1)
    ylabels = ("Degrees", "Degrees/sec")
    plot!(p, 1:T, fixdeg.(rad2deg.(sim(f, s0, T)[:, state_var])), legend = false, xlabel="Time Step", ylabel=ylabels[state_var])
end

plotsim(args...; kwargs...) = plotsim!(plot(), args...; kwargs...)

function plotting(model)
   NN(x) = Tracker.data(model(x))[1]
   p1 = plotsim(NN)
   [plotsim!(p1, NN, s0 = [deg2rad(10)*(rand() - 0.5), deg2rad(5)*(rand() - 0.5)]) for i in 1:100]
   p2 = plotsim(NN, state_var = 2)
   [plotsim!(p2, NN, s0 = [deg2rad(10)*(rand() - 0.5), deg2rad(5)*(rand() - 0.5)], state_var = 2) for i in 1:100]
   plot(p1, p2)
end

function save_weights(model)
   weights_c = Tracker.data.(Flux.params(model));
   @save "controller_weights.bson" weights_c
end

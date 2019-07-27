using Flux
using Plots, UnicodePlots

Base.:-(x::AbstractArray, y::Number) = x.-y
Base.:+(x::AbstractArray, y::Number) = x.+y
Base.:-(x::Number, y::AbstractArray) = x.-y
Base.:+(x::Number, y::AbstractArray) = x.+y

model = Chain(RNN(2, 2, relu), Dense(2, 1, identity))

g = 9.81;
L = 0.85;
m = 1;
I = 8;
ksi = 1;
w_n = 10;

desired_control(x) = -(m*L^2 + I)*((w_n^2 + m*g*L/(m*L^2 + I))*x[1] + (ksi*w_n)*x[2])
function loss(x, y)
   z = (model(x) - desired_control(x))[1]^2
   Flux.reset!(model)
   z
end
opt = ADAM(0.001, (0.9, 0.999))

N = 80000
x_train = [[pi*randn(), 2*randn()] for i in 1:N]
# x_train = [[0, 0.01] for i in 1:N]
y_train = zeros(N)
data = zip(x_train, y_train)
Flux.train!(loss,  Flux.params(model), data,  opt, cb = Flux.throttle(() -> @show(loss([0.0, 0.1], 0)), 1))

function sim(model, x, T, dt = 0.01)
   u = 0
   X = zeros(T, 2)
   for t in 1:T
      θ, θ_dot = x
      θ_ddot = u/(m*L^2) + (g/L)*sin(θ)
      u = Tracker.data(model(x))[1]
      # @show u
      x .+= [θ_dot, θ_ddot].*dt
      x[1] = fixrad(x[1])
      X[t, :] .= x
   end
   return X
end

fixrad(x) = (y = mod2pi(x); y > pi ? y - 2pi : y)

T = 1000
X = sim(model, [deg2rad(1), deg2rad(1)], T)
plot(1:T, rad2deg.(X[:, 1]))
lineplot(1:T, rad2deg.(X[:, 1]))

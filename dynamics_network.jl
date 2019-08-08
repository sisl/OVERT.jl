using Flux, LinearAlgebra

#### method 1, approximate step function
function bin(θ, I = 6)
    a = [θ - (i-1)*pi/2 for i in 1:I]
    relu.(a - 1e30*relu.(a .- pi/2))
end
function UB(θ, I = 6)
    D = Diagonal([1, -1, -2/pi, 2/pi, 1, -1])
    out = D*bin(θ, I)
    out + [(i-1)*pi/2 for i in 1:I]
end

##### method 2, learn it with flux
# Overload operations for our purposes NOTE: THIS IS TYPE PIRACY
Base.:-(x::AbstractArray, y::Number) = x .- y
Base.:+(x::AbstractArray, y::Number) = x .+ y
Base.:-(x::Number, y::AbstractArray) = x .- y
Base.:+(x::Number, y::AbstractArray) = x .+ y

model = Chain(Dense(1, 10, relu), Dense(10, 10, relu), Dense(10, 4, relu), Dense(4, 1, identity))
opt = ADAM(0.001, (0.9, 0.999))
Xs = map(x-> [x], 0:0.1:2pi)
data = collect(zip(Xs, foo.(Xs)))

function loss(x, y)
    return maximum(abs.(model(x) - y))
end

function foo(θ::Number)
    θ = mod2pi(θ)
    if 0 <= θ < pi
        y = -abs(θ - pi/2) + pi/2
    elseif pi <= θ <= 2pi
        y = 2/pi*abs(θ - 3pi/2) - 1
    end
    return y
end
foo(θ) = foo.(θ)


Flux.@epochs 500 Flux.train!(loss,  Flux.params(model), data,  opt)

function _removedim(X)
    v = eltype(X)()
    for x in X
        append!(v, x)
    end
    return v
end

xs = _removedim(Xs)
ys = _removedim(Tracker.data.(model.(Xs)))
# plot(xs, foo.(xs))
plot(xs, sin.(xs))
plot!(xs, ys)




##### method 3 hard coded netword
struct ReLUBypass{T}
    protected::T
end
ReLUBypass() = ReLUBypass(nothing)
ReLUBypass(args...) = ReLUBypass(collect(args))
# so that a ReLUBypass can be used as an activation function.
(RB::ReLUBypass{Nothing})(x) = identity(x)
function (RB::ReLUBypass)(x)
    out = relu.(x)
    out[RB.protected] = x[RB.protected]
    return out
end

Base.show(io::IO, RB::ReLUBypass{Nothing})  = print(io, "ReLUBypass(nothing)")
Base.show(io::IO, RB::ReLUBypass{<:Number}) = print(io, "ReLUBypass($(RB.protected))")
Base.show(io::IO, RB::ReLUBypass{<:Vector}) = print(io, "ReLUBypass$(Tuple(RB.protected))")

# Also type piracy:
# TODO to harmonize better with Flux, define a specialized broadcast behavior instead for ReLUBypass
# (D::Dense{<:ReLUBypass, A, B})(x) where {A,B} = D.σ(D.W*x + D.b)
(D::Dense)(x::Number) = D.σ.(D.W*x + D.b)
FluxArr = AbstractArray{<:Union{Float32, Float64}, N} where N
(D::Dense{<:ReLUBypass, <:FluxArr, B})(x::FluxArr) where B = D.σ(D.W*x + D.b)


# If it's anything other than a bypass do nothing
relu_bypass(L1::Dense, L2::Dense) where {A, B} = L1, L2
# if it's a ReLUBypass
function relu_bypass(L1::Dense{<:ReLUBypass, A, B}, L2::Dense) where {A, B}
    W, b, protected = L1.W, L1.b, L1.σ.protected
    if protected == nothing
        protected = collect(axes(b, 1))
    end
    n = size(W, 1)
    I = Matrix(LinearAlgebra.I, n, n)
    # `before` only protects the indices we want, and `after` undoes the transformation
    # `before` needs to be left-multiplied by the weights and `after` right-multiplied
    # I.e. the full thing:   W₂*B'*σ(B*(W₁*x + b₁)) + b₂
    before = [I; -I[protected, :]]
    after = before'

    L1_new = Dense(before*W, before*b, relu)
    L2_new = Dense(L2.W*after, L2.b, L2.σ)
    return L1_new, L2_new
end

function relu_bypass(C::Chain)
    C_bypassed = collect(C)
    for i in 1:length(C)-1
        C_bypassed[i:i+1] .= relu_bypass(C_bypassed[i], C_bypassed[i+1])
    end
    Chain(C_bypassed...)
end

#=
 Ax            -- [0, π/2]
-Ax + Aπ       -- [π/2, π]
-2Ax/π + 2A    -- [π, 3π/2]
 2Ax/π - 4A    -- [3π/2, π]
=#
A = 1
x = collect(0:0.01:2*pi)
z1 = @. 2*A*x - A*pi
z2 = @. 4*A*x/pi - 6*A

term1 = @. (A*pi - (relu.(z1) + relu.(-z1)))/2
term2 = @. (-2*A + (relu.(z2) + relu.(-z2)))/2
term3 = @. (term2 - (relu.(term2) + relu.(-term2)))/2
term4 = @. (term1 + term3 + relu.(term1 - term3) + relu.(term3 - term1))/2
# plot(x, term4)

#=
x ->
[2Ax - Aπ,  4A/π x - 6A] = [z1, z2]->
Id ->
[z1, -z1, z2, -z2] ->
relu -> 0.5*[-1 -1 0 0;
             0  0  1 1] [z1, -z1, z2, -z2] + [Aπ/2, -A] = [t1, t2] ->
Id ->

[t1, t2, -t2] ->

relu(t1 and t2 get bypass) --> [t1, t2, r(t2), r(-t2)] ->

0.5*[1, 0, 0, 0
     0, 1, -1, -1] * [t1, t2, r(t2), r(-t2)]  = [t1, t3] ->

Id -> [t1, t3, t1-t3, t3-t1]

relu ->

0.5*[1 1 1 1] [t1, t3, r(t1-t3), r(t3-t1)] ->

-> OUT
=#

L1 = Dense([2A, 4A/π], [-A*π, -6A], ReLUBypass()) # [z1, z2]

W2 = [ 1 0
      -1 0
       0 1
       0 -1]
L2 = Dense(W2, zeros(4), relu) # [z1, -z1, z2, -z2]

W3 = [-1 -1 0 0
      0 0 1 1
      0 0 1 1
      0 0 -1 -1]
L3 = Dense(0.5*W3, A*[π/2, -1, -1, 1], ReLUBypass(1, 2)) # [t1, t2, r(t2), r(-t2)]

W4 = [2 0 0 0
      0 1 -1 -1
      2 -1 1 1
      -2 1 -1 -1]
L4 = Dense(0.5*W4, zeros(4), ReLUBypass(1, 2))  # [t1, t3, r(t1-t3), r(t3-t1)]

W5 = [1, 1, 1, 1]'
L5 = Dense(0.5*W5, 0, identity) # bypass everything

C = Chain(L1, L2, L3, L4, L5)
C2 = relu_bypass(C)
upperPlot = plot(x, C2.(x))
upperPlot = plot!(x, sin.(x))

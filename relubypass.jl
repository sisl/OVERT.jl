using Flux, LinearAlgebra

"""
    struct ReLUBypass{T}
        protected::T
    end

an activation function that calls `relu` only on some indices. `ReLUBypass()` or `ReLUBypass(nothing)`
default to the behavior of `identity`. To get the behavior of a regular `relu`, use `ReLUBypass([])`.

### Examples
```jldoctest
julia> a = [-1, 5, 3, -10];

julia> R = ReLUBypass()
ReLUBypass(nothing)

julia> R(a)
4-element Array{Int64,1}:
  -1
   5
   3
 -10

julia> R = ReLUBypass([1]) # equivalent to ReLUBypass(1)
ReLUBypass([1])

julia> R(a)
4-element Array{Int64,1}:
 -1
  5
  3
  0

julia> R = ReLUBypass(2:4)
ReLUBypass(2:4)

julia> R(a)
4-element Array{Int64,1}:
   0
   5
   3
 -10
```
"""
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

Base.show(io::IO, RB::ReLUBypass) = print(io, "ReLUBypass($(repr(RB.protected)))")






#### FLUX COMPATIBILITY SHOULD LIVE ON ITS OWN.

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
    B = [I; -I[protected, :]]
    L1_new = Dense(B*W, B*b, relu)
    L2_new = Dense(L2.W*B', L2.b, L2.σ)
    return L1_new, L2_new
end

function relu_bypass(C::Chain)
    C_bypassed = collect(C)
    for i in 1:length(C)-1
        C_bypassed[i:i+1] .= relu_bypass(C_bypassed[i], C_bypassed[i+1])
    end
    Chain(C_bypassed...)
end

Base.:-(x::AbstractArray, y::Number) = x .- y
Base.:+(x::AbstractArray, y::Number) = x .+ y
Base.:-(x::Number, y::AbstractArray) = x .- y
Base.:+(x::Number, y::AbstractArray) = x .+ y

(D::Dense{<:ReLUBypass, A, B})(x::AbstractArray) where {A, B} = D.σ(D.W*x + D.b)
(D::Dense{<:ReLUBypass, A, <:FluxArr})(x::AbstractArray) where A = D.σ(D.W*x + D.b)
(D::Dense{<:ReLUBypass, <:FluxArr, <:FluxArr})(x::AbstractArray) where A = D.σ(D.W*x + D.b)
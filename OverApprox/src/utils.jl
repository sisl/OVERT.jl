# convenience functions for julia

using LinearAlgebra

# conveniance
eye(dim) = Matrix{Float64}(I, dim, dim)

# Flux section:
using Flux
weights(D::Dense) = Tracker.data(D.W)
bias(D::Dense) = Tracker.data(D.b)
activation(D::Dense) = D.σ

weights(R::Flux.RNNCell) = Tracker.data(R.Wi)
bias(R::Flux.RNNCell) = Tracker.data(R.b)
latent_weights(R::Flux.RNNCell) = Tracker.data(R.Wh)
latent_bias(R::Flux.RNNCell) = Tracker.data(R.h)
latent_size(R::Flux.RNNCell) = size(latent_weights(R), 1)
activation(R::Flux.RNNCell) = R.σ

weights(R::Flux.Recur) = weights(R.cell)
bias(R::Flux.Recur) = bias(R.cell)
latent_weights(R::Flux.Recur) = latent_weights(R.cell)
latent_bias(R::Flux.Recur) = latent_bias(R.cell)
latent_size(R::Flux.Recur) = latent_size(R.cell)
activation(R::Flux.Recur) = activation(R.cell)

layer_size(L, i = nothing) = i == nothing ? size(weights(L)) : size(weights(L), i)

#### Stack the networks. Essentially a layer by layer vcat (with some added intricacies)
"""
    stack_layers(layers...; [σ = nothing], [block_diagonal_weights = false])
Vertically stack compatible FFNN layers.
 - `σ` - set the activation function for the output layer. If all layers do not
  have the same activation, then this field is required.
 - `block_diagonal_weights` - whether to stack weights *vertically* or into a
  *block diagonal*, i.e. `[W₁; W₂;...]` or `[W₁ 0 ...; 0, W₂, 0...; ...]`.
"""
function stack_layers(layers::LayerType...; σ = nothing, block_diagonal_weights = false) where LayerType
    if σ == nothing
        σ = activation(layers[1])
        @assert all(l-> σ == activation(l), layers) "All the layers to be concatenated
            must have the same activation function unless the `σ` keyword is set"
    end
    Ws = (weights(l) for l in layers)
    biases  = (bias(l) for l in layers)
    W = block_diagonal_weights ? block_diagonal(Ws...) : vcat(Ws...)
    b = vcat(biases...)
    LayerType(W, b, σ)
end

"""
    stack_networks(chains::NetworkType...)
Vertically stack compatible FFNNs. Each `chain` must be an iterable/indexable object
consisting of FFNN "layers" that implement `weights`, `bias`, and `activation`.
"""
function stack_networks(chains::NetworkType...) where NetworkType
    c1 = first(chains)
    @assert all(length(c1) .== length.(chains)) "All of the Chains must be the same length to be `stack`ed"
    C = collect(c1)
    for i in 1:length(c1)
        C[i] = stack_layers((chain[i] for chain in chains)..., block_diagonal_weights = (i != 1))
    end
    NT = Base.typename(NetworkType).wrapper
    NT(C...)
end



# General Utils:
"""
    block_diagonal(arrays...)
Construct a fully populated block diagonal matrix from a sequence of Abstract Vector or Matrices.

# Example
    julia> LI = collect(LinearIndices((3,3)))
    3×3 Array{Int64,2}:
     1  4  7
     2  5  8
     3  6  9

    julia> block_diagonal(LI, LI)
    6×6 Array{Int64,2}:
     1  4  7  0  0  0
     2  5  8  0  0  0
     3  6  9  0  0  0
     0  0  0  1  4  7
     0  0  0  2  5  8
     0  0  0  3  6  9

    julia> block_diagonal(vec(LI)', LI)
    4×12 Array{Int64,2}:
     1  2  3  4  5  6  7  8  9  0  0  0
     0  0  0  0  0  0  0  0  0  1  4  7
     0  0  0  0  0  0  0  0  0  2  5  8
     0  0  0  0  0  0  0  0  0  3  6  9
"""
function block_diagonal(arrays::AbstractMatrix...)
    T = promote_type(eltype.(arrays)...)
    A = Matrix{T}(undef, sum(size.(arrays, 1)), sum(size.(arrays, 2)))
    fill!(A, zero(T <: Number ? T : Int))
    n = m = 0
    for B in arrays
        n1, m1 = size(B) .+ (n, m)
        A[(n+1):n1, (m+1):m1] .= B
        n, m = n1, m1
    end
    A
end
block_diagonal(arrays...) = block_diagonal(_as_matrix.(arrays)...)
_as_matrix(A::AbstractVector) = reshape(A, :, 1)
_as_matrix(M::AbstractMatrix) = M
_as_matrix(x) = [x]


"""
    add_bypass_variables(layer, n)
    add_bypass_variables(chain, n)

Adds `n` additional input variables to the layer or network and activation-bypasses
them so that they are also in the output unchanged.
Note, the new variables are stacked at the *bottom* of the input and output.
"""
function add_bypass_variables(L::Dense, n_vars)
    W, b, σ = weights(L), bias(L), activation(L)
    n = length(b)
    b = [b; zeros(n_vars)]
    W = block_diagonal(W, eye(n_vars))

    if σ == relu
        R = ReLUBypass(collect(n .+ (1:n_vars)))
    elseif σ == identity
        R = identity
    elseif σ isa ReLUBypass
        R = ReLUBypass([σ.protected; collect(n .+ (1:n_vars))])
    else
        error("unsupported activation $σ")
    end
    Dense(W, b, R)
end

function add_bypass_variables(L::Flux.RNNCell, n_vars)
    Wi, b, σ = weights(L), bias(L), activation(L)
    Wh, h = latent_weights(L), latent_bias(L)

    n = length(b)
    b = [b; zeros(n_vars)]
    h = [h; zeros(n_vars)]
    Wi = block_diagonal(Wi, eye(n_vars))
    Wh = block_diagonal(Wh, zeros(n_vars, n_vars))

    if σ == relu
        R = ReLUBypass(collect(n .+ (1:n_vars)))
    elseif σ == identity
        R = identity
    elseif σ isa ReLUBypass
        R = ReLUBypass([σ.protected; collect(n .+ (1:n_vars))])
    else
        error("unsupported activation $σ")
    end
    cell = Flux.RNNCell(R, promote(Wi, Wh)..., b, h)
    Flux.Recur(cell)
end
add_bypass_variables(L::Flux.Recur, n_vars) = Flux.Recur(add_bypass_variables(L.cell, n_vars))


function add_bypass_variables(C::Chain, n_vars)
    C_new = []
    for L in C
        push!(C_new, add_bypass_variables(L, n_vars))
    end
    Chain(C_new...)
end


function stack_activations(L1, L2)
    n1, n2 = layer_size(L1, 1), layer_size(L2, 1)
    act1, act2 = L1.σ, L2.σ

    act1 == relu == act2                        && return relu
    act1 == identity == act2                    && return identity

    act1 == identity && act2 == relu            && return ReLUBypass(1:n1)
    act1 == relu && act2 == identity            && return ReLUBypass(n1 .+ (1:n2))

    act1 == relu && act2 isa ReLUBypass         && return ReLUBypass(n1 .+ act2.protected)
    act1 isa ReLUBypass && act2 isa ReLUBypass  && return ReLUBypass([act1.protected; n1 .+ act2.protected])
    act1 isa ReLUBypass && act2 == relu         && return deepcopy(act1)

    error("unsupported pair of activations. $act1 and $act2")
end

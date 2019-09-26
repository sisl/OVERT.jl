# conveniance functions for julia

# TYPE PIRACY SECTION DANGER ZONE TODO MUST DELETE
Base.:-(x::AbstractArray, y::Number) = x .- y
Base.:+(x::AbstractArray, y::Number) = x .+ y
Base.:-(x::Number, y::AbstractArray) = x .- y
Base.:+(x::Number, y::AbstractArray) = x .+ y


using LinearAlgebra

# Maybe this is also type piracy who knows
(I::UniformScaling)(T::DataType, n, m) where R = Matrix{T}(I, n, m)
(I::UniformScaling{T})(n, m) where T = I(T, n, m)
(I::UniformScaling{T})(n) where T = I(T, n, n)
(I::UniformScaling{T})(A::AbstractMatrix) where T = I(size(A)...)

# function eye(dim)
# 	return Matrix{Float64}(I, dim, dim)
# end

# Flux section:
using Flux
weights(D::Dense) = Tracker.data(D.W)
bias(D::Dense) = Tracker.data(D.b)

weights(R::Flux.RNNCell) = Tracker.data(R.Wi)
bias(R::Flux.RNNCell) = Tracker.data(R.b)
latent_weights(R::Flux.RNNCell) = Tracker.data(R.Wh)
latent_bias(R::Flux.RNNCell) = Tracker.data(R.h)
latent_size(R::Flux.RNNCell) = size(latent_weights(R), 1)

weights(R::Flux.Recur) = weights(R.cell)
bias(R::Flux.Recur) = bias(R.cell)
latent_weights(R::Flux.Recur) = latent_weights(R.cell)
latent_bias(R::Flux.Recur) = latent_bias(R.cell)
latent_size(R::Flux.Recur) = latent_size(R.cell)

layer_size(L, i = nothing) = i == nothing ? size(weights(L)) : size(weights(L), i)

# Type piracy for flux:
(D::Dense)(x::Number) = D.σ.(D.W*x + D.b) # NOTE: THIS ONE IS TYPE PIRACY


#### Stack the networks. Essentially a layer by layer vcat (with some added intricacies)
"""
    vcat(layers::Dense...; [σ = nothing], [block_diagonal_weights = false])
Vertically stack compatible `Flux.Dense` layers.
 - `σ` - set the activation function for the output layer. If all layers do not
  have the same activation, then this field is required.
 - `block_diagonal_weights` - whether to stack weights *vertically* or into a
  *block diagonal*, i.e. `[W₁; W₂;...]` or `[W₁ 0 ...; 0, W₂, 0...; ...]`.
"""
function Base.vcat(layers::Dense...; σ = nothing, block_diagonal_weights = false)
    if σ == nothing
        σ = layers[1].σ
        @assert all(l-> σ == l.σ, layers) "All the layers to be concatenated
            must have the same activation function unless the `σ` keyword is set"
    end
    weights = (l.W for l in layers)
    biases  = (l.b for l in layers)
    W = block_diagonal_weights ? block_diagonal(weights...) : vcat(weights...)
    b = vcat(biases...)
    Dense(W, b, σ)
end

"""
    vcat(chains::Chain...)
Vertically stack compatible `Flux.Chain`s
"""
function Base.vcat(chains::Chain...)
    c1 = first(chains)
    @assert all(length(c1) .== length.(chains)) "All of the Chains must be the same length to be `vcat`ed"
    C = collect(c1)
    for i in 1:length(c1)
        C[i] = vcat((chain[i] for chain in chains)..., block_diagonal_weights = (i != 1))
    end
    Chain(C...)
end




# General Utils:
"""
    block_diagonal(arrays...)
Construct a fully populated block diagonal matrix from a sequence of arrays.

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

     julia> block_diagonal(vec(LI), LI)
    12×4 Array{Int64,2}:
     1  0  0  0
     2  0  0  0
     3  0  0  0
     4  0  0  0
     5  0  0  0
     6  0  0  0
     7  0  0  0
     8  0  0  0
     9  0  0  0
     0  1  4  7
     0  2  5  8
     0  3  6  9
"""
function block_diagonal(arrays::AbstractMatrix...)
    T = promote_type(eltype.(arrays)...)
    A = zeros(T, sum(size.(arrays, 1)), sum(size.(arrays, 2)))
    n = m = 0
    for B in arrays
        n1, m1 = size(B) .+ (n, m)
        A[(n+1):n1, (m+1):m1] .= B
        n, m = n1, m1
    end
    A
end
block_diagonal(arrays...) = block_diagonal(to_matrix.(arrays)...)
to_matrix(A::AbstractVector) = reshape(A, :, 1)
to_matrix(M::AbstractMatrix) = M


"""
    add_bypass_variables(layer, n)
    add_bypass_variables(chain, n)

Adds `n` additional input variables to the layer or network and activation-bypasses them so that they are also in the output unchanged.
Note, the new variables are stacked at the *bottom* of the input and output.
"""
function add_bypass_variables(L::Dense, n_vars)
    W, b, σ = weights(L), bias(L), L.σ
    n = length(b)
    b = [b; zeros(n_vars)]
    W = block_diagonal(W, I(n_vars))
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

function add_bypass_variables(L::Flux.Recur, n_vars)
    Wi, b, σ = weights(L), bias(L), L.cell.σ
    Wh, h = latent_weights(L), latent_bias(L)

    n = length(b)
    b = [b; zeros(n_vars)]
    h = [h; zeros(n_vars)]
    Wi = block_diagonal(Wi, I(n_vars))
    Wh = block_diagonal(Wh, zeros(n_vars, n_vars))

    if σ == relu
        R = ReLUBypass(collect(n .+ (1:n_vars)))
    elseif σ == identity
        R = identity
    end
    cell = Flux.RNNCell(R, promote(Wi, Wh)..., b, h)
    Flux.Recur(cell)
end


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

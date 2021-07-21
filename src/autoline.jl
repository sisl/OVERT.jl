# using Flux, SymPy, Plots
using MacroTools, SymEngine
import MacroTools.postwalk

#include("utils.jl")
#include("relubypass.jl")

"""
    abs_to_relu!(ex::Expr)
    abs_to_relu(ex::Expr)
function to convert an `abs(x)` expression to `relu(x) + relu(-x)`. Mutating and nonmutating versions.
"""
function abs_to_relu(ex::Expr)
    check_for_abs(ex)
    x = ex.args[2] # the thing getting abs'ed
    return :(relu($x) + relu(-$x))
end
function abs_to_relu!(ex::Expr)
    check_for_abs(ex)
    x = ex.args[2] # the thing getting abs'ed
    ex.args = [:+, :(relu($x)), :(relu(-$x))]
end

function check_for_abs(ex::Expr)
    ex.args[1] == :abs || throw(ArgumentError("Not an absolute value expression"))
    @assert length(ex.args) == 2 "Malformed expression. `abs` can take only one argument. Got $ex"
    return nothing
end

function check_for_maxmin(ex::Expr)
    ex.args[1] ∈ (:max, :min) || throw(ArgumentError("function is neither max nor min. Got $(ex.args[1])"))
    @assert length(ex.args) == 3 "max/min_to_abs can't handle more than two inputs at the moment."
    return nothing
end

function maxmin_to_abs(ex::Expr)
    check_for_maxmin(ex)
    a, b = ex.args[2:3]
    if ex.args[1] == :max
        return :(0.5*($a + $b + abs($a-$b)))
    elseif ex.args[1] == :min
        return :(0.5*($a + $b - abs($a-$b)))
    end
end

function max0torelu(ex::Expr)
    @assert(length(ex.args)==3)
    a, b = ex.args[2:3]
    if (a == 0)
        return :(relu($b))
    elseif (b == 0)
        return :(relu($a))
    end
end

to_relu_expression(not_ex) = not_ex
function to_relu_expression(ex::Expr)
    ex.head != :call && return ex
    if ex.args[1] ∈ (:max, :min)
        if (ex.args[1] == :max) & ((ex.args[2] == 0) | (ex.args[3] == 0))
            ex = max0torelu(ex)
        else
            ex = maxmin_to_abs(ex)
        end
    elseif ex.args[1] == :abs
        ex = abs_to_relu(ex) end
    for i in 1:length(ex.args)
        ex.args[i] = to_relu_expression(ex.args[i])
    end
    return ex
end

##########################################################################################

"""
Construct a symbolic expression for a line between the points (x₀, 1.0) on the left and (x₁, 0) on the right.
`pos_unit` has positive slope while `neg_unit` has negative slope. Note that due to the left-to-right assumption

    neg_unit(x₀, x₁) == pos_unit(x₁, x₀)
"""
neg_unit(x0, x1) = :($(1/(x0-x1)) * (x - $x1))
"""
Construct a symbolic expression for a line between the points (x₀, 0.0) on the left and (x₁, 1.0) on the right.
`pos_unit` has positive slope while `neg_unit` has negative slope. Note that due to the left-to-right assumption

    neg_unit(x₀, x₁) == pos_unit(x₁, x₀)
"""
pos_unit(x0, x1) = :($(1/(x1-x0)) * (x - $x0))

"""
    closed_form_piecewise_linear(pts)::Expr

Constructs a closed-form piecewise linear expression from an ordered (left to right) sequence of points.
The method is inspired by the paper by Lum and Chua (cite) that considers a piecewise linear function of the form:
`f(x) = Σᵢ(yᵢ⋅gᵢ(xᵢ))` where `gᵢ(xⱼ) = δᵢⱼ`
Here in the 1D case, `gᵢ = max(0, yᵢ*min(L1, L2))`, where `L1` and `L2` are the lines "ramping up" towards xᵢ
and "ramping down" away from xᵢ. The function returns an `Expr` based on a variable `x`.
This can be turned into a callable function `f` by running something like `eval(:(f(x) = \$expression_of_x))`.

# Example
    julia> pts = [(0,0), (1,1), (2, 0)]
    3-element Array{Tuple{Int64,Int64},1}:
     (0, 0)
     (1, 1)
     (2, 0)

    julia> closed_form_piecewise_linear(pts)
    :(max(0, 0 * (-1.0 * (x - 1))) + max(0, 1 * min(1.0 * (x - 0), -1.0 * (x - 2))) + max(0, 0 * (1.0 * (x - 1))))
"""
function closed_form_piecewise_linear(pts)
    n = length(pts)
    x, y = first.(pts), last.(pts) # split the x and y coordinates
    G = []
    for i in 2:n-1
        x0, x1, x2 = x[i-1:i+1] # consider the "triangulation" of points x0,x1,x2
        L1 = pos_unit(x0, x1) # x0-x1 is an increasing linear unit
        L2 = neg_unit(x1, x2) # x1-x2 is a decreasing linear unit
        gᵢ = :($(y[i]) * max(0, min($L1, $L2)))
        push!(G, gᵢ)
    end
    # first and last points are special cases that ignore the min
    g₀ = :($(y[1]) * max(0, $(neg_unit(x[1], x[2]))))
    gᵣ = :($(y[end]) * max(0, $(pos_unit(x[end-1], x[end]))))
    # Order doesn't matter now but for our debugging purposes earlier we enforce sequential ordering.
    pushfirst!(G, g₀)
    push!(G, gᵣ)
    return :(+$(G...))
end

##########################################################################################

function make_expr_dict(ex)
    D = Dict()
    ex = postwalk(ex) do e
        if is_relu_expr(e)
            return get!(D, e, Symbol("z$(length(D)+1)"))
        end
        return e
    end
    D[ex] = Symbol("z$(length(D)+1)")
    return D
end

# is_negative_expr(ex) = ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
is_relu_expr(ex) = ex isa Expr && ex.head == :call && ex.args[1] == :relu

simplify(ex::Expr) = postwalk(e -> _simplify(e), ex)
_simplify(s) = s
_simplify(e::Expr) = Meta.parse(string(expand(Basic(e))))

# thinking/scripting:
Base.occursin(x, y) = x == y
function Base.occursin(needle::Union{Symbol, Expr}, haystack::Expr)
    needle == haystack && return true
    for arg in haystack.args
        occursin(needle, arg) && return true
    end
    return false
end

function layer_sort(D::AbstractDict)
    # construct adjacency matrix. Not checking for cycles since
    # a neural network can't have cycles anyway, but maybe at some point.
    B = BitArray(occursin(v, k) for v in values(D), k in keys(D))
    layer_sort(B)
end

function layer_sort(B::AbstractMatrix)
    V = findall(vec(sum(B, dims = 1)) .== 0)
    isempty(V) && error("no parentless nodes")
    S = Set(V)
    A = setdiff(Set(axes(B, 1)), S)
    layers = [V]
    while !isempty(A)
        L = []
        for i in A
            parents = findall(B[:, i])
            if all(p -> p ∈ S, parents)
                pop!(A, i)
                push!(L, i)
            end
        end
        union!(S, L)
        push!(layers, L)
    end
    layers
end

# Type piracy again:
Base.Expr(B::Basic) = Meta.parse(string(B))


function get_symbols(ex::Union{Expr, Symbol})
    syms = Symbol[]
    ops = (:*, :+, :-, :relu)
    postwalk(e -> e isa Symbol && e ∉ ops ? push!(syms, e) : nothing, ex)
    unique(syms)
end

"""
    W, b, R, in = layerize(out::Vector{Union{Expr, Symbol}})

Constructs a neural network layer that produces `out` as its output.
The input to the layer is inferred based on the free symbols in `out`.

# Returns
    - `W` - weights matrix.
    - `b` - bias vector.
    - `R` - activation function. Generally a ReLUBypass.
    - `in`- the inferred vector of inputs.

# Examples
    julia> out = [ :(relu(10x1 + x2 - 11)) ]
    1-element Array{Expr,1}:
     :(relu((10x1 + x2) - 11))

    julia> layerize(out)
    ([1.0 10.0], [-11.0], ReLUBypass(Int64[]), Union{Expr, Symbol}[:x2, :x1])
"""
function layerize(out)
    syms = free_symbols(Basic.(out))
    n, m = length(out), length(syms)
    W = zeros(n, m)
    b = zeros(n)
    bypass_indices = Int[]
    for (i, o) in enumerate(out)
        if is_relu_expr(o)
            o = o.args[2]
        else
            push!(bypass_indices, i)
        end
        symbolic = Basic(o)
        W[i, :] = coeff.(symbolic, syms)
        b[i] = subs(symbolic, (syms .=> 0)...)
    end
    if length(bypass_indices) == length(syms)
        R = identity
    elseif isempty(bypass_indices)
        R = relu
    else
        R = ReLUBypass(bypass_indices)
    end

    W, b, ReLUBypass(bypass_indices), Union{Expr, Symbol}[Expr.(syms)...]
end

## in general, must check that the dict is reversible
reverse_dict(D::Dict) = Dict(v=>k for (k,v) in D)

# turn an expression into a dense network
function to_network(D::Dict)
    layer_indices = layer_sort(D)
    exprs = collect(keys(D))
    syms = collect(values(D))
    layers = []
    out = Union{Expr, Symbol}[Symbol("z$(length(D))")]
    for (i, layer) in enumerate(reverse(layer_indices))
        replace_ind = map(z -> findfirst(z .== out), syms[layer])
        out[replace_ind] = exprs[layer]
        W, b, R, out = layerize(out)
        if i == 1
            R = identity
        end
        push!(layers, Dense(W, b, R))
    end
    reverse(layers)
end

# TODO RENAME
function to_network(pts::Vector)
    ex = to_relu_expression(closed_form_piecewise_linear(pts))
    D = make_expr_dict(simplify(ex))
    Chain(to_network(D)...) # Chain is a type from Flux.jl
end



# # SCRIPT
# pts = [(0,0), (1,1), (2, 0), (3, 1), (6, 11.258)]
# # ex = closed_form_piecewise_linear(pts)
# include("newpiecewise.jl")
# ex = amirs_piecewise_linear(pts)
# ex = simplify(to_relu_expression(ex))
# D = make_expr_dict(ex)
# ks = collect(keys(D))
# vs = collect(values(D))
# # B = BitArray(occursin(v, k) for v in vs, k in ks)
# # starting_nodes = findall(vec(sum(B, dims = 1)) .== 0)
# # layers = layer_sort(B)
# C = Chain(to_network(D)...)
# CC = relu_bypass(C)

# # h(x) = (l = C(x); length(l) == 1 && return l[1])
# # g(x) = (l = CC(x); length(l) == 1 && return l[1])
# # eval(:(f(x) = $ex))
# # plot([f, g, h], -2pi:0.01:2pi)

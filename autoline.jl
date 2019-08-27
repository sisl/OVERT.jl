# using Flux, SymPy, Plots
using MacroTools, SymEngine
import MacroTools.postwalk

include("utils.jl")
include("relubypass.jl")

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

to_relu_expression(not_ex) = not_ex
function to_relu_expression(ex::Expr)
    ex.head != :call && return ex
    if ex.args[1] ∈ (:max, :min)
        ex = maxmin_to_abs(ex)
    elseif ex.args[1] == :abs
        ex = abs_to_relu(ex)
    end
    for i in 1:length(ex.args)
        ex.args[i] = to_relu_expression(ex.args[i])
    end
    return ex
end

##########################################################################################
##### NOTE this section superceded by amir's piecewise method in newpiecewise.jl

function slope_int(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1)/(x2 - x1)
    b = -m*x1 + y1
    return m, b
end
function lip_const(pts)
    m = [slope_int(pts[i], pts[i+1])[1] for i in 1:length(pts)-1]
    ceil(maximum(abs.(m)))
end

lip_line(ℓ, pt) = :($(ℓ)*x + $(pt[2] - ℓ*pt[1]))

function closed_form_piecewise_linear(pts)
    m, b = slope_int(pts[1], pts[2])
    equation = :($m*x + $b)
    ℓ = lip_const(pts)
    for i in 2:length(pts)-1
        m_new, b_new = slope_int(pts[i], pts[i+1])
        if m_new == m
            continue
        end
        line = :($m_new*x + $b_new)
        if m_new > m
            equation = Expr(:call, :max, equation, :(min($(lip_line(ℓ, pts[i])), $line)))
        else
            equation = Expr(:call, :min, equation, :(max($(lip_line(-ℓ, pts[i])), $line)))
        end
        m = m_new
    end
    return equation
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

# based on https://gist.github.com/davidagold/b94552828f4cf33dd3c8
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
    W, b, ReLUBypass(bypass_indices), Union{Expr, Symbol}[Expr.(syms)...]
end

## in general, must check that the dict is reversible
reverse_dict(D::Dict) = Dict(v=>k for (k,v) in D)

function to_network(D::Dict)
    layer_indices = layer_sort(D)
    exprs = collect(keys(D))
    syms = collect(values(D))
    layers = []
    out = Union{Expr, Symbol}[Symbol("z$(length(D))")]
    for layer in reverse(layer_indices)
        replace_ind = map(z -> findfirst(z .== out), syms[layer])
        out[replace_ind] = exprs[layer]
        W, b, R, out = layerize(out)
        push!(layers, Dense(W, b, R))
    end
    reverse(layers)
end


# SCRIPT
pts = [(0,0), (1,1), (2, 0), (3, 1), (6, 11.258)]
ex = closed_form_piecewise_linear(pts)
ex = simplify(to_relu_expression(ex))
D = make_expr_dict(ex)
ks = collect(keys(D))
vs = collect(values(D))
# B = BitArray(occursin(v, k) for v in vs, k in ks)
# starting_nodes = findall(vec(sum(B, dims = 1)) .== 0)
# layers = layer_sort(B)
C = Chain(to_network(D)...)
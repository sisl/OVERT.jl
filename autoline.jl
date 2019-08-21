# using Flux, SymPy, Plots

# abs_val(x) = relu.(x) + relu.(-x)
# max_eval(a, b) = 0.5*(a + b + abs_val(a-b))
# min_eval(a, b) = 0.5*(a + b - abs_val(a-b))

# struct Point
#     x::Float64
#     y::Float64
# end

# function slope_int(pt1, pt2)
#     slope = (pt2.y - pt1.y)/(pt2.x - pt1.x)
#     intercept = -slope*pt1.x + pt1.y
#     return slope, intercept
# end

# # pts should be a vector of Point types
# function get_line(pts)
#     @vars x
#     slope, int = slope_int(pts[1], pts[2])
#     equation = slope*x + int
#     for i in 2:length(pts)-1
#         slope_new, int_new = slope_int(pts[i], pts[i+1])
#         equation_new = slope_new*x + int_new
#         if slope_new > slope
#             equation = max_eval(equation, equation_new)
#         elseif slope_new < slope
#             equation = min_eval(equation, equation_new)
#         else
#             error("Non-vertex point given! Please remove this point.")
#         end
#         slope = slope_new
#     end

#     return equation
# end

# ## Example ##
# points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 0.0), Point(3.0, 1.0)]
# equation = get_line(points)
# print("\nPoints: ", points, "\n")
# print("\nEquation: ", equation, "\n")
# x = collect(0:0.01:3)
# plot(x, equation.(x))



## expression method:
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

function make_expr_dict(ex)
    D = Dict()
    ex = deepcopy(ex)
    _make_expr_dict(ex, D)
    lastone = Symbol("z$(length(D)+1)")
    D[ex] = lastone
    return D
end

function _make_expr_dict(ex, D = Dict())
    ex isa Expr || return ex
    for (i, arg) in enumerate(ex.args)
        ex.args[i] = _make_expr_dict(arg, D)
    end
    is_negative_expr(ex) && return ex
    !is_relu_expr(ex) && return ex
    if !haskey(D, ex)
        n = length(D)+1
        D[ex] = Symbol("z$n")
    end
    return D[ex]
end


is_negative_expr(ex) = ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
is_relu_expr(ex) = ex.head == :call && ex.args[1] == :relu

# based on https://gist.github.com/davidagold/b94552828f4cf33dd3c8
function simplify(e::Expr)
    # _simplify(e)
    Meta.parse(string(expand(Basic(e))))
end

# _simplify(e) = e
# function _simplify(e::Expr)
#     # apply the following only to expressions that call a function on arguments
#     if e.head == :call
#         op = e.args[1] # in such expressions, `args[1]` is the called function
#         simplified_args = [ _simplify(arg) for arg in e.args[2:end] ]
#         if op == :*
#             0 in e.args[2:end] && return 0 # return 0 if any args are 0
#             simplified_args = simplified_args[simplified_args .!= 1] # remove 1s
#         elseif op ∈ (:+, :-)
#             simplified_args = simplified_args[simplified_args .!= 0] # remove 0s
#         elseif op ∈ (:min, :max)
#             simplified_args = unique(simplified_args) # unique min/max args
#         elseif op == :relu
#             @assert length(simplified_args) == 1 "relu of more than one argument isn't allowed"
#             if simplified_args[1] isa Number
#                 return simplified_args[1] <= 0 ? 0 : simplified_args[1] # relu +/- when known
#             end
#             return :(relu($(simplified_args[1])))
#         end
#         length(simplified_args) == 0 && return 0
#         return Expr(:call, op, simplified_args...)
#     end
# end

# exex = :(8x + 9z1 + 0*x + relu(x + x - 2 + 0) - min(1, 1, -4, -1, x))
# postwalk(exex) do e
#     @capture(e, h_(a__))
#     Meta.parse()
#     return e
# end

# function simplify_addition(ex)
#     # requires SymEngine
#     Meta.parse(string(Basic(ex)))
# end

# thinking/scripting:
Base.occursin(x, y) = x == y
function Base.occursin(needle::Union{Symbol, Expr}, haystack::Expr)
    needle == haystack && return true
    for arg in haystack.args
        occursin(needle, arg) && return true
    end
    return false
end

ex = closed_form_piecewise_linear([(0,0), (1,1), (2, 0), (3, 1), (6, 11.258)])
ex = simplify(to_relu_expression(ex))
D = make_expr_dict(ex)
ks = collect(keys(D))
vs = collect(values(D))
B = BitArray(occursin(v, k) for v in vs, k in ks)
starting_nodes = findall(vec(sum(B, dims = 1)) .== 0)

function layer_sort(B)
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

layers = layer_sort(B)

using MacroTools, SymEngine
import MacroTools.postwalk
# Type piracy again:
Base.Expr(B::Basic) = Meta.parse(string(B))


function get_symbols(ex)
    syms = Symbol[]
    ops = (:*, :+, :-, :relu)
    postwalk(e -> e isa Symbol && e ∉ ops ? push!(syms, e) : nothing, ex)
    unique(syms)
end

# minus_to_plus(ex::Expr) = Meta.parse(str_minus_to_plus(string(ex)))
# function str_minus_to_plus(str::String)
#     i = 1
#     while i < length(str)
#         if str[i] == '-' && str[i+1] == ' '
#             str = str[1:i-1] * "+ -" * str[i+2:end]
#             i += 2
#         end
#         i += 1
#     end
#     str
# end

# function extract_bias_term(ex)
#     ex = deepcopy(ex)
#     @assert ex.head == :call && ex.args[1] ∈ (:+, :-)
#     i = findfirst(i-> i isa Number, ex.args)
#     b = ex.args[i]
#     deleteat!(ex.args, i)
#     return ex, b
# end

# out is the vector or scalar symbol/expr we want
function layerize(out)
    syms = Basic.(union(get_symbols.(out)...))
    n, m = length(out), length(syms)
    W = zeros(n, m)
    b = zeros(n)
    bypass_indices = Int[]
    for (i, o) in enumerate(out)
        if o isa Expr && o.args[1] == :relu
            o = o.args[2]
        else
            push!(bypass_indices, i)
        end
        symbolic = Basic(string(o))
        W[i, :] = coeff.(symbolic, syms)
        b[i] = subs(symbolic, (syms .=> 0)...)
    end
    W, b, ReLUBypass(bypass_indices), Any[Expr.(syms)...]
end

## in general, must check that the dict is reversible
reverse_dict(D::Dict) = Dict(v=>k for (k,v) in D)

function to_network(D::Dict)
    D = reverse_dict(D)
    last_sym = Symbol("z$(length(D))")
    out = [D[last_sym]]
    layers = []
    while !isempty(D)
        W, b, R, out = layerize(out)
        push!(layers, Dense(W, b, R))
        pop!(D, last_sym)
        if isempty(D)
            break
        end
        last_sym = Symbol("z$(length(D))")
        out[findfirst(out .== last_sym)] = D[last_sym]
    end
    reverse(layers)
end


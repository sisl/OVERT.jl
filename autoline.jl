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
    m = 0
    for i in 1:length(pts)-1
        mᵢ, b = slope_int(pts[i], pts[i+1])
        if abs(mᵢ) > m
            m = abs(mᵢ)
        end
    end
    ceil(m)
end

lip_line(ℓ, pt) = lip_line = :($(ℓ)*x + $(pt[2] - ℓ*pt[1]))

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

# make_expr_dict(ex) = _make_expr_dict(deepcopy(ex))
# function _make_expr_dict(ex, D = Dict())
#     if !(ex isa Expr)
#         return
#     end
#     for (i, arg) in enumerate(ex.args)
#         _make_expr_dict(arg, D)
#         if arg isa Symbol || arg isa Number
#             continue
#         end

#         if is_negative_expr(arg)
#             # the key for the positive part is necessarily
#             # already in the dict due to recursion order
#             return
#         elseif haskey(D, arg)
#             simplified_expr = D[arg]
#         else
#             D[:counter] = n = get(D, :counter, 0) + 1
#             D[arg] = Symbol("z$n")
#             simplified_expr = D[arg]
#         end
#         ex.args[i] = simplified_expr
#     end
#     D
# end

function make_expr_dict(ex)
    D = Dict()
    _make_expr_dict(deepcopy(ex), D)
    delete!(D, :counter)
    return D
end

function _make_expr_dict(ex, D = Dict())
    ex isa Expr || return ex
    for (i, arg) in enumerate(ex.args)
        ex.args[i] = _make_expr_dict(arg, D)
    end
    is_negative_expr(ex) && return ex
    if !haskey(D, ex)
        D[:counter] = n = get(D, :counter, 0) + 1
        D[ex] = Symbol("z$n")
    end
    return D[ex]
end


is_negative_expr(ex) = ex.head == :call && ex.args[1] == :- && length(ex.args) == 2


# based on https://gist.github.com/davidagold/b94552828f4cf33dd3c8
function simplify(e::Expr)
    _simplify(e)
end

_simplify(e) = e
function _simplify(e::Expr)
    # apply the following only to expressions that call a function on arguments
    if e.head == :call
        op = e.args[1] # in such expressions, `args[1]` is the called function
        simplified_args = [ _simplify(arg) for arg in e.args[2:end] ]
        if op == :*
            0 in e.args[2:end] && return 0 # return 0 if any args are 0
            simplified_args = simplified_args[simplified_args .!= 1] # remove 1s
        elseif op ∈ (:+, :-)
            simplified_args = simplified_args[simplified_args .!= 0] # remove 0s
        elseif op ∈ (:min, :max)
            simplified_args = unique(simplified_args) # unique min/max args
        elseif op == :relu
            length(simplified_args) > 1 && error("bad relu")
            if simplified_args[1] isa Number
                return simplified_args[1] <= 0 ? 0 : simplified_args[1] # relu +/- when known
            end
            return Expr(:call, op, simplified_args...)
        end
        length(simplified_args) == 0 ? 0 :
        length(simplified_args) == 1 ? simplified_args[1] :
        return Expr(:call, op, simplified_args...)
    end
end

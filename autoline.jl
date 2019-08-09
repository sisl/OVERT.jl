using Flux, SymPy, Plots

abs_val(x) = relu.(x) + relu.(-x)
max_eval(a, b) = 0.5*(a + b + abs_val(a-b))
min_eval(a, b) = 0.5*(a + b - abs_val(a-b))

struct Point
    x::Float64
    y::Float64
end

function slope_int(pt1, pt2)
    slope = (pt2.y - pt1.y)/(pt2.x - pt1.x)
    intercept = -slope*pt1.x + pt1.y
    return slope, intercept
end

# pts should be a vector of Point types
function get_line(pts)
    @vars x
    slope, int = slope_int(pts[1], pts[2])
    equation = slope*x + int
    for i in 2:length(pts)-1
        slope_new, int_new = slope_int(pts[i], pts[i+1])
        equation_new = slope_new*x + int_new
        if slope_new > slope
            equation = max_eval(equation, equation_new)
        elseif slope_new < slope
            equation = min_eval(equation, equation_new)
        else
            error("Non-vertex point given! Please remove this point.")
        end
        slope = slope_new
    end

    return equation
end

## Example ##
points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 0.0), Point(3.0, 1.0)]
equation = get_line(points)
print("\nPoints: ", points, "\n")
print("\nEquation: ", equation, "\n")
x = collect(0:0.01:3)
plot(x, equation.(x))



## expression method:
"""
    abs_to_relu!(ex::Expr)
    abs_to_relu(ex::Expr)
function to convert an `abs(x)` expression to `relu(x) - relu(-x)`. Mutating and nonmutating versions.
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

function closed_form_piecewise_linear(pts)
    m, b = slope_int(pts[1], pts[2])
    equation = :($m*x + $b)
    for i in 2:length(pts)-1
        m_new, b_new = slope_int(pts[i], pts[i+1])
        if m_new == m
            continue
        end
        line = :($m_new*x + $b_new)
        maxmin = m_new > m ? :max : :min
        equation = :($maxmin($equation, $maxmin($(pts[i][2]), $line)))
        m = m_new
    end
    return equation
end

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

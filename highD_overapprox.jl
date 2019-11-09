
using Plots
plotly()
include("/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/OverApprox/src/OverApprox.jl")
using Calculus

# exp(cos(x^2) + (1/2)*sin(x + y^2)) x \in [-pi,pi] y\in [-pi, pi]

# f2: cos(x^2)
# +
# f3: x (don't need to approx)
# +
# f4: y^2
# f5: .5*sin(b)
# f6: exp(c)

function to_pairs(B)
    xp,yp = B
    x = vcat(xp...)
    y = vcat(yp...)
    pairs = collect(zip(x,y))
    return pairs
end

f2 = [x -> cos(x^2),
      x -> -2*x*sin(x^2),
      x -> -2*(sin(x^2) + 2*(x^2)*cos(x^2)) ]

f4 = [y -> y^2,
      y -> 2*y,
      y -> 2]

f5 = [b -> .5*sin(b),
      b -> .5*cos(b),
      b -> -.5*sin(b)]

f6 = [c -> exp(c),
      c -> exp(c),
      c -> exp(c)]

funs = [f2, f4, f5, f6]
overapproxes = []
plot([1],[1])
for i = 1:length(funs)
    tup = funs[i]
    f = tup[1]
    df = tup[2]
    d2f = tup[3]
    if (i == 2) | (i==4)
        convex = true
    else
        convex = nothing
    end
    if (i == 3)
        dmin = -π
        dmax = π + π^2
    elseif (i == 4)
        dmin = -1.5
        dmax = 1.5
    else
        dmin = -π
        dmax = π
    end
    LB, UB = Main.OverApprox.overapprox(f, dmin, dmax, 10; convex=convex, df = df, d2f = d2f, plot=false)
    LBpts = unique(sort(to_pairs(LB), by = x -> x[1]))
    UBpts =  unique(sort(to_pairs(UB), by = x -> x[1]))
    fLB = Main.OverApprox.closed_form_piecewise_linear(LBpts)
    fUB = Main.OverApprox.closed_form_piecewise_linear(UBpts)
    push!(overapproxes, (eval(:(x -> $fLB)), eval(:(x -> $fUB))))
end

plot!(1,1)
# exp(cos(x^2) + (1/2)*sin(x + y^2)) x \in [-pi,pi]
# f2: cos(x^2)
# +
# f3: x (don't need to approx)
# +
# f4: y^2
# f5: .5*sin(b)
# f6: exp(c)
expUB = overapproxes[4][2]
hsinUB = overapproxes[3][2]
ysqUB = overapproxes[2][2]
cos2UB = overapproxes[1][2]
bigfUB(x,y) = .5*expUB(cos2UB(x) +
hsinUB(x + ysqUB(y)))

bigf(x,y) = .5*exp(cos(x.^2) +
(1/2)*sin(x + y.^2))

function plot_stuff(f, fUB)
    pyplot()
    x = y = range(-π, stop=π, length=30)
    plot(x, y, f, st = :surface, camera=(15,40))
    plot!(x, y, fUB, st = :surface, camera=(15,40), alpha=0.5, color = :blue)
end

# plot sections
x = zeros(30)
y = range(-π, stop=π, length=200)
bigfy(y) = bigf(0,y)
bigfUBy(y) = bigfUB(0, y)

scatter(y, bigfy)
scatter!(y, bigfUBy)


# todos:
# had a problem with y^2 ... needed to pass "convex"
# number of points should be normalized accross unit distance not concavity section

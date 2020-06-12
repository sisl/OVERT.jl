include("overest_new.jl")
include("autoline.jl")
using Plots


function to_pairs(B)
    """
    This function converts the output of overest in the form ready
        for closed form generator.
    B is an array of points.
    """
    xp,yp = B
    x = vcat(xp...)
    y = vcat(yp...)
    pairs = collect(zip(x,y))
    return pairs
end

function pnt_to_sym(UB_pnts)
    UB_pnts_unique = unique(sort(UB_pnts, by = x -> x[1]))
    UB_sym = closed_form_piecewise_linear(UB_pnts_unique)
    return UB_sym
end

function sym_to_eval(UB_sym)
    return eval(:(x -> $UB_sym))
end

function old_composition(f,g,a,b,N)

    gU = bound(g, a, b, N)
    gL = bound(g, a, b, N; lowerbound=true)
    gU_pnts = to_pairs(gU)
    gL_pnts = to_pairs(gL)
    gU_eval = sym_to_eval(pnt_to_sym(gU_pnts))
    gL_eval = sym_to_eval(pnt_to_sym(gL_pnts))
    fig = plot!(0,0)
    return gU_eval, gL_eval, fig
end

function new_composition(f,g,a,b,N)
    df_zeros = fzeros(Calculus.derivative(f), a, b)
    dg_zeros = fzeros(Calculus.derivative(g), a, b)
    zeros = vcat(df_zeros, dg_zeros, a, b)
    zeros = sort(unique(zeros))
    Ni = max(1, div(N, length(zeros)-1))

    gU_pnts = []
    gL_pnts = []
    for i=1:length(zeros)-1
        ai = zeros[i]
        bi = zeros[i+1]
        gUi = bound(g, ai, bi, Ni)
        gLi = bound(g, ai, bi, Ni, lowerbound=true)
        gU_pnts = vcat(gU_pnts, to_pairs(gUi))
        gL_pnts = vcat(gL_pnts, to_pairs(gLi))
    end
    fig = plot!(0,0)
    gU_eval = sym_to_eval(pnt_to_sym(gU_pnts))
    gL_eval = sym_to_eval(pnt_to_sym(gL_pnts))
    return gU_eval, gL_eval, fig
end

plot(0,0)
f = x-> x*(x-1/2)*(x+1)-2
g, a, b, N = x -> x^2, -1/2, 1, 1
# choose one of these
gU_eval, gL_eval, fig = old_composition(f,g,a,b,N) # old method that does not work
#gU_eval, gL_eval, fig = new_composition(f,g,a,b,N) # new method that works

dx = (b-a)/500
x  = a:dx:b
gx   = g.(x)
fgx  = f.(gx)
gUx  = gU_eval.(x)
fgUx = f.(gUx)
gLx  = gL_eval.(x)
fgLx = f.(gLx)

plot(x, fgx, line=1.0, linestyle=:solid, color=:black, label="f(g(x))")
#plot!(x, fgUx, line=2.0, linestyle=:dash, color=:red, label="f(gU(x))")
#plot!(x, fgLx, line=2.0, linestyle=:dash, color=:blue, label="f(gL(x))")
plot!(x, min.(fgUx.-0.01, fgLx.-0.01), line=1.0, linestyle=:solid, color=:red,
label="lB")
p2 = plot!(x, max.(fgUx.+0.01, fgLx.+0.01), line=1.0, linestyle=:solid, color=:yellow,
label="uB")

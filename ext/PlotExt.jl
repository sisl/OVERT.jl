module PlotExt

using LaTeXStrings
using Plots
import PGFPlots

global NPLOTS = 0
pgfplots_flag = true

global plottype = "pgf"

function set_plottype(t)
    """
    You can change to "html" or "pgf" (latex).
    """
    global plottype = t
end

# function to plot
function plot_bound(f, a, b, xp, yp; existing_plot=nothing, saveflag=false)
	"""
	This function plots f and its overapproximator
		defined by poins xp and yp
	"""
	x = range(a, stop=b, length=1000)
	y = f.(x)
	if pgfplots_flag
		if isnothing(existing_plot)
			p = PGFPlots.Axis([PGFPlots.Linear(x,  y, style="black, thick", mark="none", legendentry=L"f(x)"),
			PGFPlots.Linear(xp, yp, style="blue, thick", mark=:o, legendentry=L"g(x)")],
			legendPos="east")
			xlabel!(p, L"x")
			#display(p)
			global NPLOTS
			NPLOTS += 1
			# PGFPlots.save("plots/bound_"*string(NPLOTS)*".pdf")
			# PGFPlots.save("plots/bound_"*string(NPLOTS)*".tex") # relative to run dir of top level file...dumb...
			#return p
		else
			# push!(existingAxis.plots, new_plot)
			plot!(existing_plot, x,  y, color="red", linewidth=2, label=L"f(x)");
			plot!(existing_plot, xp, yp, color="blue", marker=:o, linewidth=2, label=L"g(x)", legend=:right);
			xlabel!(existing_plot, L"x")
			display(existing_plot);
			global NPLOTS
			NPLOTS += 1
			# savefig("plots/bound_"*string(NPLOTS)*".pdf")
			#return existing_plot
		end
	else
		if isnothing(existing_plot)
			p = plot(x,  y, color="red", linewidth=2, label=L"f(x)");
			p = plot!(p, xp, yp, color="blue", marker=:o, linewidth=2, label=L"g(x)", legend=:right);
			xlabel!(p, L"x")
			display(p)
			global NPLOTS
			NPLOTS += 1
			#savefig("plots/bound_"*string(NPLOTS)*".pdf") # relative to run dir of top level file...dumb...
			#return p
		else
			plot!(existing_plot, x,  y, color="red", linewidth=2, label=L"f(x)");
			plot!(existing_plot, xp, yp, color="blue", marker=:o, linewidth=2, label=L"g(x)", legend=:right);
			xlabel!(existing_plot, L"x")
			display(existing_plot);
			global NPLOTS
			NPLOTS += 1
			if saveflag
				savefig(existing_plot, "plots/bound_"*string(NPLOTS)*".html")
			end
			#return existing_plot
		end
	end
	
	print("\u1b[1F")
end

function make_plot(fun, f_x_expr, lb, ub, LBpoints, UBpoints)
    #p = Plots.plot(0,0)
    global NPLOTS
    NPLOTS += 1
    if f_x_expr.args[1] ∈ [:/, :^]
        # use whole expression in title 
        fun_string = string(f_x_expr)
    else 
        fun_string = string(fun)
    end
    println("funstring = $(fun_string)")
    if plottype != "pgf"
        plotly()
        p = plot(range(lb, ub, length=100), fun, label="function", color="black")
        plot!(p, [p[1] for p in LBpoints], [p[2] for p in LBpoints],  color="purple", marker=:o, markersize=1, label="lower bound")
        plot!(p, [p[1] for p in UBpoints], [p[2] for p in UBpoints], color="blue", marker=:diamond, markersize=1,  label="upper bound", legend=:right, title=fun_string, xlabel=string(x))
        # display(p)
        savefig(p, "plots/bound_"*string(NPLOTS)*".html")
    else # plottype == pgf
        println("Saving PGF plot")
        x_plot_points = range(lb, ub, length=100)
        f_x_plot_points = fun.(x_plot_points)
        fig = PGFPlots.Axis(style="width=10cm, height=10cm", ylabel="\$f(x)\$", xlabel="x", title="\$"*fun_string*"\$")
        push!(fig, PGFPlots.Plots.Linear(x_plot_points, f_x_plot_points, legendentry=fun_string, style="solid, black, mark=none"))
        push!(fig, PGFPlots.Plots.Linear([p[1] for p in LBpoints], [p[2] for p in LBpoints], legendentry="lower bound", style="solid, purple, mark=none"))
        push!(fig, PGFPlots.Plots.Linear([p[1] for p in UBpoints], [p[2] for p in UBpoints], legendentry="upper bound", style="solid, blue, mark=none"))
        fig.legendStyle =  "at={(1.05,1.0)}, anchor=north west"
        PGFPlots.save("plots/bound_"*string(NPLOTS)*".tex", fig)
        PGFPlots.save("plots/bound_"*string(NPLOTS)*".pdf", fig)
    end
end

end
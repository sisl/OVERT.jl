using Plots

global NPLOTS = 0
pgfplots_flag = true

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
#import Pkg
#Pkg.add("NLsolve")
#Pkg.add("Roots")
#Pkg.add("Calculus")
#Pkg.add("Plots")

using Calculus
using NLsolve
using Roots
using Plots
plotly()


RTOL = 0.01
myapprox(x,y) = abs(x-y)<RTOL


function give_interval(d2f_zeros, a, b)
	"""
	this function divides internval [a,b] into subintervals
		[(a, def_zeros[1]), (d2f_zeros[1], d2f_zeros[2]), ... , (d2f_zeros[end], b) ]
	if the interval length is less that RTOL, it will be ignored.
	"""
	if isempty(d2f_zeros)
		return [(a,b)]
	else
		intervals = []
		if !myapprox(d2f_zeros[1], a)
			d2f_zeros = vcat(a, d2f_zeros)
		end
		if !myapprox(d2f_zeros[end], b)
			d2f_zeros = vcat(d2f_zeros, b)
		end
		for i in 1:length(d2f_zeros)-1
			push!(intervals, (d2f_zeros[i], d2f_zeros[i+1]))
		end
		return intervals
	end
end

function bound(f, a, b, N; lowerbound=false, df=nothing, d2f=nothing,
	d2f_zeros=nothing, convex=nothing, out=nothing)

	"""
	This function upper or lower bounds function f(x).

	f:         the function to be bounded
	[a,b]:     domain of f
	N:         number of point within each sub-interval. Results
			   in N+1 sub-sub-intervals
	           sub-intervals are regions where second derivative does not change sign.

	lowerbound:  true if you want a lowerbound
	df:        derivative of f, if available
	d2f:       second derivative of f, if available
	d2f_zeros: zeros of the second derivative, if available
	convex:    true for convex, false for concave, nothing for mixed.
	out:       nothing for plotting, "points" for returning points xp, yp of the piecewise-linear function
	"""

	"""
	Example:
		bound(cos, 0, π, 3)
		bound(x->exp(x^2), 0, 2, 3, convex=true)
		bound(sin, 0, π, 3, df = cos, lowerbound=true)
		bound(sin, -π/2, π, 3, df = cos, d2f= x->-sin(x), d2f_zero=[0])
		bound(x-> x^3-sin(x), 0, 2, 3, out=points)
	"""

	"""
	TODO: add the option that user pass a list and function computes values
	      of the overapproximator at those values.
		  Can be done easily using linear interpolation.
	"""
	function line_equation(p)
		function line(x)
			df(p)*(x - p) + f(p)
		end
	end

	# function to plot
	function plot_bound(f, a, b, xp, yp)
		x = range(a, stop=b, length=200)
		y = f.(x)
		plot(x,  y, color="red", linewidth=2, label="f(x)")
		plot!(xp, yp, color="blue", linewidth=2, label="bound(f(x))")
	end

    if isnothing(df) # calculate derivative if not given.
        df = Calculus.derivative(f)
    end
    if !isnothing(convex) # if convexity is specified, no sub-intervals necessary.
        intervals = [(a,b)]
		d2f = x-> 2*convex -1 # this just gives a + or - value for d2f.
    else
		if isnothing(d2f) # calculate second derivative, if not given.
			d2f = Calculus.second_derivative(f)
		end
        if isnothing(d2f_zeros) # calculate zeros of second derivative, if not given.
            d2f_zeros = fzeros(d2f, a, b)
        end
        intervals = give_interval(d2f_zeros, a, b)  # find subintervals.
    end

	println("algorithm begins")
    xp = []
    yp = []
    for interval in intervals
		aa, bb = interval
        zGuess = range(aa, stop=bb, length=N+2)
        zGuess = reshape(zGuess, N+2,1)[2:end-1]
        if xor(d2f((aa+bb)/2) >= 0, lowerbound)
        	# set up system of equations that must be satisfied to find tightset possible bound for a convex region
			function bound_jensen!(F, z)
				if N==1
					F[1] = f(bb) - f(aa) - df(z[1])*(bb-aa)
				else
					F[1] = f(z[2]) - f(aa) - df(z[1])*(z[2]-aa)
					for i in 2:N-1
						F[i] = f(z[i+1]) - f(z[i-1]) - df(z[i])*(z[i+1]-z[i-1])
					end
					F[N] = f(bb) - f(z[N-1]) - df(z[N])*(bb-z[N-1])
				end
			end
			# optimize points
            z = nlsolve(bound_jensen!, zGuess)
            xx = vcat(aa, z.zero, bb)
            yy = [f(x) for x in xx]
        else
			function lines_in_bound(z)
				# given tie points z_i, generate equations of lines in bound
				# calcuate tangency points
				# construct equation of each line in the bound
				L = []
				x = (aa + z[1])/2
				push!(L, line_equation(x))
				for i=1:N-1
					x = (z[i] + z[i+1])/2
					push!(L, line_equation(x))
				end
				x = (z[N] + bb)/2
				push!(L, line_equation(x))
				return L
			end
			function bound_tangent!(F,z)
				# z represent the x values of the tie points between the bounds
				# get equations of lines in bound
				Leqs = lines_in_bound(z)
				# construct equations, F, to set to 0
				for i=1:N
					F[i] = Leqs[i](z[i]) - Leqs[i+1](z[i])
				end
				return F
			end

			# optimize points
            z = nlsolve(bound_tangent!, zGuess)
            print(z)
            xx = vcat(aa, z.zero, bb)
            yy = zeros(N+2)
            # calculate y-values for concave region (do not lie on function like for convex)
            lines = lines_in_bound(z.zero)
            yy[1] = lines[1](xx[1])
            for i=1:N+1
                yy[i+1] = lines[i](xx[i+1])
            end
        end
        push!(xp, xx)
        push!(yp, yy)
    end
    if isnothing(out)
		plot_bound(f, a, b, xp, yp)
	elseif out == "points"
		return xp, yp
	end
end

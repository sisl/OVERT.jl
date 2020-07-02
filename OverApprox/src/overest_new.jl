
#import Pkg
#Pkg.add("NLsolve")
#Pkg.add("Roots")
#Pkg.add("Calculus")
#Pkg.add("Plots")

using Calculus
using NLsolve
using Roots
using Plots
using Interpolations
#plotly()


RTOL = 0.01
myapprox(x,y) = abs(x-y)<RTOL  # This is defined to identify small intervals

# function to plot
function plot_bound(f, a, b, xp, yp; existing_plot=nothing)
	"""
	This function plots f and its overapproximator
		defined by poins xp and yp
	"""
	x = range(a, stop=b, length=200)
	y = f.(x)
	if isnothing(existing_plot)
		p = plot(x,  y, color="red", linewidth=2, label="f(x)");
		p = plot!(p, xp, yp, color="blue", marker=:o, linewidth=2, label="overest(f(x))", legend=false);
		display(p)
		#return p
	else
		plot!(existing_plot, x,  y, color="red", linewidth=2, label="f(x)");
		plot!(existing_plot, xp, yp, color="blue", marker=:o, linewidth=2, label="overest(f(x))", legend=false);
		display(existing_plot);
		#return existing_plot
	end
	print("\u1b[1F")
end


function give_interval(d2f_zeros, a, b)
	"""
	this function divides interval [a,b] into subintervals
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
			if !myapprox(d2f_zeros[i], d2f_zeros[i+1])
				push!(intervals, (d2f_zeros[i], d2f_zeros[i+1]))
			end
		end
		return intervals
	end
end

# Two auxillary functions that will be used later.
aux_func_1(f, df, alpha, beta) = (beta*df(beta)-alpha*df(alpha) -(f(beta)-f(alpha)))/(df(beta)-df(alpha))
line_equation(f, df, p, x) = df(p)*(x-p)+f(p)

function bound_jensen(f, df, N, aa, bb)
	"""
	This function generates a function that computes the system of equations
		to be solved for a convex case
	"""
	function obj_jensen!(F, z)
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
	return obj_jensen!
end

function bound_tangent(f, df, N, aa, bb, method)
	"""
	This function generates a function that computes the system of equations
		to be solved for a concave case.

	method can be "continuous" or "optimal", depending on what algorithm is preferred.
	"optimal" method produces the tightest overapproximator, but it might be discontinuous
	at inflection points of f. Method "continuous" takes care of the continuity, but
	is sub-optimal.
	"""

	function obj_tangent_continuous!(F, z)
		if N == 1
			F[1] = aux_func_1(f, df, aa, bb) - z[1]
		else
			for i=1:N
				if i == 1
					xm1 = aa
				else
					xm1 = (z[i]+z[i-1])/2
				end
				if i == N
					xm2 = bb
				else
					xm2 = (z[i]+z[i+1])/2
				end
				F[i] = aux_func_1(f, df, xm1, xm2) - z[i]
			end
		end

	end

	function obj_tangent_optimal!(F, z)
		if N == 1
			F[1] = (aa+bb)/2 - z[1]
		else
			for i in 1:N
				if i == 1
					xm1 = (z[i]+aa)/2
					dz1 = z[i] - aa
				else
					xm1 = (z[i]+z[i-1])/2
					dz1 = z[i] - z[i-1]
				end
				if i == N
					xm2 = (z[i]+bb)/2
					dz2 = z[i] - bb
				else
					xm2 = (z[i]+z[i+1])/2
					dz2 = z[i] - z[i+1]
				end
				F[i] = 1/2*df(xm1)*dz1 + f(xm1) - 1/2*df(xm2)*dz2 - f(xm2)
			end
		end
	end

	if method == "continuous"
		return obj_tangent_continuous!
	elseif method == "optimal"
		return obj_tangent_optimal!
	else
		error(" the method is not defined")
	end
end


"""
Interface to the bound function that accepts symbolic functions, not executable ones, so
that symbolic differentiation can be used.
"""

function bound(f, a, b, N; conc_method="continuous", lowerbound=false, df=nothing,
	d2f=nothing, d2f_zeros=nothing, convex=nothing, plot=false, existing_plot=nothing)

	"""
	This function over(under)-approximate function f(x).

	f:         the function to be bounded
	[a,b]:     domain of f
	N:         N-1 is the number of linear segments within each sub-interval.
	           sub-intervals are regions where second derivative does not change sign.
	conc_method: determines the concave algorithm. can take continuous or optimal.
	lowerbound:  true if you want a lower bound
	df:        derivative of f, if available
	d2f:       second derivative of f, if available
	d2f_zeros: zeros of the second derivative, if available
	convex:    true for convex, false for concave, nothing for mixed.
	plot:      true to plot
	"""

	"""
	Example:
		overest(cos, 0, π, 3)
		overest(x->exp(x^2), 0, 2, 3, convex=true)
		overest(sin, 0, π, 3, lowerbound=true)
		overest(sin, -π/2, π, 3, d2f_zero=[0])
		overest(x-> x^3-sin(x), 0, 2, 3, out=points)
	"""

	if N == -1 # optimally choose N
		return bound_optimal(f, a, b; conc_method=conc_method,
			lowerbound=lowerbound, df=df, d2f=d2f, d2f_zeros=d2f_zeros, convex=convex,
			plot=plot, existing_plot=existing_plot)
	end

	try
		f(a)
		f(b)
	catch
		error("$a or $b is not in the domanin of $f")
	end

	if isnothing(df)
    	df = Calculus.derivative(f)
	end
    if ! isnothing(convex) # if convexity is specified, no sub-intervals necessary.
        intervals = [(a,b)]
    else
		if isnothing(d2f)
			d2f = Calculus.second_derivative(f)
		end
		if isnothing(d2f_zeros) # calculate zeros of second derivative, if not given.
			d2f_zeros = fzeros(d2f, a, b)
        end
        intervals = give_interval(d2f_zeros, a, b)  # find subintervals.
    end

	# println("OVERT applied on $f within ranges [$a, $b]")
    xp = []
    yp = []
    for interval in intervals
		aa, bb = interval

		# initial guess for nonlinear solver. we divide [aa,bb] by N equally spaced points.
        zGuess = range(aa, stop=bb, length=N+2)
        zGuess = reshape(zGuess, N+2,1)[2:end-1]

        if xor(d2f((aa+bb)/2) >= 0, lowerbound)  # upper bound for convex or lower bound for concave
			obj = bound_jensen(f, df, N, aa, bb)
		else
			obj = bound_tangent(f, df, N, aa, bb, conc_method)
		end

		# solving nonlinear system.
		# z = nlsolve(obj, zGuess, autodiff = :forward)

		z = 0
		itr_nlsolve = 0
		try # This may overshoot to outside of domain for log
			z = nlsolve(obj, zGuess, autodiff = :forward)

			# check zeros are within the interval
			@assert z.zero[1] > aa
			@assert z.zero[end] < bb
			for ii=1:length(z.zero)-1
				@assert z.zero[ii] < z.zero[ii+1]
			end
    	catch # if overshoots, change the zGuess until it works. 10 iterations allowed
			zGuess_orig = copy(zGuess)
			# shifting zGuess towards left
			while itr_nlsolve <= 10
				zGuess[1:end-1] = zGuess[2:end]
				zGuess[end] = 0.5*(bb + zGuess[end])
				try
					z = nlsolve(obj, zGuess, autodiff = :forward)

					# check zeros are within the interval, unique, and in ascending order
					@assert z.zero[1] > aa
					@assert z.zero[end] < bb
					for ii=1:length(z.zero)-1
						@assert z.zero[ii] < z.zero[ii+1]
					end
					break
				catch
					itr_nlsolve += 1
				end
				#z = mcpsolve(obj, sol_lb, sol_ub , zGuess_exp, autodiff = :forward)
			end

			# reset zGuess
			itr_nlsolve == 11 ? zGuess = zGuess_orig : nothing
			# shifting zGuess towards right
			while (itr_nlsolve <= 20) && (itr_nlsolve >= 11)
				zGuess[2:end] = zGuess[1:end-1]
				zGuess[1] = 0.5*(aa + zGuess[1])
				try
					z = nlsolve(obj, zGuess, autodiff = :forward)
					# check zeros are within the interval
					@assert z.zero[1] > aa
					@assert z.zero[end] < bb
					for ii=1:length(z.zero)-1
						@assert z.zero[ii] < z.zero[ii+1]
					end
					break
				catch
					itr_nlsolve += 1
				end
				#z = mcpsolve(obj, sol_lb, sol_ub , zGuess_exp, autodiff = :forward)
			end
			if itr_nlsolve == 21
				error("nlsolve could not converge")
			end
		end

        xx = vcat(aa, z.zero, bb)  # add endpoints to the intermediate points
		if xor(d2f((aa+bb)/2) >= 0, lowerbound) # upper bound for convex or lower bound for concave
            yy = [f(x) for x in xx]  # for the convex case, points lie on the function
        else # for the concave case, y-values should be calculated.
			yy = zeros(N+2)
			if conc_method == "continuous"
				yy[1] = f(xx[1])
				yy[N+2] = f(xx[N+2])
				for i in 2:N+1
					if i==N+1
						xm2 = bb
					else
						xm2 = (xx[i]+xx[i+1])/2
					end
					yy[i] = line_equation(f,df, xm2,xx[i])
				end
			elseif conc_method == "optimal"
				for i in 1:N+1
					xm2 = (xx[i]+xx[i+1])/2
					yy[i] = line_equation(f, df, xm2, xx[i])
				end
				yy[N+2] = line_equation(f, df, (xx[N+1]+xx[N+2])/2, xx[N+2])
			end
        end
        push!(xp, xx)
        push!(yp, yy)
    end
    if plot
		plot_bound(f, a, b, xp, yp; existing_plot=existing_plot)
		return xp, yp
	else
		#println("no plotting 4 u")
		return xp, yp
	end
end

function overapprox(f,a,b,N; conc_method="continuous", df=nothing, d2f=nothing,
	d2f_zeros=nothing, convex=nothing, plot=false)

	LB = bound(f, a, b, N; conc_method=conc_method, lowerbound=true, df=df, d2f=d2f,
	d2f_zeros=d2f_zeros, convex=convex, plot=plot, reuse=false)

	UB = bound(f, a, b, N; conc_method=conc_method, lowerbound=false, df=df, d2f=d2f,
	d2f_zeros=d2f_zeros, convex=convex, plot=plot, reuse=true)
	return LB, UB
end

function bound_optimal(f, a, b; rel_error_tol=0.02, Nmax = 20, conc_method="continuous",
	lowerbound=false, df=nothing, d2f=nothing, d2f_zeros=nothing, convex=nothing,
	plot=false, existing_plot=nothing)

	try
		f(a)
		f(b)
	catch
		error("$a or $b is not in the domanin of $f")
	end

	if isnothing(df)
		df = Calculus.derivative(f)
	end
	if ! isnothing(convex) # if convexity is specified, no sub-intervals necessary.
		intervals = [(a,b)]
	else
		if isnothing(d2f)
			d2f = Calculus.second_derivative(f)
		end
		if isnothing(d2f_zeros) # calculate zeros of second derivative, if not given.
			d2f_zeros = fzeros(d2f, a, b)
		end
		intervals = give_interval(d2f_zeros, a, b)  # find subintervals.
	end
	xp = []
	yp = []
	for interval in intervals
		aa, bb = interval
		convex = d2f((aa + bb) / 2) > 0   # specify convexity
		for N = 1:Nmax
			xp_candidate, yp_candidate = bound(f, aa, bb, N; conc_method="continuous", lowerbound=lowerbound, df=df,
			d2f=d2f, d2f_zeros=d2f_zeros, convex=convex, plot=plot)

			# interpolate can sometimes give an error because it thinks the
			# points are outside the domain.
			# see this issue https://github.com/JuliaMath/Interpolations.jl/issues/158
			# hence shorten the range of xtest by epsilon
			ϵ_itp = 1E-8


			itp = interpolate((xp_candidate[1],), yp_candidate[1], Gridded(Linear()))
			xtest = range(aa + ϵ_itp, stop=bb - ϵ_itp, length=100)
			ytest = [itp(xt) for xt in xtest]
			ftest = [f(xt)   for xt in xtest]
			sc = maximum(ftest) - minimum(ftest)
			error = [abs(itp(xt) - f(xt))/sc for xt in xtest]
			if (maximum(error) < rel_error_tol) || (N == Nmax)
				xp = vcat(xp, xp_candidate)
				yp = vcat(yp, yp_candidate)
				break
			end
		end
	end

	xp = unique(xp)
	yp = unique(yp)

	if plot
		plot_bound(f, a, b, xp, yp; existing_plot=existing_plot);
		return xp, yp
	else
		return xp, yp
	end
end

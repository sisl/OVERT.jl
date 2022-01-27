using Calculus
using NLsolve
using Roots
using Interpolations
using LaTeXStrings
include("plot_utils.jl")

RTOL = 1e-5
myapprox(x,y) = abs(x-y)<RTOL  # This is defined to identify small intervals

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

# generate y_i candidates for concave upper bound / convex lower bound 
function get_yi_candidates(xx, f, df, N)
	y_i_m = zeros(N+2)
	y_i_p = zeros(N+2)
	# first get all points "looking forwards" (eval right endpoint of the line)
	############################################################################
	y_i_m[1] = f(xx[1])
	y_i_m[2] = line_equation(f,df, xx[1],xx[2]) # args are: tangent point, eval point
	for i = 3:N+1
		y_i_m[i] = line_equation(f, df, (xx[i-1]+xx[i])/2 , xx[i]) 
	end
	y_i_m[N+2] = line_equation(f, df, xx[N+2], xx[N+2]) # should just be f(xx[N+2])
	# then get all points "looking backwards" (eval left endpoint of the line)
	############################################################################
	y_i_p[1] = line_equation(f, df, xx[1], xx[1])
	for i = 3:N+1
		y_i_p[i-1] = line_equation(f, df, (xx[i-1] + xx[i])/2, xx[i-1])
	end
	y_i_p[N+1] = line_equation(f, df, xx[N+2], xx[N+1])
	y_i_p[N+2] = f(xx[N+2])
	return y_i_m, y_i_p
end

function check_concave_upper_optimality(y_i_m, y_i_p; ζ=1e-6)
	"""
	Check if continuity conditions have been successfully enforced.
	How: check if two y_i generated by two candidate tangent bounds are within 1e-5 of each other.
	"""
	## DEBUG: 
	# if maximum(abs.(y_i_m - y_i_p)) > ζ
	# 	@warn "Max abs continuity violation of $(maximum(abs.(y_i_m - y_i_p))) is greater than continuity violation threshold of $ζ ,"
	# end
	return all(abs.(y_i_m - y_i_p) .<= ζ)
end

function get_safe_yi(y_i_m, y_i_p, xx, f, this_interval_convex)
	yy = zeros(length(xx))
	yy[1] = f(xx[1])
	yy[end] = f(xx[end])
	if this_interval_convex # means we are computing lower bound for convex function 
		yy[2:end-1] = min.(y_i_m[2:end-1], y_i_p[2:end-1])
	else # means we are computing upper bound for concave function
		yy[2:end-1] = max.(y_i_m[2:end-1], y_i_p[2:end-1])
	end
	return yy
end

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
	d2f=nothing, d2f_zeros=nothing, convex=nothing, plot=false)
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
		bound(cos, 0, π, 3)
		bound(x->exp(x^2), 0, 2, 3, convex=true)
		bound(sin, 0, π, 3, lowerbound=true)
		bound(sin, -π/2, π, 3, d2f_zeros=[0])
		bound(x-> x^3-sin(x), 0, 2, 3)
	"""
	@debug "Number of points in bound, N= $N"
	if N == -1 # optimally choose N
		return bound_optimal(f, a, b; conc_method=conc_method,
			lowerbound=lowerbound, df=df, d2f=d2f, d2f_zeros=d2f_zeros, convex=convex,
			plot=plot)
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
	if isnothing(d2f)
		d2f = Calculus.second_derivative(f)
	end
    if ! isnothing(convex) # if convexity is specified, no sub-intervals necessary.
		intervals = [(a,b)]
		@debug "convexity specified, no sub intervals necessary." convex
    else #if concavity is not uniform, calculate intervals of uniform concavity
		if isnothing(d2f_zeros) # calculate zeros of second derivative, if not given.
			println("WARNING: d2f_zeros have not been specified for function $f. Convex and concave regions will be identified using a numerical procedure. Soundness not guaranteed. ")
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

		if isnothing(convex)
			this_interval_convex = d2f((aa+bb)/2) >= 0
		else
			@assert convex == (d2f((aa+bb)/2) >= 0)
			this_interval_convex = convex
		end
        if xor(this_interval_convex, lowerbound)  # upper bound for convex or lower bound for concave
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
		if xor(this_interval_convex, lowerbound) # upper bound for convex or lower bound for concave
            yy = [f(x) for x in xx]  # for the convex case, points lie on the function
        else # for the concave case, y-values should be calculated.
			yy = zeros(N+2)
			if conc_method == "continuous"
				y_i_m, y_i_p = get_yi_candidates(xx, f, df, N)
				@assert check_concave_upper_optimality(y_i_m, y_i_p)
				yy = get_safe_yi(y_i_m, y_i_p, xx, f, this_interval_convex)
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
		#plot_bound(f, a, b, xp, yp; existing_plot=existing_plot)
		return xp, yp
	else
		#println("no plotting 4 u")
		return xp, yp
	end
end

# function overapprox(f,a,b,N; conc_method="continuous", df=nothing, d2f=nothing,
# 	d2f_zeros=nothing, convex=nothing, plot=false)

# 	LB = bound(f, a, b, N; conc_method=conc_method, lowerbound=true, df=df, d2f=d2f,
# 	d2f_zeros=d2f_zeros, convex=convex, plot=plot, reuse=false)

# 	UB = bound(f, a, b, N; conc_method=conc_method, lowerbound=false, df=df, d2f=d2f,
# 	d2f_zeros=d2f_zeros, convex=convex, plot=plot, reuse=true)
# 	return LB, UB
# end

function bound_optimal(f, a, b; rel_error_tol=0.005, Nmax = 20, conc_method="continuous",
	lowerbound=false, df=nothing, d2f=nothing, d2f_zeros=nothing, convex=nothing,
	plot=false)
	@debug("optimal tol is $(rel_error_tol)")

	try
		f(a)
		f(b)
	catch
		error("$a or $b is not in the domanin of $f")
	end

	if isnothing(df)
		df = Calculus.derivative(f)
	end
	if isnothing(d2f)
		d2f = Calculus.second_derivative(f)
	end
	if ! isnothing(convex) # if convexity is specified, no sub-intervals necessary.
		intervals = [(a,b)]
		@debug "convexity specified, no sub intervals necessary."
	else
		if isnothing(d2f_zeros) # calculate zeros of second derivative, if not given.
			println("WARNING: d2f_zeros have not been specified for function $f. Convex and concave regions will be identified using a numerical procedure. Soundness not guaranteed. ")
			d2f_zeros = fzeros(d2f, a, b)
		end
		intervals = give_interval(d2f_zeros, a, b)  # find subintervals.
	end
	xp = []
	yp = []
	for interval in intervals
		aa, bb = interval
		# specify convexity
		if isnothing(convex)
			this_interval_convex = (d2f((aa+bb)/2) >= 0)
		else
			@assert convex == (d2f((aa+bb)/2) >= 0)
			this_interval_convex = convex
		end
		for N = 1:Nmax
			xp_candidate, yp_candidate = bound(f, aa, bb, N; conc_method="continuous", lowerbound=lowerbound, df=df,
			d2f=d2f, d2f_zeros=d2f_zeros, convex=this_interval_convex, plot=plot)

			# interpolate can sometimes give an error because it thinks the
			# points are outside the domain.
			# see this issue https://github.com/JuliaMath/Interpolations.jl/issues/158
			# hence shorten the range of xtest by epsilon
			ϵ_itp = 1E-8


			itp = Interpolations.interpolate((xp_candidate[1],), yp_candidate[1], Gridded(Linear()))
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
		#plot_bound(f, a, b, xp, yp; existing_plot=existing_plot, saveflag=true);
		return xp, yp
	else
		return xp, yp
	end
end

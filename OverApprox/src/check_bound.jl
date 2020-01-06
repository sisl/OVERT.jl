using PyCall
include("overest_nd.jl")

function check_bound()
	py"""
	from dreal import *
	from check_bound import *
	"""
end

function sym_to_eval(x_sym)
	x_eval = eval(:(x -> $x_sym))
	return x_eval
end

# The line below allows to call local python modules.
current_folder = pwd()
pushfirst!(PyVector(pyimport("sys")."path"), current_folder);

check_bound()
expr = "sin(x)"
expr_dict_range= Dict(:x => (-2,2))
expr_bound = upperbound_expr(expr; N=1, range_dict=expr_dict_range)[1]
expr_bound = repr(expr_bound)[2:end]
expr_dict_range= Dict("x" => (-2,2))
sat = py"check_bound_2d"(expr, expr_bound, expr_dict_range)
if sat
	println("checked!")
else
	expr_val = sym_to_eval(Meta.parse(expr))(sat[2][1])
	expr_bound_val = sym_to_eval(Meta.parse(expr_bound))(sat[2][1])
	println("found a contradiction.")
	println("x = $(sat[2][1]), expr= $expr_val and expr_bound = $expr_bound_val")
end

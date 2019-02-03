# conveniance functions for julia

function eye(dim)
	return Matrix{Float64}(I, dim, dim)
end
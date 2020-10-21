# logical functions

mutable struct Property
    temporal # e.g. "always_true" or "eventually_true"
    type # e.g. "safe" or "avoid"
    constraint # e.g. hyperrectangle, for now, or list of constraints
end

"""
Can use matrix mul to create constraints:
julia> @constraint(model, con, A * x .== b)
2-element Array{ConstraintRef{Model,MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64},MathOptInterface.EqualTo{Float64}},ScalarShape},1}:
 x[1] + 2 x[2] == 5.0
 3 x[1] + 4 x[2] == 6.0
"""
mutable struct Constraint
    # e.g. 5*x + 3*theta < 3.2
    coeffs # 5, 3
    relation # <
    scalar # 3.2
end

# logical functions

mutable struct Property
    temporal # e.g. "always_true" or "eventually_true"
    type # e.g. "safe" or "avoid"
    constraint # e.g. hyperrectangle, for now, or list of constraints
end

mutable struct Constraint
    # e.g. 5*x + 3*theta < 3.2
    coeffs # 5, 3
    relation # <
    scalar # 3.2
end

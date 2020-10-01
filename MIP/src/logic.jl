# logical functions

mutable struct Property
    temporal # e.g. "always_true" or "eventually_true"
    type # e.g. "safe" or "avoid"
    constraint # e.g. hyperrectangle, for now 
end

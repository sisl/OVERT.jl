module OverApprox

include("overest_new.jl")
include("autoline.jl")

export overapprox,
       bound,
      closed_form_piecewise_linear,
      to_relu_expression

end

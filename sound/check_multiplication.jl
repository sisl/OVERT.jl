include("OverApprox/src/overapprox_nd_relational.jl")
include("OverApprox/src/overt_parser.jl")
include("MIP/src/overt_to_mip.jl")
include("MIP/src/mip_utils.jl")
include("sound/soundness.jl")

# check multiplication
# x*y, x ∈ [a,b] ∧ y ∈ [c,d], ξ>0
#          x2 = (x - a)/(b - a) + ξ  , x2 ∈ [ξ, 1 + ξ] aka x2 > 0   (recall, b > a)
#             x = (b - a)(x2 - ξ) + a
#          y2 = (y - c)/(d - c) + ξ , y2 ∈ [ξ, 1 + ξ] aka y2 > 0   (recall, d > c)
#             y = (d - c)(y2 - ξ) + c
#         x*y = ((b - a)(x2 - ξ) + a )*((d - c)(y2 - ξ) + c)
#             = (b - a)*(d - c)*x2*y2  + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))
#             = (b - a)*(d - c)*exp(log(x2*y2)) + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))
#             = (b - a)*(d - c)*exp(log(x2) + log(y2)) + (b - a)(c - ξ(d - c))*x2 + (d - c)(a - ξ(b - a))*y2 + (c - ξ(d - c))(a - ξ(b - a))

fs = FormulaStats()

# domain
domain_constraints1 = assert_all([:(a < b), :(c < d), :(e == 0.1)], fs)
d = Dict(:x => [:a,:b], :y => [:c, :d])
domain_constraints2 = define_domain(d, fs)
domain_constraints = vcat(domain_constraints1, domain_constraints2)
# new variables x2 and y2
x2_y2_def = assert_all([:(x2 == (x - a)/(b - a) + e)
                        , :(y2 == (y - c)/(d - c) + e)], fs)
# re-written x*y
rewritten_xty = assert_literal(:(xtimesy == (b - a)*(d - c)*exp(log(x2) + log(y2)) + (b - a)*(c - e*(d - c))*x2 + (d - c)*(a - e*(b - a))*y2 + (c - e*(d - c))*(a - e*(b - a))), fs)

# unsat expr
unsat_expr = assert_negated_literal(:(x*y == xtimesy), fs)

whole_formula = vcat(header(), declare_reals(fs), domain_constraints, x2_y2_def, rewritten_xty, unsat_expr, footer())
smt2f = SMTLibFormula(whole_formula, fs) 
write_to_file(smt2f, "check_multiplication.smt2", dirname="sound/")


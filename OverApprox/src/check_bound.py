from dreal import *
import numpy as np

def pass_to_dreal(expr, expr_bound, dict_range, upperbound=True, eps=1E-4):
    for i, var in enumerate(dict_range.keys()):
        if i == 0:
            cond = And(dict_range[var][0] <= var,
                             dict_range[var][1] >= var)
        else:
            cond = And(cond,
                             dict_range[var][0] <= var,
                             dict_range[var][1] >= var)
    if upperbound:
        cond = And(cond, expr_bound - expr <= -eps)
    else:
        cond = And(cond, expr - expr_bound <= -eps)
    sat = CheckSatisfiability(cond, eps)
    return cond, sat


def check_bound_2d(expr, expr_bound, expr_dict_range, nd=1):
    if nd == 1:
        x = Variable("x")
        dict_range = {x: expr_dict_range["x"]}
    elif nd == 2:
        x = Variable("x")
        y = Variable("y")
        dict_range = {x: expr_dict_range["x"], y: expr_dict_range["y"]}
    else:
        raise(NotImplementedError)

    if isinstance(expr, str) and isinstance(expr_bound, str):
        cond, sat = pass_to_dreal(eval(expr), eval(expr_bound), dict_range)
    else:
        raise(NotImplementedError)

    print(sat)
    if sat is None:
        return True
    else:
        return sat.keys(), [x.mid() for x in sat.values()]

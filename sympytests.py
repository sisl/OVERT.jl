from sympy import *
from sympy.solvers.solveset import linsolve
import numpy as np
from MC_constraints import ReluConstraint

def eval_constraints(tf_constraints, solution_dict = {}):
    # var dict: string variables names -> sympy variables
    # sol'n dict: string variable snames -> real values
    vardict = {}
    for i in range(1,tf_constraints.numVars+1):
        vardict['x'+str(i)] = symbols('x'+str(i))

    for c in tf_constraints.constraints:
        if not isinstance(c, ReluConstraint):
            varsineq = [vardict[v] for v in c.x.flatten().tolist()]
            A = c.A
            b = c.b
            for string_varname in solution_dict.keys():
                # if appears in solution dict already, add row to matrix
                if string_varname in c.x:
                    Aaddition = np.array([1 if e==string_varname else 0 for e in c.x]).reshape(1, -1)
                    A = Matrix(np.vstack((Aaddition, A)))
                    b = Matrix(np.vstack(([[solution_dict[string_varname]]], b)))
            solns = linsolve((A, b), varsineq)
            sol_list = list([i for i in solns][0])
            for i in range(len(c.x)):
                solution_dict[c.x[i][0]] = sol_list[i]
        else:
            # if isa ReluConstraint
            for i in range(len(c.varout)):
                solution_dict[c.varout[i]] = max(0, solution_dict[c.varin[i]])

    return solution_dict
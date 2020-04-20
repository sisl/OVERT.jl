import h5py
from keras.models import load_model

import os
import numpy as np
from MC_constraints import Monomial, Constraint, ConstraintType, MaxConstraint, ReluConstraint
from MC_Keras_parser import KerasConstraint, getNewVariable

#
# class OvertConstraint():
#     def __init__(self, file_name, var_dict={}):
#         self.f = h5py.File(file_name, "r")
#         self.n_eq  = len(self.f['/eq/']) // 3
#         self.n_min = len(self.f['/min/']) // 2
#         self.n_max = len(self.f['/max/']) // 2
#         self.n_ineq = len(self.f['/ineq/']) // 2
#
#         self.var_dict = var_dict
#         # var_dict is a dictionary with key and values of julia and corresponding python variables.
#         # when processing more than one overt function, e.g. for a double pendulum,
#         # the var_dict of the previous functions should be passed, to avoid creating dublicate variables.
#
#         self.eq_list = []
#         self.max_list = []
#         self.relu_list = []
#         self.ineq_list = []
#         self.state_vars = []
#         self.output_vars = []
#         self.control_vars = []
#
#         self.read_equations()
#         self.read_min_equations()
#         self.read_max_equations()
#         self.read_inequalities()
#         self.read_input_output_control_vars()
#         self.constraints = self.eq_list + self.max_list + self.relu_list + self.ineq_list
#
#     def read_input_output_control_vars(self):
#         self.state_vars = [self.var_dict[v] for v in self.f['vars/states'][()]]
#         self.control_vars = [self.var_dict[v] for v in self.f['vars/controls'][()]]
#         self.output_vars = [self.var_dict[v] for v in self.f['vars/outputs'][()]]
#
#     def read_equations(self):
#         for i in range(self.n_eq):
#             var = self.f['/eq/v%d'%(i+1)][()]
#             for v in var:
#                 if v not in self.var_dict.keys():
#                     self.var_dict[v] = v #getNewVariable()
#
#             coef = self.f['/eq/c%d'%(i+1)][()]
#             b = self.f['/eq/b%d'%(i+1)][()]
#             monomial_list = [Monomial(c, self.var_dict[v]) for (c, v) in zip(coef, var)]
#             self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, b))
#
#     def read_inequalities(self):
#         for i in range(self.n_ineq):
#             lvar = self.f['/ineq/l%d'%(i+1)][()][0]
#             rvar = self.f['/ineq/r%d'%(i+1)][()][0]
#             for v in [lvar, rvar]:
#                 if v not in self.var_dict.keys():
#                     self.var_dict[v] = v #getNewVariable()
#
#             # add lvar <= rvar
#             monomial_list = [Monomial(1, self.var_dict[lvar]),  Monomial(-1, self.var_dict[rvar])]
#             self.ineq_list.append(Constraint(ConstraintType('LESS_EQ'), monomial_list, 0))
#
#     def read_min_equations(self):
#         for i in range(self.n_min):
#             var = self.f['/min/v%d'%(i+1)][()]
#             var_out = var[0]
#             var_in1 = var[1]
#             var_in2 = var[2]
#             if var_out not in self.var_dict.keys(): self.var_dict[var_out] = var_out #getNewVariable()
#             if var_in1 not in self.var_dict.keys(): self.var_dict[var_in1] = var_in1 #getNewVariable()
#             if var_in2 not in self.var_dict.keys(): self.var_dict[var_in2] = var_in2 #getNewVariable()
#
#             coef = self.f['/min/c%d'%(i+1)][()]
#             coef1 = coef[0]
#             coef2 = coef[1]
#
#             assert((coef1 != 0) & (coef2 != 0) & (var_out != '0') & (var_in2 != '0'))
#
#             if (var_in1 != '0'):
#                 #z = min(ax, by) = > z2 = max(x2, y2),  z2=-z,  x2=-ax, y2=-by.
#                 # define new variables z2=-z,
#                 new_var_out = getNewVariable()
#                 monomial_list1 = [Monomial(1, self.var_dict[var_out]), Monomial(1, new_var_out)]  #z + z2 = 0
#                 self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list1, 0))
#
#                 # define new variable x2=-ax,
#                 new_var_in1 = getNewVariable()
#                 monomial_list2 = [Monomial(coef1, self.var_dict[var_in1]), Monomial(1, new_var_in1)]  # ax + x2 = 0
#                 self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list2, 0))
#
#                 # define new variable y2=-by,
#                 new_var_in2 = getNewVariable()
#                 monomial_list3 = [Monomial(coef2, self.var_dict[var_in2]), Monomial(1, new_var_in2)]  #by + y2 = 0
#                 self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list3, 0))
#
#                 self.max_list.append(MaxConstraint(varsin=[new_var_in1, new_var_in2], varout=new_var_out))  # z2 = max(x2, y2)
#             else:
#                 # z = min(0, ay) => z2 = relu(y2), z2=-z, y2=-ay
#                 new_var_out = getNewVariable()
#                 monomial_list = [Monomial(1, self.var_dict[var_out]), Monomial(1, new_var_out)]  # z + z2 = 0
#                 self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))
#
#                 # define new variable y2=-by,
#                 new_var_in2 = getNewVariable()
#                 monomial_list = [Monomial(coef2, self.var_dict[var_in2]), Monomial(1, new_var_in2)]  # by + y2 = 0
#                 self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))
#
#                 self.relu_list.append(ReluConstraint(varin=new_var_in2, varout=new_var_out))
#
#     def read_max_equations(self):  # these are all relu equations
#         for i in range(self.n_max):
#             var = self.f['/max/v%d'%(i+1)][()]
#             coef = self.f['/max/c%d'%(i+1)][()]
#             assert(var[1] == '0')
#
#             var_out = var[0]
#             var_in = var[2]
#             coef2 = coef[1]
#             if var_out not in self.var_dict.keys(): self.var_dict[var_out] = var_out # getNewVariable()
#             if var_in not in self.var_dict.keys(): self.var_dict[var_in] = var_in #getNewVariable()
#
#             # z = max(by, 0) => z = relu(y2), where y2 = -by
#             # define new variable y2=-by,
#             new_var_in = getNewVariable()
#             monomial_list = [Monomial(coef2, self.var_dict[var_in]), Monomial(1, new_var_in)]  # by + y2 = 0
#             self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))
#
#             self.relu_list.append(ReluConstraint(varin=new_var_in, varout=self.var_dict[var_out]))

class OvertConstraint():
    def __init__(self, file_name, var_dict={}):
        self.f = h5py.File(file_name, "r")
        self.n_eq  = len(self.f['/eq/']) // 3
        self.n_relu = len(self.f['/relu/']) // 2
        self.n_max = len(self.f['/max/']) // 2
        self.n_ineq = len(self.f['/ineq/']) // 2

        self.var_dict = var_dict
        # var_dict is a dictionary with key and values of julia and corresponding python variables.
        # when processing more than one overt function, e.g. for a double pendulum,
        # the var_dict of the previous functions should be passed, to avoid creating dublicate variables.

        self.eq_list = []
        self.max_list = []
        self.relu_list = []
        self.ineq_list = []
        self.state_vars = []
        self.output_vars = []
        self.control_vars = []

        self.read_equations()
        self.read_relu_equations()
        self.read_max_equations()
        self.read_inequalities()
        self.read_input_output_control_vars()
        self.constraints = self.eq_list + self.max_list + self.relu_list + self.ineq_list

    def read_input_output_control_vars(self):
        for v in self.f['vars/states'][()]:
            if v not in self.var_dict:
                self.var_dict[v] = getNewVariable('xd')
            self.state_vars.append(self.var_dict[v])
        for v in self.f['vars/controls'][()]:
            if v not in self.var_dict:
                self.var_dict[v] = getNewVariable('xd')
            self.control_vars.append(self.var_dict[v])
        for v in self.f['vars/outputs'][()]:
            if v not in self.var_dict:
                self.var_dict[v] = getNewVariable('xd')
            self.output_vars.append(self.var_dict[v])

    def read_equations(self):
        for i in range(self.n_eq):
            vars = self.f['/eq/vars%d'%(i+1)][()]
            for v in vars:
                if v not in self.var_dict.keys():
                    self.var_dict[v] = getNewVariable('xd')

            coeffs = self.f['/eq/coeffs%d'%(i+1)][()].astype(np.float)
            b = self.f['/eq/scalar%d'%(i+1)][()].astype(np.float)[0]
            monomial_list = [Monomial(c, self.var_dict[v]) for (c, v) in zip(coeffs, vars)]
            self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, b))

    def read_inequalities(self):
        for i in range(self.n_ineq):
            left_var = self.f['/ineq/varleft%d'%(i+1)][()][0]
            rite_var = self.f['/ineq/varright%d'%(i+1)][()][0]
            for v in [left_var, rite_var]:
                if v not in self.var_dict.keys():
                    self.var_dict[v] = getNewVariable('xd')

            # add lvar <= rvar
            monomial_list = [Monomial(1, self.var_dict[left_var]),  Monomial(-1, self.var_dict[rite_var])]
            self.ineq_list.append(Constraint(ConstraintType('LESS_EQ'), monomial_list, 0))

    def read_relu_equations(self):
        for i in range(self.n_relu):
            var_in = self.f['/relu/varin%d'%(i+1)][()][0]
            var_out = self.f['/relu/varout%d'%(i+1)][()][0]

            if var_out not in self.var_dict.keys(): self.var_dict[var_out] = getNewVariable('xd')
            if var_in not in self.var_dict.keys(): self.var_dict[var_in] = getNewVariable('xd')

            self.relu_list.append(ReluConstraint(varin=self.var_dict[var_in], varout=self.var_dict[var_out]))

    def read_max_equations(self):  # these are all relu equations
        for i in range(self.n_max):
            vars_in = self.f['/max/varsin%d'%(i+1)][()]
            var_out = self.f['/max/varout%d'%(i+1)][()][0]
            var_in1 = vars_in[0]
            var_in2 = vars_in[1]

            if var_out not in self.var_dict.keys(): self.var_dict[var_out] = getNewVariable('xd')
            if var_in1 not in self.var_dict.keys(): self.var_dict[var_in1] = getNewVariable('xd')
            if var_in2 not in self.var_dict.keys(): self.var_dict[var_in2] = getNewVariable('xd')

            self.max_list.append(MaxConstraint(varsin=[self.var_dict[var_in1], self.var_dict[var_in2]],
                                               varout=self.var_dict[var_out]))


if __name__ == "__main__":
    o = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")

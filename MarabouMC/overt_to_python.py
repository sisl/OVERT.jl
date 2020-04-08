import h5py
from keras.models import load_model

import os
import sys

#sys.path.insert(0, os.getcwd() + "/MarabouMC")
print(sys.path)
from MC_constraints import Monomial, Constraint, ConstraintType, MaxConstraint, ReluConstraint
from MC_Keras_parser import KerasConstraint, getNewVariable


class OvertConstraint():
    def __init__(self, file_name):
        self.f = h5py.File(file_name, "r")
        self.n_eq  = len(self.f['/eq/']) // 3
        self.n_min = len(self.f['/min/']) // 2
        self.n_max = len(self.f['/max/']) // 2
        self.n_ineq = len(self.f['/ineq/']) // 2

        self.var_dict = {}

        self.eq_list = []
        self.max_list = []
        self.relu_list = []
        self.ineq_list = []
        self.state_vars = []
        self.output_vars = []
        self.control_vars = []

        self.read_input_output_control_vars()
        self.read_equations()
        self.read_min_equations()
        self.read_max_equations()
        self.read_inequalities()
        self.constraints = self.eq_list + self.max_list + self.relu_list + self.ineq_list

    def read_input_output_control_vars(self):
        self.state_vars = self.f['vars/states'][()]
        self.control_vars = self.f['vars/controls'][()]
        self.output_vars = self.f['vars/outputs'][()]

    def read_equations(self):
        for i in range(self.n_eq):
            var = self.f['/eq/v%d'%(i+1)][()]
            for v in var:
                if v not in self.var_dict.keys():
                    self.var_dict[v] = getNewVariable()

            coef = self.f['/eq/c%d'%(i+1)][()]
            b = self.f['/eq/b%d'%(i+1)][()]
            monomial_list = [Monomial(c, self.var_dict[v]) for (c, v) in zip(coef, var)]
            self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, b))

    def read_inequalities(self):
        for i in range(self.n_ineq):
            lvar = self.f['/ineq/l%d'%(i+1)][()][0]
            rvar = self.f['/ineq/r%d'%(i+1)][()][0]
            for v in [lvar, rvar]:
                if v not in self.var_dict.keys():
                    self.var_dict[v] = getNewVariable()

            # add lvar <= rvar
            monomial_list = [Monomial(1, self.var_dict[lvar]),  Monomial(-1, self.var_dict[rvar])]
            self.ineq_list.append(Constraint(ConstraintType('LESS_EQ'), monomial_list, 0))

    def read_min_equations(self):
        for i in range(self.n_min):
            var = self.f['/min/v%d'%(i+1)][()]
            var_out = var[0]
            var_in1 = var[1]
            var_in2 = var[2]
            if var_out not in self.var_dict.keys(): self.var_dict[var_out] = getNewVariable()
            if var_in1 not in self.var_dict.keys(): self.var_dict[var_in1] = getNewVariable()
            if var_in2 not in self.var_dict.keys(): self.var_dict[var_in2] = getNewVariable()

            coef = self.f['/min/c%d'%(i+1)][()]
            coef1 = coef[0]
            coef2 = coef[1]

            assert((coef1 != 0) & (coef2 != 0) & (var_out != '0') & (var_in2 != '0'))

            if (var_in1 != '0'):
                #z = min(ax, by) = > z2 = max(x2, y2),  z2=-z,  x2=-ax, y2=-by.
                # define new variables z2=-z,
                new_var_out = getNewVariable()
                monomial_list = [Monomial(1, self.var_dict[var_out]), Monomial(1, new_var_out)]  #z + z2 = 0
                self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

                # define new variable x2=-ax,
                new_var_in1 = getNewVariable()
                monomial_list = [Monomial(coef1, self.var_dict[var_in1]), Monomial(1, new_var_in1)]  # ax + x2 = 0
                self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

                # define new variable y2=-by,
                new_var_in2 = getNewVariable()
                monomial_list = [Monomial(coef2, self.var_dict[var_in2]), Monomial(1, new_var_in2)]  #by + y2 = 0
                self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

                self.max_list.append(MaxConstraint([new_var_in1, new_var_in2], new_var_out))  # z2 = max(x2, y2)
            else:
                # z = min(0, ay) => z2 = relu(y2), z2=-z, y2=-ay
                new_var_out = getNewVariable()
                monomial_list = [Monomial(1, self.var_dict[var_out]), Monomial(1, new_var_out)]  # z + z2 = 0
                self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

                # define new variable y2=-by,
                new_var_in2 = getNewVariable()
                monomial_list = [Monomial(coef2, self.var_dict[var_in2]), Monomial(1, new_var_in2)]  # by + y2 = 0
                self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

                self.relu_list.append(ReluConstraint(new_var_in2, new_var_out))

    def read_max_equations(self):  # these are all relu equations
        for i in range(self.n_max):
            var = self.f['/max/v%d'%(i+1)][()]
            coef = self.f['/max/c%d'%(i+1)][()]
            assert(var[1] == '0')

            var_out = var[0]
            var_in = var[2]
            coef2 = coef[1]
            if var_out not in self.var_dict.keys(): self.var_dict[var_out] = getNewVariable()
            if var_in not in self.var_dict.keys(): self.var_dict[var_in] = getNewVariable()

            # define new variable y2=-by,
            new_var_in = getNewVariable()
            monomial_list = [Monomial(coef2, self.var_dict[var_in]), Monomial(1, new_var_in)]  # by + y2 = 0
            self.eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0))

            self.relu_list.append(ReluConstraint(self.var_dict[var_out], new_var_in))

# class SinglePendulumOvertDynamics(OvertConstraint):
#     def __init__(self, file_name, dt):
#         super().__init__(file_name)
#
#         self.dt = dt
#
#     def compute_next_state(self):
#         th, dth = self.state_vars
#         torque = self.control_vars
#         ddth = self.output_vars
#
#         return next_state_vars
        # add

# m1 = load_model("../OverApprox/models/single_pend.h5")
# p = KerasConstraint(m1)
# o = OvertConstraint("../OverApprox/src/up.h5")
#
# print(o.state_vars)
# print(o.control_vars)
# print(o.output_vars)
# # consolidate_constraints
#
# print(getNewVariable())
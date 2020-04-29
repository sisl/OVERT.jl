import numpy as np
from keras.models import load_model
from keras.layers import Dense, SimpleRNN
from MC_constraints import ConstraintType, Constraint, MatrixConstraint, ReluConstraint, Monomial, MaxConstraint

import tensorflow as tf


NUM_VARS = 0
def getNewVariable(var_sym='xc'):
    """
    This function generates variables as strings x1, x2, ...
    NUM_VARS keeps the record of variables produced.
    """
    global NUM_VARS
    NUM_VARS += 1
    return var_sym+str(NUM_VARS)


class KerasConstraint():
    """
    This class parse a sequential tensorflow model build with keras.
    Only Dense and SimpleRNN are supported for now.
    If there is any SimpleRNN layers, given that the parsing happens during test time, it is assumed
        1) statefull=True, meaning that it will remember the previous states
        2) return_sequence=True, meaning that the output is of the same size as the input.
        3) batch_input_shape=(1, 1, Some_Number), the batch size must be specified because of stateful condition,
                                                it is assumed batch size during test is 1.

    TODO: Can I incorporate n_t in the batch_input_shape? The could be two options here for batch_input_shape.

    input
        - model: a sequential tensorflow model.
        - n_t: number of timesteps of unrolling. Needed for parsing SimpleRNN layers.
    main attribute
        - constrain: list of all Matrix Equality and Relu constraints that'll passed to Marabou
    auxillary attributes:
        - model: model (keras.models)
        - layers: layers of the model (list of keras.models.layers)
        - names: names of each layer (list of strings)
        - input_sizes: size of input variable for each layer (list of integers)
        - output_sizes: size of output variable for each layer (list of integers)
        - activations: activation functions of each layer. Only linear and relu are supported (list of strings)
        - input_vars: list of all input variables assigned to each layer (list of lists)
        - output_vars: list of all output variables assigned to each layer (list of lists)
        - model_input_vars: list of variables that input to the network (inputs of the first layer)
        - model_output_vars: list of variables that output to the network (output of the last layer AFTER activation function)
        - type: list of all layer types. Only Dense and SimpleRNN are supported. (list of strings)
    methods:
        - setup_constraint: depending on the layer, generates enough number of input and output variables and assigns them
                            to each layer. Then the constraints are made using rnn_constraint or dense constraint methods.

    """
    def __init__(self, model, n_time=1, condensed=True):
        self.model  = model
        self.layers = model.layers
        self.n_time = n_time
        self.names = []
        self.type  = []
        self.input_sizes = []
        self.output_sizes= []
        self.activations = []
        self.input_vars  = []
        self.output_vars = []
        self.constraints = []
        self.model_input_vars = []
        self.model_output_vars= []

        for l in self.layers:
            self.names.append(l.input.name)
            self.input_sizes.append(l.input.shape.as_list()[-1])
            self.output_sizes.append(l.output.shape.as_list()[-1])
            self.activations.append(self.find_activation(l))

        # setup all equality and relu constraints
        self.set_constraint(condensed=condensed)

        # specify the input and output variables of the model and setup the
        # the constraints for the output layer
        self.set_input_output(condensed=condensed)

        self.check_constraints()

    def set_constraint(self, condensed=False):
        """
        This function assigned input and output variables to each layer.
        The construct the equality matrix constraints that prescribe the behavior of that model
        If there is any RNN layer, variables are built for all rollouts, i.e. n_input = n_output = n_input_1 * n_t
        The output of each layer is connected via the input of the next layer via the activation function.
        If the activation is relu, a reluconstraint is built. If the activation is linear, an identity maping is built.
        The later is somewhat redundant, but more expressive.

        if condensed=True, instead of creating identity mapping for layers with linear activation,
        the output of the layer is directly assigned to the input of the next layer.
        No new variables are created for the next layer inputs.
        """
        n_t = self.n_time
        for i in range(len(self.layers)):
            l, s_in, s_out = self.layers[i], self.input_sizes[i], self.output_sizes[i]

            # assign all input output variables.
            # if condensed option, and no activation, output variables of previous layer is assigned to the input of the current layer
            if (condensed) and (i > 0) and (self.activations[i-1] == "linear"):
                x_in = self.output_vars[-1]  # assign last layer output
            else:
                x_in = [getNewVariable() for _ in range(s_in * n_t)] # assign new variables.

            # assign all input output variables.
            x_out = [getNewVariable() for _ in range(s_out * n_t)]
            self.input_vars.append(x_in)
            self.output_vars.append(x_out)

            # setup equality constraints depending on the layer type
            if isinstance(l, Dense):
                self.type.append("Dense")
                A, b = self.dense_constraint(n_t, *l.get_weights())
            elif isinstance(l, SimpleRNN):
                self.type.append("SimpleRNN")
                A, b = self.rnn_constraint(n_t, *l.get_weights())
            else:
                raise (NotImplementedError("only Dense and SimpleRNN layers are supported."))

            # add constraint to the list of constraints. variables include all input and output variables.
            x = np.array(x_in + x_out)  # concatenate two lists
            self.constraints.append(MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x, b=b))

        # setup activation function equality (for linear) or inequality (for relu) constraints.
        for i in range(len(self.layers) - 1):
            assert len(self.input_vars[i+1]) == len(self.output_vars[i])
            if self.activations[i] == "relu":
                for v_in, v_out in zip(self.output_vars[i], self.input_vars[i+1]):
                    self.constraints.append(ReluConstraint(varin=v_in, varout=v_out))
            elif self.activations[i] == "linear":
                if condensed:
                    pass # no mapping is necessary
                else:
                    w = np.eye(len(self.output_vars[i]))
                    A = np.hstack((w, -w))
                    b = np.zeros(2*len(self.output_vars[i]))
                    x = np.array(self.input_vars[i+1] + self.output_vars[i])
                    self.constraints.append(MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x, b=b))
            else:
                raise(IOError("Activation %s is not supported" %self.activations[i]))

        # the output of the last layer after activation function may change.
        # the necessary constraint for this is set in self.last_layer_constraint

    @staticmethod
    def rnn_constraint(n_t, weight_x, weight_h, bias):
        n_in, n_out = weight_x.shape
        w = np.zeros((n_out * n_t, (n_in + n_out) * n_t))
        b = np.zeros(n_out * n_t)
        for i in range(n_t):
            for j in range(n_out):
                w[i * n_out + j, i * n_in: (i + 1) * n_in] = weight_x[:, j]
                w[i * n_out + j, n_in * n_t + i * n_out + j] = -1.
                if i > 0:
                    w[i * n_out + j, n_in * n_t + (i - 1) * n_out: n_in * n_t + i * n_out] = weight_h[:, j]
                b[i * n_out + j] = -bias[j]
        return w, b

    # @staticmethod
    # def dense_constraint(weight_x, bias):
    #     n_in, n_out = weight_x.shape
    #     w = np.zeros((n_out, n_in + n_out))
    #     b = np.zeros(n_out)
    #     for j in range(n_out):
    #         w[j, :n_in] = weight_x[:, j]
    #         w[j, n_in + j] = -1
    #         b[j] = -bias[j]
    #
    #     return w, b

    def dense_constraint(self, n_t, weight_x, bias):
        dummy_w_h = np.zeros((weight_x.shape[1], weight_x.shape[1]))
        return self.rnn_constraint(n_t, weight_x, dummy_w_h, bias)

    @staticmethod
    def find_activation(layer):
        if layer.activation(-1) == -1:
            return "linear"
        elif (tf.is_tensor(layer.activation(0))) & ('Relu' in layer.activation(0).name):
            return "relu"
        else:
            raise (IOError("activation function is not supported"))

    def set_input_output(self, condensed=False):
        # find input variables of the model
        input_idx = [l.input.name for l in self.layers].index(self.model.input.name)
        self.model_input_vars = self.input_vars[input_idx].copy()

        # find output variables of the model
        output_layer_idx = [l.output.name for l in self.layers].index(self.model.output.name)
        output_layer_activ_func = self.activations[output_layer_idx]
        if (condensed) and output_layer_activ_func == "linear":
            self.model_output_vars = self.output_vars[output_layer_idx].copy()
            # no additional constraint
        else:
            output_layer_before_activ_func = self.output_vars[output_layer_idx]
            output_layer_after_active_func = [getNewVariable() for _ in self.output_vars[output_layer_idx]]
            self.model_output_vars = output_layer_after_active_func
            if output_layer_activ_func == "linear":
                w = np.eye(len(output_layer_before_activ_func))
                A = np.hstack((w, -w))
                b = np.zeros(2 * len(output_layer_before_activ_func))
                x = np.array(output_layer_before_activ_func + output_layer_after_active_func)
                self.constraints.append(MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x, b=b))
            else:
                for v_in, v_out in zip(output_layer_before_activ_func, output_layer_after_active_func):
                    self.constraints.append(ReluConstraint(varin=v_in, varout=v_out))

    def check_constraints(self):
        if all([type(c) == MatrixConstraint for c in self.constraints]):
            self.check_constraints_no_relu()
        elif all([a == "relu" for a in self.activations[:-1]]):
            self.check_constraints_with_relu()
        else:
            raise(NotImplementedError("for a relu network, only the last layer can have linear activation."))

    def check_constraints_no_relu(self):
        """
        This function compares the output of parser and a tensorflow output for random input values.
        No relu activation should be present.

        """
        # find input variables of the model, and assign random numbers.
        input_variables = self.model_input_vars
        input_values = [np.random.random() for _ in input_variables]
        input_variable_value_dict = dict(zip(input_variables, input_values))

        # consolidate all equality constraints
        all_variables = [x for xs in self.input_vars for x in xs] + [x for xs in self.output_vars for x in xs]
        all_variables = list(set(all_variables))  # remove repeated variables.
        n_vars = len(all_variables)
        A_tot = np.zeros((n_vars, n_vars))
        b_tot = np.zeros(n_vars)

        n_eq = 0
        for j, v in enumerate(all_variables):
            if v in input_variable_value_dict.keys():
                A_tot[n_eq, j] = 1
                b_tot[n_eq] = input_variable_value_dict[v]
                n_eq += 1

        for c in self.constraints:
            if isinstance(c, ReluConstraint):
                raise(IOError("This check does not work for relu constraints."))
            else:
                x = c.x.flatten().tolist()
                A = c.A
                b = c.b
                idx = [all_variables.index(v) for v in x]
                for j in range(A.shape[0]):
                    A_tot[n_eq, idx] = A[j, :]
                    b_tot[n_eq] = b[j]
                    n_eq += 1

        assert (n_eq) == n_vars # check number of variables and number of equations.

        # solve the linear system
        sol = np.linalg.solve(A_tot, b_tot)

        # find output variables of the model
        output_variables = self.model_output_vars
        output_variables_idx = [all_variables.index(v) for v in output_variables]
        sol_parser = sol[output_variables_idx]

        # find tf sol
        sol_tf = np.array([])
        assert len(input_variables) % self.n_time == 0
        m = len(input_variables) // self.n_time
        for i in range(self.n_time):
            x = np.array(input_values[i*m:(i+1)*m])
            if len(self.model.input_shape) == 2:
                x = x.reshape(1, -1)
            else:
                x = x.reshape(1, 1, -1)
            y = self.model.predict(x)
            sol_tf = np.concatenate((sol_tf, y.reshape(-1)))

        if np.all(np.isclose(sol_parser, sol_tf)):
            print("Keras parser: Test passed.")
        else:
            raise(ValueError("Keras parser: Test did not pass!"))

    def check_constraints_with_relu(self):
        """
        This function compares the output of parser and a tensorflow output for random input values.
        activations should contain at least one relu constraint.

        This only works for n_t = 1

        """

        assert self.n_time == 1

        # find input variables of the model, and assign random numbers.
        input_variables = self.model_input_vars
        input_values = [np.random.random() for _ in input_variables]
        input_variable_value_dict = dict(zip(input_variables, input_values))

        relu_constraints = [c for c in self.constraints if type(c) == ReluConstraint]
        eq_constraints = [c for c in self.constraints if type(c) == MatrixConstraint]

        for eq in eq_constraints:
            A = eq.A.copy()
            b = eq.b.copy()
            x = eq.x.tolist()
            common_x = set(x).intersection(set(input_variable_value_dict.keys()))
            assert(len(common_x) == A.shape[1] - A.shape[0])  # check if the constraint can be removed.
            for v in common_x:
                new_row = np.zeros((1, A.shape[1]))
                new_row[0, x.index(v)] = 1
                A = np.concatenate((A, new_row))
                b = np.concatenate((b, [input_variable_value_dict[v]]))
            sol = np.linalg.solve(A, b)
            for v, s in zip(x, sol):
                input_variable_value_dict[v] = s

            for rel in relu_constraints:
                varin = rel.varin
                if varin in input_variable_value_dict.keys():  # this constrained can be removed.
                    varout = rel.varout
                    input_variable_value_dict[varout] = max(0, input_variable_value_dict[varin])

        # find output variables of the model
        output_variables = self.model_output_vars
        sol_parser = [input_variable_value_dict[v] for v in output_variables]

        # find tf sol
        sol_tf = np.array([])
        for i in range(self.n_time):
            x = np.array(input_values)
            if len(self.model.input_shape) == 2:
                x = x.reshape(1, -1)
            else:
                x = x.reshape(1, 1, -1)
            y = self.model.predict(x)
            sol_tf = np.concatenate((sol_tf, y.reshape(-1)))

        if np.all(np.isclose(sol_parser, sol_tf)):
            print("Keras parser: Test passed.")
            #print("sol_tf:", sol_tf, " sol_parser:", sol_parser)
        else:
            #print("sol_tf:", sol_tf, " sol_parser:", sol_parser)
            raise(ValueError("Keras parser: Test did not pass!"))

    def cap(self, cap_min, cap_max):
        # this function adds additional constraints to cap the controller between cap_min and cap_max
        # cap_min and cap_max are both a list of size self.model_output_vars. None indicate no cap.
        # given x, return min(max(x, cap_min), cap_max)
        assert len(cap_min) == len(cap_max) == len(self.model_output_vars)
        for idx in range(len(self.model_output_vars)):
            self._cap_top(idx, cap_max[idx])
            self._cap_bottom(idx, cap_min[idx])

    def _cap_bottom(self, idx, cap_min):
        if cap_min is None:
            return

        # x_out =  max(x, cap_min)
        x = self.model_output_vars[idx]
        xs1 = getNewVariable("xs")
        x_out = getNewVariable("xs")
        constraint1 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs1)], scalar=cap_min)  # set xs1 = cap_min
        constraint2 = MaxConstraint(varsin=[x, xs1], varout=x_out)  # x_out = max(x, xs1)
        self.constraints += [constraint1, constraint2]
        self.model_output_vars[idx] = x_out

    def _cap_top(self, idx, cap_max):
        if cap_max is None:
            return

        # x_out = min(x, cap_max) or equivalently x_out = -max(-x, -cap_max)
        # x_out = -xs1, xs1 = max(-x, -cap_max) or equivalently
        # x_out = -xs1, xs1 = max(xs2, xs3), xs2 = -x, xs3 = -cap_max
        x = self.model_output_vars[idx]
        xs1 = getNewVariable("xs")
        xs2 = getNewVariable("xs")
        xs3 = getNewVariable("xs")
        x_out = getNewVariable("xs")

        constraint1 = Constraint("EQUALITY", monomials=[Monomial(1.0, x_out), Monomial(1.0, xs1)], scalar=0)  # x_out = -xs1
        constraint2 = MaxConstraint(varsin=[xs2, xs3], varout=xs1)  # xs1 = max(xs2, xs3)
        constraint3 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs2), Monomial(1.0, x)], scalar=0)  # set xs2 = -x
        constraint4 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs3)], scalar=-cap_max)  # set xs1 = -cap_max
        self.constraints += [constraint1, constraint2, constraint3, constraint4]
        self.model_output_vars[idx] = x_out


if __name__ == "__main__":
    n_t = 1
    m1 = load_model('dense_rnn_model_stateful.h5')
    p1 = KerasConstraint(m1, n_time=n_t)

    #m2 = load_model('../OverApprox/models/single_pend_controller_nn_not_trained.h5')
    m2 = load_model("m2.h5")
    p2 = KerasConstraint(m2, n_time=n_t)
# A, b = eval_constraints(cfeed)
# sol = np.linalg.solve(A, b)
# all_variables = [x for xs in p.input_vars for x in xs] + [x for xs in p.output_vars for x in xs]
# all_variables = list(set(all_variables))
# output_idx = [all_variables.index(v) for v in output_variables]
# print(sol[output_idx])
#
# print(model.predict(np.array(input_values[:5]).reshape(1,1,-1)))
# print(model.predict(np.array(input_values[5:]).reshape(1,1,-1)))



# TF parser 1.x for model checker interface to marabou
# a class that loads a TF pb and produces a set of constraints that represent the network

# Based on MarabouNetworkTF.py and MarabouNetwork.py

import numpy as np
from tensorflow.python.framework import tensor_util
from sympy import *
from sympy.solvers.solveset import linsolve
#from tensorflow.python.framework import graph_util
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from MC_constraints import ConstraintType, Constraint, MatrixConstraint, ReluConstraint

import tensorflow as tf

class TFConstraint:
    def __init__(self, filename="", sess=None, inputNames=None, outputName=None):
        """
        Constructs a TFConstraint object from a frozen Tensorflow protobuf

        Args:
            filename: (string) path to the frozen graph .pb file.
            sess: active session with associated graph that we want to parse.
            inputNames: [(string)] optional, names of operations corresponding to inputs.
            outputName: (string) optional, names of operation corresponding to output.

        Note: you can always use adds and matmuls to turn multiple output ops into a single output op
        """
        # initializations
        self.constraints = []
        # aux: 
        self.inputVars = []
        self.outputVars = []
        self.inputOps = None
        self.outputOp = None
        self.numVars = 0
        self.varMap = dict() # maps ops -> vars
        self.shapeMap = dict() # maps ops -> tensor shapes
        if sess is None:
            self.readFromPb(filename)
        else:
            self.sess = sess # should have associated graph in session
            # TODO: If tf.Variables haven't been converted to tf.Constants, will this cause
            # any problems?
        ##################### The Important Bit ###################################
        self.parse(inputNames, outputName) 
        ###########################################################################
        self.relus = [c for c in self.constraints if isinstance(c, ReluConstraint)]
    
    def getNewVariable(self):
        self.numVars += 1
        # x1, x2, x3, ... etc
        return 'x'+str(self.numVars)

    def clear(self):
        """
        Reset values to represent empty network
        """
        self.constraints = []
        self.relus = []
        self.max = []
        # aux: 
        self.inputVars = []
        self.outputVars = []
        self.inputOps = None
        self.outputOp = None
        self.varMap = dict() # maps ops -> vars
        self.shapeMap = dict() # maps ops -> their shapes
    
    def setInputOps(self, ops):
        """
        Function to set input operations
        Arguments:
            [ops]: (tf.op) list representing input
        """
        self.inputVars = []
        for op in ops:
            try:
                shape = tuple(op.outputs[0].shape.as_list())
                self.shapeMap[op] = shape
            except:
                self.shapeMap[op] = [None]
            self.inputVars.append(self.opToVarArray(op))
        self.inputOps = ops

    def setOutputOp(self, op):
        """
        Function to set output operation
        Arguments:
            op: (tf.op) Representing output
        """
        try:
            shape = tuple(op.outputs[0].shape.as_list())
            self.shapeMap[op] = shape
        except:
            self.shapeMap[op] = [None]
        self.outputOp = op
        self.outputVars = self.opToVarArray(self.outputOp)
        
    def readFromPb(self, filename):
        """
        Sets self.sess from a frozen Tensorflow protobuf 
        Args:
            filename: (string) path to the frozen graph .pb file.  
        """
         ### Read protobuf file and begin session ### after "freezing"
        with tf.gfile.GFile(filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        self.sess = tf.Session(graph=graph)
        ### END reading protobuf ###

    def parse(self, inputNames, outputName):
        """
        inputNames: [(string)] optional, name of operation corresponding to input.
        outputNames: [(string)] optional, name of operation corresponding to output.
        """
        # find and set input and output operations
        inputOps, OutputOp = self.find_inputs_outputs(inputNames, outputName)
        self.setInputOps(inputOps)
        self.setOutputOp(OutputOp)

        ### Traverse the graph of the network and generate constraints ###
        self.foundnInputFlags = 0
        self.traverseGraph(self.outputOp)
        assert self.foundnInputFlags == len(inputOps)
        ### END generating equations ###
    
    def find_inputs_outputs(self, inputNames, outputName):
        ### Find operations corresponding to input and output ###
        if inputNames: # is not None
            inputOps = []
            for i in inputNames:
               inputOps.append(self.sess.graph.get_operation_by_name(i))
        else: # If there are placeholders, use these as inputs
            ops = self.sess.graph.get_operations()
            placeholders = [x for x in ops if x.node_def.op == 'Placeholder']
            inputOps = placeholders
        if outputName: # is not None
            outputOp = self.sess.graph.get_operation_by_name(outputName)
        else: # Assume that the last operation is the output
            outputOp = self.sess.graph.get_operations()[-1]
        return inputOps, outputOp
    
    def traverseGraph(self, op):
        """
        Function to generate equations for network necessary to calculate op
        Arguments:
            op: (tf.op) representing operation until which we want to generate network equations
        """
        if (op in self.varMap.keys()) and (op not in [*self.inputOps, self.outputOp]):
            return
        if op in self.inputOps:
            self.foundnInputFlags += 1
        #
        in_ops = [x.op for x in op.inputs]
        for x in in_ops:
            self.traverseGraph(x)
        self.makeNeuronEquations(op)
    
    def makeNeuronEquations(self, op):
        """
        Function to generate equations corresponding to given operation
        Arguments:
            op: (tf.op) for which to generate equations
        TODO:
        support: 'Pack', 'ConcatV2', 'Split', 'StridedSlice'
        """
        # no need to add constraints for these ops
        if op.node_def.op in ['Identity', 'Shape', 'Reshape', 'Placeholder', 'Const', 'Transpose']:
            return
        # need to add constraints for these ops
        if op.node_def.op == 'MatMul':
            self.matMulConstraint(op)
        elif op.node_def.op == 'Mul':
            input_ops = [i.op for i in op.inputs]
            if self.isVariable(input_ops[0]) or self.isVariable(input_ops[1]): 
                self.mulConstraint(op)
            else:
                return
        elif op.node_def.op == 'BiasAdd':
            self.biasAddConstraint(op)
        elif op.node_def.op == 'Add':
            self.additionConstraint(op)
        # elif op.node_def.op == 'Sub':
        #     self.subEquations(op)
        # elif op.node_def.op == 'Conv2D':
        #     self.conv2DEquations(op)
        elif op.node_def.op == 'Relu':
            self.reluConstraint(op)
        # elif op.node_def.op == 'Maximum':
        #     self.maxEquations(op)
        # elif op.node_def.op == 'MaxPool':
        #     self.maxpoolEquations(op)
        else:
            print("Operation ", str(op.node_def.op), " not implemented")
            raise NotImplementedError

    def opToVarArray(self, x):
        """
        Function to find variables corresponding to operation outputs
        Arguments:
            x: (tf.op) the operation to find variables for
        Returns:
            v: (np array) of variables, in same shape as x
        """
        if x in self.varMap:
            return self.varMap[x]

        ### Find number of new variables needed ###
        if x in self.shapeMap:
            shape = self.shapeMap[x]
            shape = [a if a is not None else 1 for a in shape]
        else:
            shape = [a if a is not None else 1 for a in x.outputs[0].get_shape().as_list()]
        size = 1
        for a in shape:
            size*=a
        ### END finding number of new variables ###

        v = np.array([self.getNewVariable() for _ in range(size)]).reshape(shape)
        self.varMap[x] = v
        return v

    def getValues(self, op):
        """
        Function to find underlying constants/variables representing operation
        Arguments:
            op: (tf.op) to get values of
        Returns:
            values: (np array) of scalars or variable numbers depending on op
        """
        input_ops = [i.op for i in op.inputs]
        
        ### Operations not requiring new variables ###
        if op.node_def.op in ['Identity', 'Shape']:
            return self.getValues(input_ops[0])
        if op.node_def.op in ['Reshape']:
            if input_ops[1].node_def.op == 'Pack':
                raise NotImplementedError
                # inputValues = self.getValues(input_ops[0])
                # input_dims = op.inputs[0].shape.dims
                # input_size = np.prod(np.array([d.value for d in input_dims])[1:])
                # shape = (-1, input_size)
            else:
                inputValues = [self.getValues(i) for i in input_ops]
                shape = inputValues[1]
            return np.reshape(inputValues[0], shape)
        if op.node_def.op == 'Transpose':
            print("Using transpose op")
            inputValues = [self.getValues(i) for i in input_ops]
            if len(inputValues) > 1:
                permutation = inputValues[1]
            else:
                permutation = None
            return np.transpose(inputValues[0], axes=permutation)
            # how to ask if the keyword argument for complex numbers has been passed?
            #assert inputValues[3] != True, "Complex transpose not supported"
        # if op.node_def.op == 'ConcatV2':
        #     inputValues = [self.getValues(i) for i in input_ops]
        #     values = inputValues[0:2]
        #     axis = inputValues[2]
        #     return np.concatenate(values, axis=axis)
        # if op.node_def.op == 'Split':
        #     print("using split op.")
        #     inputValues = [self.getValues(i) for i in input_ops]
        #     axis = int(inputValues[0])
        #     values = inputValues[1]
        #     indices_or_sections = op.get_attr("num_split")
        #     return np.split(values, indices_or_sections, axis=axis)
        #     # TODO: implement support for when you don't pass "axis"
        if op.node_def.op == 'Const':
            tproto = op.node_def.attr['value'].tensor
            return tensor_util.MakeNdarray(tproto)
        if op.node_def.op == 'Mul':
            if (not self.isVariable(input_ops[0])) and (not self.isVariable(input_ops[1])):
                i0 = self.getValues(input_ops[0])
                i1 = self.getValues(input_ops[1])
                # allow broadcasting
                return np.multiply(i0, i1)

        ### END operations not requiring new variables ###

        if op.node_def.op in ['MatMul', 'Mul', 'BiasAdd', 'Add', 'Sub', 'Relu', 'Maximum', 'MaxPool', 'Conv2D', 'Placeholder']:
            # need to create variables for these
            return self.opToVarArray(op)

        raise NotImplementedError

    def isVariable(self, op):
        """
        Function returning whether operation represents variable or constant
        Arguments:
            op: (tf.op) representing operation in network
        Returns:
            isVariable: (bool) true if variable, false if constant
        """
        if op.node_def.op == 'Placeholder':
            return True
        if op.node_def.op == 'Const':
            return False
        return any([self.isVariable(i.op) for i in op.inputs])

    def matMulConstraint(self, op):
        """
        Function to generate constraints corresponding to matrix multiplication
        Arguments:
            op: (tf.op) representing matrix multiplication operation
        """

        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        if self.isVariable(input_ops[0]):
            convention = "xW"
        elif self.isVariable(input_ops[1]):
            convention = "Wx"
        else:
            raise NotImplementedError
        inputValues = [self.getValues(i) for i in input_ops] 
        outputValues = self.getValues(op) 
        aTranspose = op.node_def.attr['transpose_a'].b
        bTranspose = op.node_def.attr['transpose_b'].b
        a = inputValues[0]
        b = inputValues[1]
        if aTranspose:
            a = np.transpose(a)
        if bTranspose:
            b = np.transpose(b)
        assert (a.shape[0], b.shape[1]) == outputValues.shape
        assert a.shape[1] == b.shape[0]
        ### END getting inputs ###

        ### Generate actual constraints ###
        # Wx = y
        # [W, -I] [x; y] = 0
        # W \in mxn
        # I \in mxm
        # x \in nx1
        # y \in mx1    
        if convention == "xW":
            x = a
            W = b
            # take transpose of W and store: from xW = y to W^T x^T = y^T to [W.T, -I] [x^T; y^T] = 0
            A = np.hstack((W.T, -np.eye(W.shape[1])))
            constraint_x = np.vstack((x.T, outputValues.T))
            constraint_b = np.zeros((W.shape[1], 1))
            c = MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=constraint_x, b=constraint_b)
            self.constraints.append(c)
        elif convention == "Wx":
            W = a
            x = b
            # Wx = y -> [W, -I] [x; y] = 0
            A = np.hstack((W, -np.eye(W.shape[0])))
            constraint_x = np.vstack((x, outputValues))
            constraint_b = np.zeros((W.shape[0], 1))
            c = MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=constraint_x, b=constraint_b)
            self.constraints.append(c)
        else:
            print("Whatchyu doin bro??")
            raise NotImplementedError

    def mulConstraint(self, op):
        """
        Function to generate equations corresponding to elementwise matrix multiplication 
        Arguments:
            op: (tf.op) representing elementwise multiplication operation
        TODO: this is unecessarily verbose
        """
        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        inputValues = [self.getValues(i) for i in input_ops]
        outputValues = self.getValues(op)
        assert not (self.isVariable(input_ops[0]) and self.isVariable(input_ops[1]))
        if self.isVariable(input_ops[0]):
            #convention = "xW"
            x = inputValues[0]
            W = inputValues[1]
        elif self.isVariable(input_ops[1]):
            #convention = "Wx"
            W = inputValues[0]
            x = inputValues[1]
        else:
            print("Multiplying two constants not supported")
            import pdb; pdb.set_trace()
            raise NotImplementedError
        W = W.reshape(-1)
        x = x.reshape(-1)
        if x.shape != W.shape:
            # broadcast
            W = np.tile(W, len(x)//len(W))
        assert x.shape == W.shape
        y = outputValues.reshape(-1)
        assert x.shape == y.shape
        ### END getting inputs ###

        ### Generate actual equations ###
        # w^T x = y
        # [w^T -I] [x; y] = 0
        #  1xn 1x1 
        A = np.hstack((W.T, -np.eye(1)))
        x_constraint = np.vstack((x, y))
        b_constraint = np.zeros((1,1));
        c = MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x_constraint, b=b_constraint)
        self.constraints.append(c)

    def biasAddConstraint(self, op):
        """
        Function to generate equations corresponding to bias addition
        Arguments:
            op: (tf.op) representing bias add operation
        """
        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        assert len(input_ops) == 2
        inputValues = [self.getValues(i) for i in input_ops]
        outputValues = self.getValues(op)
        inputVars = inputValues[0].reshape(-1,1)
        inputConsts = inputValues[1].reshape(-1,1)
        # broadcasting
        inputConsts = np.tile(inputConsts, len(inputVars)//len(inputConsts))
        outputVars = outputValues.reshape(-1,1)
        assert len(inputVars)==len(outputVars) and len(outputVars)==len(inputConsts)
        ### END getting inputs ###

        # x + b = y  --> x - y = -b
        # [I -I] [x; y] = -b
        # (nx2n)x(2nx1) = (nx1)
        n = inputVars.shape[0]
        A = np.hstack((np.eye(n), -np.eye(n)));
        x = np.vstack((inputVars, outputVars))
        c = MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x, b= -inputConsts)
        self.constraints.append(c)

    def additionConstraint(self, op):
        """
        Function to generate equations corresponding to addition
        Arguments:
            op: (tf.op) representing add operation
        """
        input_ops = [i.op for i in op.inputs]
        assert len(input_ops) == 2
        input1 = input_ops[0]
        input2 = input_ops[1]
        assert self.isVariable(input1)
        if self.isVariable(input2):
            outputVars = self.getValues(op).reshape(-1, 1)
            input1Vars = self.getValues(input1).reshape(-1, 1)
            input2Vars = self.getValues(input2).reshape(-1, 1)
            assert len(input1Vars) == len(input2Vars)
            assert len(outputVars) == len(input1Vars)
            # x + y = z   -->   [I, I, -I] [x; y; z] = 0
            A = np.hstack((np.eye(len(input1Vars)), np.eye(len(input2Vars)), -np.eye(len(outputVars))))
            x_constraint = np.vstack((input1Vars, input2Vars, outputVars))
            b_constraint = np.zeros((len(outputVars), 1))
            c = MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=x_constraint, b=b_constraint)
            self.constraints.append(c)
        else:
            self.biasAddConstraint(op)

    def reluConstraint(self, op):
        """
        Function to generate constraints corresponding to pointwise Relu
        Arguments:
            op: (tf.op) representing Relu operation
        """

        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        inputValues = np.array(self.getValues(input_ops[0]), dtype='object').flatten()
        outputValues = np.array(self.getValues(op), dtype='object').flatten()
        assert len(inputValues) == len(outputValues)
        ### END getting inputs ###

        ### Generate actual constraint ###
        for inp, outp in zip(inputValues, outputValues):
            c = ReluConstraint(varin=inp, varout=outp)
            self.constraints.append(c)


    def eval_constraints(self, solution_dict = {}):
        """
        Evaluate constraints without tensorflow. Relies upon the fact that all constraints are EQUALITYs
        """
        # var dict: string variables names -> sympy variables
        # sol'n dict: string variable snames -> real values
        vardict = {}
        for i in range(1,self.numVars+1):
            vardict['x'+str(i)] = symbols('x'+str(i))

        for c in self.constraints:
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

    def evaluateWithoutMarabou(self, inputValues):
        """
        Function to evaluate network at a given point using Tensorflow
        Arguments:
            inputValues: list of (np array)s representing inputs to network
        Returns:
            outputValues: (np array) representing output of network
        TODO: untested since port
        """
        print("Evaluating without Marabou")
        inputValuesReshaped = []
        for j in range(len(self.inputOps)):
            inputOp = self.inputOps[j]
            inputShape = self.shapeMap[inputOp]
            inputShape = [i if i is not None else 1 for i in inputShape]        
            # Try to reshape given input to correct shape
            inputValuesReshaped.append(inputValues[j].reshape(inputShape))
        inputNames = [o.name+":0" for o in self.inputOps]
        feed_dict = dict(zip(inputNames, inputValuesReshaped))
        outputName = self.outputOp.name
        out = self.sess.run(outputName + ":0", feed_dict=feed_dict)

        return out

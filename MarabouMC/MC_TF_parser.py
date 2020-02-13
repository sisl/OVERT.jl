# TF parser for model checker interface to marabou
# a class that loads a TF pb and produces a set of constraints that represent the network

# Based on MarabouNetworkTF.py and MarabouNetwork.py

import numpy as np
from tensorflow.python.framework import tensor_util
#from tensorflow.python.framework import graph_util
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from MC_constraints import ConstraintType, Constraint, MatrixConstraint, Relu

import tensorflow as tf

class TFConstraint:
    def __init__(self, filename, inputNames=None, outputName=None):
        """
        Constructs a TFConstraint object from a frozen Tensorflow protobuf

        Args:
            filename: (string) path to the frozen graph .pb file.
            inputNames: [(string)] optional, names of operations corresponding to inputs.
            outputName: (string) optional, names of operation corresponding to output.

        Note: you can always use adds and matmuls to turn multiple output ops into a single output op
        """
        self.lin_constraints = []
        self.relus = []
        self.max = []
        # aux: 
        self.inputVars = []
        self.outputVars = []
        self.inputOps = None
        self.outputOp = None
        self.numVars = 0
        self.varMap = dict() # maps ops -> vars
        self.shapeMap = dict() # maps ?? -> ??
        self.readFromPb(filename, inputNames, outputName)
    
    def getNewVariable(self):
        self.numVars += 1
        # x1, x2, x3, ... etc
        return 'x'+str(self.numVars)

    def clear(self):
        """
        Reset values to represent empty network
        """
        self.lin_constraints = []
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
        
    def readFromPb(self, filename, inputNames, outputName):
        """
        Constructs object from a frozen Tensorflow protobuf 

        Args:
            filename: (string) path to the frozen graph .pb file.
            inputNames: [(string)] optional, name of operation corresponding to input.
            outputNames: [(string)] optional, name of operation corresponding to output.
        """
         ### Read protobuf file and begin session ### after "freezing"
        with tf.gfile.GFile(filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        self.sess = tf.Session(graph=graph)
        ### END reading protobuf ###

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
        if op in list(self.varMap.keys()):
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
        # bookmark 
        elif op.node_def.op == 'Mul':
            input_ops = [i.op for i in op.inputs]
            if self.isVariable(input_ops[0]) or self.isVariable(input_ops[1]): 
                self.mulEquations(op)
            else:
                return
        elif op.node_def.op == 'BiasAdd':
            self.biasAddEquations(op)
        elif op.node_def.op == 'Add':
            self.addEquations(op)
        elif op.node_def.op == 'Sub':
            self.subEquations(op)
        elif op.node_def.op == 'Conv2D':
            self.conv2DEquations(op)
        elif op.node_def.op == 'Relu':
            self.reluEquations(op)
        elif op.node_def.op == 'Maximum':
            self.maxEquations(op)
        elif op.node_def.op == 'MaxPool':
            self.maxpoolEquations(op)
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
        A = inputValues[0]
        B = inputValues[1]
        if aTranspose:
            A = np.transpose(A)
        if bTranspose:
            B = np.transpose(B)
        assert (A.shape[0], B.shape[1]) == outputValues.shape
        assert A.shape[1] == B.shape[0]
        m, n = outputValues.shape
        p = A.shape[1]
        # A \in mxp , B \in  pxn
        ### END getting inputs ###

        ### Generate actual constraints ###
        if convention == "xW":
            x = A
            W = B
            # take transpose of W and store: from xW = b  to  W^T x^T = b^T
            c = MatrixConstraint(ConstraintType('EQUALITY'), A=W.transpose(), x=x.transpose(), b=b.transpose())
            self.lin_constraints.append(c)
        elif convention == "Wx":
            W = A
            x = B
            c = MatrixConstraint(ConstraintType('EQUALITY'), A=W, x=x, b=b)
            self.lin_constraints.append(c)
        else:
            print("Whatchyu doin bro??")
            raise NotImplementedError

    def mulEquations(self, op):
        """
        Function to generate equations corresponding to elementwise matrix multiplication 
        Arguments:
            op: (tf.op) representing elementwise multiplication operation
        """
        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        assert not (self.isVariable(input_ops[0]) and self.isVariable(input_ops[1]))
        if self.isVariable(input_ops[0]):
            #convention = "xW"
            x = prevValues[0]
            W = prevValues[1]
        elif self.isVariable(input_ops[1]):
            #convention = "Wx"
            W = prevValues[0]
            x = prevValues[1]
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
        curValues = curValues.reshape(-1)
        assert x.shape == curValues.shape
        ### END getting inputs ###

        ### Generate actual equations ###
        for i in range(len(curValues)):
            e = []
            e = MarabouUtils.Equation()
            e.addAddend(W[i], x[i])
            e.addAddend(-1, curValues[i])
            e.setScalar(0.0)
            self.addEquation(e)

    def biasAddEquations(self, op):
        """
        Function to generate equations corresponding to bias addition
        Arguments:
            op: (tf.op) representing bias add operation
        """
        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        assert len(input_ops) == 2
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        prevVars = prevValues[0].reshape(-1)
        prevConsts = prevValues[1].reshape(-1)
        # broadcasting
        prevConsts = np.tile(prevConsts, len(prevVars)//len(prevConsts))
        curVars = curValues.reshape(-1)
        assert len(prevVars)==len(curVars) and len(curVars)==len(prevConsts)
        ### END getting inputs ###

        ### Do not generate equations, as these can be eliminated ###
        for i in range(len(prevVars)):
            # prevVars = curVars - prevConst
            self.biasAddRelations += [(prevVars[i], curVars[i], -prevConsts[i])]

    def processBiasAddRelations(self):
        """
        Either add an equation representing a bias add,
        Or eliminate one of the two variables in every other relation
        """
        biasAddUpdates = dict()
        participations = [rel[0] for rel in self.biasAddRelations] + \
                            [rel[1] for rel in self.biasAddRelations]
        for (x, xprime, c) in self.biasAddRelations:
            # x = xprime + c
            # replace x only if it does not occur anywhere else in the system
            if self.lowerBoundExists(x) or self.upperBoundExists(x) or \
                    self.participatesInPLConstraint(x) or \
                    len([p for p in participations if p == x]) > 1:
                e = MarabouUtils.Equation()
                e.addAddend(1.0, x)
                e.addAddend(-1.0, xprime)
                e.setScalar(c)
                self.addEquation(e)
            else:
                biasAddUpdates[x] = (xprime, c)
                self.setLowerBound(x, 0.0)
                self.setUpperBound(x, 0.0)

        for equ in self.equList:
            participating = equ.getParticipatingVariables()
            for x in participating:
                if x in biasAddUpdates: # if a variable to remove is part of this equation
                    xprime, c = biasAddUpdates[x]
                    equ.replaceVariable(x, xprime, c)

    def addEquations(self, op):
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
            curVars = self.getValues(op).reshape(-1)
            prevVars1 = self.getValues(input1).reshape(-1)
            prevVars2 = self.getValues(input2).reshape(-1)
            assert len(prevVars1) == len(prevVars2)
            assert len(curVars) == len(prevVars1)
            for i in range(len(curVars)):
                e = MarabouUtils.Equation()
                e.addAddend(1, prevVars1[i])
                e.addAddend(1, prevVars2[i])
                e.addAddend(-1, curVars[i])
                e.setScalar(0.0)
                self.addEquation(e)
        else:
            self.biasAddEquations(op)

    def subEquations(self, op): 
        """
        Function to generate equations corresponding to subtraction
        Arguments:
            op: (tf.op) representing sub operation
        """
        print("using sub equations")
        input_ops = [i.op for i in op.inputs]
        assert len(input_ops) == 2
        input1 = input_ops[0]
        input2 = input_ops[1]
        assert self.isVariable(input1)
        if self.isVariable(input2):
            curVars = self.getValues(op).reshape(-1)
            prevVars1 = self.getValues(input1).reshape(-1)
            prevVars2 = self.getValues(input2).reshape(-1)
            assert len(prevVars1) == len(prevVars2)
            assert len(curVars) == len(prevVars1)
            for i in range(len(curVars)):
                e = MarabouUtils.Equation()
                e.addAddend(1, prevVars1[i])
                e.addAddend(-1, prevVars2[i])
                e.addAddend(-1, curVars[i])
                e.setScalar(0.0)
                self.addEquation(e)
        else:
            self.biasAddEquations(op)


    def conv2DEquations(self, op):
        """
        Function to generate equations corresponding to 2D convolution operation
        Arguments:
            op: (tf.op) representing conv2D operation
        """

        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        padding = op.node_def.attr['padding'].s.decode()
        strides = list(op.node_def.attr['strides'].list.i)
        prevValues, prevConsts = prevValues[0], prevValues[1]
        _, out_height, out_width, out_channels = curValues.shape
        _, in_height,  in_width,  in_channels  = prevValues.shape
        filter_height, filter_width, filter_channels, num_filters = prevConsts.shape
        assert filter_channels == in_channels
        assert out_channels == num_filters
        # Use padding to determine top and left offsets
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_conv_ops.cc#L51
        if padding=='SAME':
            pad_top  = ((out_height - 1) * strides[1] + filter_height - in_height) // 2
            pad_left = ((out_width - 1) * strides[2] + filter_width - in_width) // 2
        elif padding=='VALID':
            pad_top  = ((out_height - 1) * strides[1] + filter_height - in_height + 1) // 2
            pad_left = ((out_width - 1) * strides[2] + filter_width - in_width + 1) // 2
        else:
            raise NotImplementedError
        ### END getting inputs ###

        ### Generate actual equations ###
        # There is one equation for every output variable
        for i in range(out_height):
            for j in range(out_width):
                for k in range(out_channels): # Out_channel corresponds to filter number
                    e = MarabouUtils.Equation()
                    # The equation convolves the filter with the specified input region
                    # Iterate over the filter
                    for di in range(filter_height):
                        for dj in range(filter_width):
                            for dk in range(filter_channels):

                                h_ind = int(strides[1]*i+di - pad_top)
                                w_ind = int(strides[2]*j+dj - pad_left)
                                if h_ind < in_height and h_ind>=0 and w_ind < in_width and w_ind >=0:
                                    var = prevValues[0][h_ind][w_ind][dk]
                                    c = prevConsts[di][dj][dk][k]
                                    e.addAddend(c, var)

                    # Add output variable
                    e.addAddend(-1, curValues[0][i][j][k])
                    e.setScalar(0.0)
                    self.addEquation(e)

    def reluEquations(self, op):
        """
        Function to generate equations corresponding to pointwise Relu
        Arguments:
            op: (tf.op) representing Relu operation
        """

        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        prev = prevValues[0].reshape(-1)
        cur = curValues.reshape(-1)
        assert len(prev) == len(cur)
        ### END getting inputs ###

        ### Generate actual equations ###
        for i in range(len(prev)):
            self.addRelu(prev[i], cur[i])
        for f in cur:
            self.setLowerBound(f, 0.0)

    def maxpoolEquations(self, op):
        """
        Function to generate maxpooling equations
        Arguments:
            op: (tf.op) representing maxpool operation
        """
        ### Get variables and constants of inputs ###
        input_ops = [i.op for i in op.inputs]
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        validPadding = op.node_def.attr['padding'].s == b'VALID'
        if not validPadding:
            raise NotImplementedError
        prevValues = prevValues[0]
        strides = list(op.node_def.attr['strides'].list.i)
        ksize = list(op.node_def.attr['ksize'].list.i)
        for i in range(curValues.shape[1]):
            for j in range(curValues.shape[2]):
                for k in range(curValues.shape[3]):
                    maxVars = set()
                    for di in range(strides[1]*i, strides[1]*i + ksize[1]):
                        for dj in range(strides[2]*j, strides[2]*j + ksize[2]):
                            if di < prevValues.shape[1] and dj < prevValues.shape[2]:
                                maxVars.add(prevValues[0][di][dj][k])
                    self.addMaxConstraint(maxVars, curValues[0][i][j][k])

    def maxEquations(self, op):
        """
        Function to generate max equations
        # TODO: 
        # extend to handle max between a constant and a variable
        # nominal right now: just handle between two variables
        """
        input_ops = [i.op for i in op.inputs]
        prevValues = [self.getValues(i) for i in input_ops]
        curValues = self.getValues(op)
        assert len(prevValues) == 2
        if not (self.isVariable(input_ops[0]) and self.isVariable(input_ops[1])):
            import pdb; pdb.set_trace()
        # max btw two variables
        prevValues = np.array(prevValues).flatten()
        curValues = np.array(curValues).flatten()
        maxVars = set()
        if (type(prevValues[0]) is np.ndarray) or (type(prevValues[0]) is np.ndarray):
            import pdb; pdb.set_trace()
        maxVars.add(prevValues[0])
        maxVars.add(prevValues[1])
        self.addMaxConstraint(maxVars, curValues[0])


    def evaluateWithoutMarabou(self, inputValues):
        """
        Function to evaluate network at a given point using Tensorflow
        Arguments:
            inputValues: list of (np array)s representing inputs to network
        Returns:
            outputValues: (np array) representing output of network
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

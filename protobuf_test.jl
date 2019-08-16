using NNet


# writeproto(iob, MyType(intval=10, strval="hello world"));   # write an instance of it
file = "protobuf_test.pb"


pb2nnet(file, nnetFile="", inputName=file, outputName="Test2")





def writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileName):
    '''
    Write network data to the .nnet file format
    Args:
        weights (list): Weight matrices in the network order
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''

    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[0]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize

        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize :
                maxLayerSize = len(b)

        # Write data to header
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write("%d," % inputSize )
        for b in biases:
            f2.write("%d," % len(b) )
        f2.write("\n")
        f2.write("0,\n") #Unused Flag

        # Write Min, Max, Mean, and Range of each of the inputs and outputs for normalization
        f2.write(','.join(str(inputMins[i])  for i in range(inputSize)) + ',\n') #Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n') #Maximum Input Values
        f2.write(','.join(str(means[i])      for i in range(inputSize+1)) + ',\n') #Means for normalizations
        f2.write(','.join(str(ranges[i])     for i in range(inputSize+1)) + ',\n') #Ranges for noramlizations

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w,b in zip(weights,biases):
            for j in range(w.shape[1]):
                for i in range(w.shape[0]):
                    f2.write("%.5e," % w[i][j]) #Five digits written. More can be used, but that requires more more space.
                f2.write("\n")

            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i]) #Five digits written. More can be used, but that requires more more space.





def pb2nnet(pbFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    '''
    Write a .nnet file from a frozen Tensorflow protobuf or SavedModel
    Args:
        pbFile (str): If savedModel is false, path to the frozen graph .pb file.
                      If savedModel is true, path to SavedModel folder, which
                      contains .pb file and variables subdirectory.
        inputMins (list): Minimum values for each neural network input.
        inputMaxes (list): Maximum values for each neural network output.
        means (list): Mean value for each input and value for mean of all outputs, used for normalization
        ranges (list): Range value for each input and value for range of all outputs, used for normalization
        inputName (str, optional): Name of operation corresponding to input. Default: ""
        outputName (str, optional) Name of operation corresponding to output. Default: ""
        savedModel (bool, optional) If false, load frozen graph. If true, load SavedModel object. Default: False
        savedModelTags (list, optional) If loading a SavedModel, the user must specify tags used. Default: []
    '''

    if nnetFile=="":
        nnetFile = pbFile[:-2] + 'nnet'

    if savedModel:
        ### Read SavedModel ###
        sess = tf.Session()
        tf.saved_model.loader.load(sess, savedModelTags, pbFile)

        ### Simplify graph using outputName, which must be specified for SavedModel ###
        simp_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),[outputName])
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(simp_graph_def, name="")
        sess = tf.Session(graph=graph)
        ### End reading SavedModel

    else:
        ### Read protobuf file and begin session ###
        with tf.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
        ### END reading protobuf ###

    ### Find operations corresponding to input and output ###
    if inputName:
        inputOp = sess.graph.get_operation_by_name(inputName)
    else: # If there is just one placeholder, use it as input
        ops = sess.graph.get_operations()
        placeholders = [x for x in ops if x.node_def.op == 'Placeholder']
        assert len(placeholders)==1
        inputOp = placeholders[0]
    if outputName:
        outputOp = sess.graph.get_operation_by_name(outputName)
    else: # Assume that the last operation is the output
        outputOp = sess.graph.get_operations()[-1]

    # Recursively search for weights and bias parameters and add them to list
    # Search until the inputOp is found
    # If inputOp is not found, than the operation does not exist in the graph or does not lead to the output operation
    weights = []
    biases = []
    foundInputFlag = False
    foundInputFlag = processGraph(outputOp, inputOp, foundInputFlag, weights, biases)
    if foundInputFlag:

        # Default values for input bounds and normalization constants
        if inputMins is None: inputMins = inputSize*[np.finfo(np.float32).min]
        if inputMaxes is None: inputMaxes = inputSize*[np.finfo(np.float32).max]
        if means is None: means = (inputSize+1)*[0.0]
        if ranges is None: ranges = (inputSize+1)*[1.0]

        # Write NNet file
        writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,nnetFile)
    else:
        print("Could not find the given input in graph: %s"%inputOp.name)

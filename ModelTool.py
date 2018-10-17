import numpy as np
import tensorflow as tf
import madmom
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
import tensorflow as tf
from tensorflow.contrib import rnn
import util
import os

def MadmomLSTMLayerParam(layer):
    input_gate = layer.input_gate
    cell = layer.cell
    forget_gate = layer.forget_gate
    output_gate = layer.output_gate

    input_dim = np.shape(input_gate.weights)[0]
    num_units = np.shape(input_gate.weights)[1]

    weights = np.concatenate((input_gate.weights, cell.weights, forget_gate.weights, output_gate.weights), 1)
    recurrent_weights = np.concatenate((input_gate.recurrent_weights, cell.recurrent_weights, forget_gate.recurrent_weights, output_gate.recurrent_weights), 1)
    kernel = np.concatenate((weights, recurrent_weights))

    bias = np.concatenate((input_gate.bias, cell.bias, forget_gate.bias, output_gate.bias))

    w_i_diag = input_gate.peephole_weights
    w_f_diag = forget_gate.peephole_weights
    w_o_diag = output_gate.peephole_weights

    return kernel, bias, w_i_diag, w_f_diag, w_o_diag, input_dim, num_units

def MadmomLSTMLayersParam(layerArr):

    # not completed

    layer = layerArr[0]
    input_gate = layer.input_gate
    cell = layer.cell
    forget_gate = layer.forget_gate
    output_gate = layer.output_gate

    input_gate_weights = input_gate.weights
    cell_weights = cell.weights
    forget_gate_weights = forget_gate.weights
    output_gate_weights = output_gate.weights

    # input_gate_recurrent_weights = input_gate.recurrent_weights
    # cell_recurrent_weights = cell.recurrent_weights
    # forget_gate_recurrent_weights = forget_gate.recurrent_weights
    # output_gate_recurrent_weights = output_gate.recurrent_weights
    input_gate_recurrent_weights = [input_gate.recurrent_weights]
    cell_recurrent_weights = [cell.recurrent_weights]
    forget_gate_recurrent_weights = [forget_gate.recurrent_weights]
    output_gate_recurrent_weights = [output_gate.recurrent_weights]

    input_gate_bias = input_gate.bias
    cell_bias = cell.bias
    forget_bias = forget_gate.bias
    output_gate_bias = output_gate.bias

    w_i_diag = input_gate.peephole_weights
    w_f_diag = forget_gate.peephole_weights
    w_o_diag = output_gate.peephole_weights

    for i in range(len(layerArr)):
        layer = layerArr[i]
        input_gate = layer.input_gate
        cell = layer.cell
        forget_gate = layer.forget_gate
        output_gate = layer.output_gate

        input_dim = np.shape(input_gate.weights)[0]
        num_units = np.shape(input_gate.weights)[1]

        input_gate_weights = np.concatenate((input_gate_weights, input_gate.weights), 1)
        cell_weights = np.concatenate((cell_weights, cell.weights), 1)
        forget_gate_weights = np.concatenate((forget_gate_weights, forget_gate.weights), 1)
        output_gate_weights = np.concatenate((output_gate_weights, output_gate.weights), 1)

        # input_gate_recurrent_weights = np.concatenate((input_gate_recurrent_weights, input_gate.recurrent_weights))
        # cell_recurrent_weights = np.concatenate((cell_recurrent_weights, cell.recurrent_weights))
        # forget_gate_recurrent_weights = np.concatenate((forget_gate_recurrent_weights, forget_gate.recurrent_weights))
        # output_gate_recurrent_weights = np.concatenate((output_gate_recurrent_weights, output_gate.recurrent_weights))
        input_gate_recurrent_weights = input_gate_recurrent_weights.append(input_gate.recurrent_weights)
        cell_recurrent_weights = input_gate_recurrent_weights.append(cell_recurrent_weights, cell.recurrent_weights)
        forget_gate_recurrent_weights = input_gate_recurrent_weights.append(forget_gate_recurrent_weights, forget_gate.recurrent_weights)
        output_gate_recurrent_weights = input_gate_recurrent_weights.append(output_gate_recurrent_weights, output_gate.recurrent_weights)

        input_gate_bias = np.concatenate((input_gate_bias, input_gate.bias))
        cell_bias = np.concatenate((cell_bias, cell.bias))
        forget_gate_bias = np.concatenate((forget_bias, forget_gate.bias))
        output_gate_bias = np.concatenate((output_gate_bias, output_gate_bias.bias))

        if w_i_diag is not None:
            w_i_diag = np.concatenate((w_i_diag, input_gate.peephole_weights))
            w_f_diag = np.concatenate((w_f_diag, forget_gate.peephole_weights))
            w_o_diag = np.concatenate((w_o_diag, output_gate.peephole_weights))

    def Eye(arr):
        size = len(arr[0])
        blockCount = len(arr)
        zeroData = np.zeros_like(arr[0])
        rowArr = []
        for i in range(blockCount):
            temp = [zeroData] * blockCount
            temp[i] = arr[i]
            rowArr.append(np.concatenate(temp, 1))

        return np.concatenate(rowArr)

    input_gate_recurrent_weights = Eye(input_gate_recurrent_weights)
    cell_recurrent_weights = Eye(cell_recurrent_weights)
    forget_gate_recurrent_weights = Eye(forget_gate_recurrent_weights)
    output_gate_recurrent_weights = Eye(output_gate_recurrent_weights)

    weights = np.concatenate((input_gate_weights, cell_weights, forget_gate_weights, output_gate_weights), 1)
    recurrent_weights = np.concatenate((input_gate_recurrent_weights, cell_recurrent_weights, forget_gate_recurrent_weights, output_gate_recurrent_weights), 1)
    kernel = np.concatenate((weights, recurrent_weights))

    bias = np.concatenate((input_gate_bias, cell_bias, forget_gate_bias, output_gate_bias))

    return kernel, bias, w_i_diag, w_f_diag, w_o_diag, input_dim, num_units

def MadmomBLSTMLayerParam(layer):
    return [MadmomLSTMLayerParam(layer.fwd_layer), MadmomLSTMLayerParam(layer.bwd_layer)]

def MadmomFeedForwardLayerParam(layer):
    return layer.weights, layer.bias, layer.activation_fn

def MadmomActivationFuncToTensorFlow(func):

    if func.__name__ == 'tanh':
        fn = tf.tanh
    elif func.__name__ == 'sigmoid':
        fn = tf.sigmoid
    elif func.__name__ == 'softmax':
        fn = tf.nn.softmax
    else:
        assert False

    return fn

def FindVarByName(varList, name, appendZero = True):
    searchName = name
    if appendZero:
        searchName = name + ':0'

    for variable in varList:
        if variable.name == searchName:
            print('search', searchName, '   succeed')
            return variable

    print('search', searchName, '   failed')
    return None

def BuildDownbeatsModelGraph(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, weightsShape, biasShape, tfActivationFunc, useLSTMBlockFusedCell):
    print('variableScopeName', variableScopeName, 'numLayers', numLayers, 'batchSize', batchSize, 'maxTime', maxTime, 'numUnits', numUnits)
    print('usePeepholes', usePeepholes, 'weightsShape', weightsShape, 'biasShape', biasShape)
    
    buildFunc = BuildDownbeatsModelGraphWithLSTMCell
    if useLSTMBlockFusedCell:
        buildFunc = BuildDownbeatsModelGraphWithLSTMBlockFusedCell
    
    return buildFunc(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, weightsShape, biasShape, tfActivationFunc)

def BuildDownbeatsModelGraphWithLSTMCell(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, weightsShape, biasShape, tfActivationFunc):
    print('BuildDownbeatsModelGraph With LSTMCell')
    with tf.variable_scope(variableScopeName):
        cells = []
        # madmom的源码里计算gate activation时，没有用forget_bias，转过来需要设置成0
        forgetBias = 0.0
        for i in range(numLayers * 2):
            cell = rnn.LSTMCell(numUnits, use_peepholes=usePeepholes, forget_bias=forgetBias)
            cells.append(cell)

        XShape = (batchSize, maxTime, inputDim)
        X = tf.placeholder(dtype=tf.float32, shape=XShape, name='X')
        sequenceLength = tf.placeholder(tf.int32, [None], name='sequence_length')
        outputs, statesFW, statesBW = rnn.stack_bidirectional_dynamic_rnn(
            cells[0:numLayers], cells[numLayers:], 
            X, sequence_length=sequenceLength, dtype=tf.float32)
        
        outlayerDim = tf.shape(outputs)[2]
        outputs = tf.reshape(outputs, [batchSize * maxTime, outlayerDim])

        weights = tf.Variable(tf.random_normal(weightsShape, dtype=tf.float32), name='output_linear_w')
        bias = tf.Variable(tf.random_normal(biasShape, dtype=tf.float32), name='output_linear_b')
        logits = tf.matmul(outputs, weights) + bias
        logits = tfActivationFunc(logits)

    tensorDic = {}
    tensorDic['X'] = X
    tensorDic['sequence_length'] = sequenceLength
    tensorDic['output'] = logits
    return tensorDic

def CreateLSTMBlockFusedCell(layerIdx, isForward, numUnits, usePeepholes, forgetBias):
    cellName = TFLSTMVariableName(layerIdx, isForward, "")
    cellName = cellName[0 : len(cellName)-1]
    cell = rnn.LSTMBlockFusedCell(numUnits, use_peephole=usePeepholes, forget_bias=forgetBias, name=cellName)
    return cell

def BuildDownbeatsModelGraphWithLSTMBlockFusedCell(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, weightsShape, biasShape, tfActivationFunc):    
    print('BuildDownbeatsModelGraph With LSTMBlockFusedCell')
    with tf.variable_scope(variableScopeName):
        cells = []
        tempInputFirst = tf.zeros([maxTime, batchSize, inputDim])
        inputShapeFirst = tempInputFirst.get_shape().with_rank(3)
        tempInputOther = tf.zeros([maxTime, batchSize, numUnits * 2])
        inputShapeOther = tempInputOther.get_shape().with_rank(3)
        # madmom的源码里计算gate activation时，没有用forget_bias，转过来需要设置成0
        forgetBias = 0.0
        for i in range(numLayers):
            fwCell = CreateLSTMBlockFusedCell(i, True, numUnits, usePeepholes, forgetBias)
            bwCell = CreateLSTMBlockFusedCell(i, False, numUnits, usePeepholes, forgetBias)
            if i == 0:
                fwCell.build(inputShapeFirst)
                bwCell.build(inputShapeFirst)
            else:
                fwCell.build(inputShapeOther)
                bwCell.build(inputShapeOther)
            cells.append((fwCell, bwCell))

        XShape = (maxTime, batchSize, inputDim)
        X = tf.placeholder(dtype=tf.float32, shape=XShape, name='X')
        sequenceLength = tf.placeholder(tf.int32, [None], name='sequence_length')

        def reverseOp(inputOp):
            return tf.reverse(inputOp, [0])

        reverseDim = [0]
        currentFWInput = X
        currentBWInput = reverseOp(currentFWInput)
        fwOutputs = None
        bwOutputs = None
        for fwCell, bwCell in cells:
            fwOutputs, _ = fwCell(currentFWInput, dtype=tf.float32, sequence_length=sequenceLength)
            bwOutputs, _ = bwCell(currentBWInput, dtype=tf.float32, sequence_length=sequenceLength)
            bwOutputs = reverseOp(bwOutputs)
            currentFWInput = tf.concat((fwOutputs, bwOutputs), 2)
            currentBWInput = reverseOp(currentFWInput)

        outputs = currentFWInput
        
        outlayerDim = tf.shape(outputs)[2]
        outputs = tf.reshape(outputs, [batchSize * maxTime, outlayerDim])

        weights = tf.Variable(tf.random_normal(weightsShape, dtype=tf.float32), name='output_linear_w')
        bias = tf.Variable(tf.random_normal(biasShape, dtype=tf.float32), name='output_linear_b')
        logits = tf.matmul(outputs, weights) + bias
        logits = tfActivationFunc(logits)

    tensorDic = {}
    tensorDic['X'] = X
    tensorDic['sequence_length'] = sequenceLength
    tensorDic['output'] = logits
    return tensorDic

def MadmomDownbeatsModelToTensorflow(madmomModel, variableScopeName, outputFilePath):
    madmomBLSTMParamArr = []
    madmomFeedForwardLayerParam = None
    for layer in madmomModel.layers:
        layerType = type(layer)
        if layerType == madmom.ml.nn.layers.BidirectionalLayer:
            madmomBLSTMParamArr.append(MadmomBLSTMLayerParam(layer))
        elif layerType == madmom.ml.nn.layers.FeedForwardLayer:
            madmomFeedForwardLayerParam = MadmomFeedForwardLayerParam(layer)
        else:
            print('not support type', layerType)
            return False


    numLayers = len(madmomBLSTMParamArr)
    firstFWLayer = madmomBLSTMParamArr[0][0]
    inputDim =  firstFWLayer[5]
    numUnits =  firstFWLayer[6]
    usePeepholes = firstFWLayer[2] is not None

    madmomWeights, madmomBias, madmomActivationFunc = madmomFeedForwardLayerParam
    
    tfActivationFunc = MadmomActivationFuncToTensorFlow(madmomActivationFunc)
    batchSize = 1
    maxTime = 128

    graph = tf.Graph()
    with graph.as_default():
        BuildDownbeatsModelGraph(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, np.shape(madmomWeights), np.shape(madmomBias), tfActivationFunc, False)

        with tf.Session() as sess:
            varList = tf.trainable_variables()

            for blstmLayerIdx in range(numLayers):
                blstmParam = madmomBLSTMParamArr[blstmLayerIdx]

                fwKernel = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, True, 'kernel', variableScopeName + '/'))
                fwBias = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, True, 'bias', variableScopeName + '/'))
                fwWIDiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, True, 'w_i_diag', variableScopeName + '/'))
                fwWFDiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, True, 'w_f_diag', variableScopeName + '/'))
                fwWODiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, True, 'w_o_diag', variableScopeName + '/'))

                bwKernel = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, False, 'kernel', variableScopeName + '/'))
                bwBias = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, False, 'bias', variableScopeName + '/'))
                bwWIDiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, False, 'w_i_diag', variableScopeName + '/'))
                bwWFDiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, False, 'w_f_diag', variableScopeName + '/'))
                bwWODiag = FindVarByName(varList, TFLSTMVariableName(blstmLayerIdx, False, 'w_o_diag', variableScopeName + '/'))

                sess.run([
                    fwKernel.assign(blstmParam[0][0]),
                    fwBias.assign(blstmParam[0][1]),
                    fwWIDiag.assign(blstmParam[0][2]),
                    fwWFDiag.assign(blstmParam[0][3]),
                    fwWODiag.assign(blstmParam[0][4]),

                    bwKernel.assign(blstmParam[1][0]),
                    bwBias.assign(blstmParam[1][1]),
                    bwWIDiag.assign(blstmParam[1][2]),
                    bwWFDiag.assign(blstmParam[1][3]),
                    bwWODiag.assign(blstmParam[1][4])
                ])
                print('assign session run for blstm idx', blstmLayerIdx)

            weights = FindVarByName(varList, variableScopeName + '/' + 'output_linear_w')
            bias = FindVarByName(varList, variableScopeName + '/' + 'output_linear_b')

            sess.run([
                weights.assign(madmomWeights),
                bias.assign(madmomBias)
            ])
            print('assign session run for weights bias')
            
            saver = tf.train.Saver()
            saver.save(sess, outputFilePath)
            print('model save to', outputFilePath)

    print('numBLSTMLayers', numLayers, 'inputDim', inputDim, 'numUnits', numUnits, 'usePeepholes', usePeepholes)
    print('feedfwdlayer info', np.shape(madmomWeights), np.shape(madmomBias), madmomActivationFunc)

    return True

def ConvertMadmomDownbeatsModelToTensorflow():  
    nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM)
    rootDir = util.getRootDir()
    for i in range(len(nn.processors[0].processors)):
        madmomModel = nn.processors[0].processors[i]
        varScopeName = 'downbeats_' + str(i)
        outputFilePath = GenerateOutputModelPath(varScopeName, mkdirIfNotExists=True)
        MadmomDownbeatsModelToTensorflow(madmomModel, varScopeName, outputFilePath)
    return True

def TFLSTMVariableName(layerIdx, isForward, paramName, prefix = None):
    layerDir = 'bw'
    if isForward:
        layerDir = 'fw'
    name = 'stack_bidirectional_rnn/cell_%d/bidirectional_rnn/%s/lstm_cell/%s' % (layerIdx, layerDir, paramName)
    if prefix is not None:
        name = prefix + name
    return name

def TFVar(shape=None, initialValue=None, dtype=tf.float32, name=None):
    if initialValue is not None:
        return tf.Variable(initial_value=initialValue, dtype=dtype, name=name)
    return tf.Variable(tf.random_normal(shape, dtype=dtype), dtype=dtype, name=name)

def BatchNormalLayer(inputData, name, meanValue=None, invStdValue=None, meanShape=None, invStdShape=None, gamma=1.0, beta=0):
    # (data - self.mean) * (self.gamma * self.inv_std) + self.beta
    # gamma = 1.0
    # beta = 0
    mean = TFVar(shape=meanShape, initialValue=meanValue, name=name+'/mean')
    invStd = TFVar(shape=invStdShape, initialValue=invStdValue, name=name+'/inv_std')
    outputData = inputData - mean
    if gamma != 1.0:
        outputData = outputData * (gamma * invStd)
    else:
        print('gamma == 1.0 skip')
        outputData = outputData * invStd
    if beta != 0.0:
        outputData = outputData + beta
    else:
        print('beta == 0.0 skip')
    return outputData

def ConvLayer(inputData, name, kernelValue=None, biasValue=None, kernelShape=None, biasShape=None, activationFunc=tf.tanh):
    kernel = TFVar(shape=kernelShape, initialValue=kernelValue, name=name+'/kernel')
    outputData = tf.nn.conv2d(inputData, kernel, [1, 1, 1, 1], 'VALID')
    bias = TFVar(shape=biasShape, initialValue=biasValue, name=name+'/bias')
    outputData = outputData + bias
    outputData = activationFunc(outputData)
    return outputData

def StrideLayer(inputData, blockSize):
    # 这个strideLayer没找到合适的实现，其实是一个重叠操作，先用frame实现
    from tensorflow.contrib import signal
    inputDataShape = tf.shape(inputData)    
    frameStep = inputDataShape[2] * inputDataShape[3]
    frameLength = frameStep * blockSize
    outputData = tf.reshape(inputData, [-1])
    outputData = signal.frame(outputData, frameLength, frameStep)
    return outputData

def FeedForwardLayer(inputData, name, weightValue=None, biasValue=None, weightShape=None, biasShape=None, activationFunc=tf.sigmoid):
    weight = TFVar(shape=weightShape, initialValue=weightValue, name=name+'/weight')
    bias = TFVar(shape=biasShape, initialValue=biasValue, name=name+'/bias')
    outputData = tf.matmul(inputData, weight) + bias
    outputData = activationFunc(outputData)
    return outputData

def GenerateName(dic, key):
    val = 0
    if key in dic:
        val = dic[key]
        val = val + 1
    dic[key] = val
    return key + '_' + str(val)

def BuildOnsetModelGraph(variableScopeName):
    nameDic = {}
    width = 80
    channel = 3
    tensorDic = {}
    with tf.variable_scope(variableScopeName):
        X = tf.placeholder(tf.float32, shape=[None, width, channel], name='X')

        inputData = tf.reshape(X, [1, -1, width, channel])

        outputData = BatchNormalLayer(inputData, GenerateName(nameDic, 'batch_normal'), meanShape=[width, channel], invStdShape=[width, channel], gamma=1.0, beta=0)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData
        
        outputData = ConvLayer(outputData, GenerateName(nameDic, 'conv'), kernelShape=[7, 3, 3, 10], biasShape=[10], activationFunc=tf.tanh)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData
        
        outputData = tf.nn.max_pool(outputData, [1, 1, 3, 1], [1, 1, 3, 1], 'SAME')
        tensorDic[GenerateName(nameDic, 'layer')] = outputData

        outputData = ConvLayer(outputData, GenerateName(nameDic, 'conv'), kernelShape=[3, 3, 10, 20], biasShape=[20], activationFunc=tf.tanh)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData
        
        outputData = tf.nn.max_pool(outputData, [1, 1, 3, 1], [1, 1, 3, 1], 'SAME')
        tensorDic[GenerateName(nameDic, 'layer')] = outputData

        outputData = StrideLayer(outputData, 7)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData

        outputData = FeedForwardLayer(outputData, GenerateName(nameDic, 'ff'), weightShape=[1120, 256], biasShape=[256], activationFunc=tf.sigmoid)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData

        outputData = FeedForwardLayer(outputData, GenerateName(nameDic, 'ff'), weightShape=[256, 1], biasShape=[1], activationFunc=tf.sigmoid)
        tensorDic[GenerateName(nameDic, 'layer')] = outputData

        output = tf.reshape(outputData, [-1], name='output')
    
    tensorDic['X'] = X
    tensorDic['output'] = output
    return tensorDic

def GenerateOutputModelPath(variableScopeName, mkdirIfNotExists = False):
    rootDir = util.getRootDir()
    outputFileDir = os.path.join(rootDir, 'madmom_to_tf')
    if mkdirIfNotExists:
        if not os.path.exists(outputFileDir):
            os.path.mkdir(outputFileDir)
    
    outputFileDir = os.path.join(outputFileDir, variableScopeName)
    if mkdirIfNotExists:
        if not os.path.exists(outputFileDir):
            os.mkdir(outputFileDir)

    outputFilePath = os.path.join(outputFileDir, variableScopeName+'.ckpt')
    return outputFilePath

def ConvertMadmomKernelToTensorFlow(src):
    kernel = np.transpose(src, (2, 3, 0, 1))
    kernelShape = np.shape(kernel)
    # 测试顺序是反的
    kernel = np.reshape(kernel, [-1, kernelShape[2], kernelShape[3]])
    kernel = kernel[::-1]
    kernel = np.reshape(kernel, kernelShape)
    return kernel

def ConvertMadmomOnsetModelToTensorflow(madmomModel=None):
    from madmom.ml.nn import NeuralNetwork
    from madmom.models import ONSETS_CNN
    if madmomModel is None:
        nn = NeuralNetwork.load(ONSETS_CNN[0])
    else:
        nn = madmomModel

    rootDir = util.getRootDir()
    variableScopeName = 'onset'
    outputFilePath = GenerateOutputModelPath(variableScopeName, mkdirIfNotExists=True)

    nameDic = {}

    width = 80
    channel = 3
    
    with tf.variable_scope(variableScopeName):
        with tf.Session() as sess:
            X = tf.placeholder(tf.float32, shape=[None, width, channel], name='X')
            outputData = tf.reshape(X, [1, -1, width, channel])
            for layer in nn.layers:
                layerType = type(layer) 
                if layerType == madmom.ml.nn.layers.BatchNormLayer:
                    outputData = BatchNormalLayer(outputData, GenerateName(nameDic, 'batch_normal'), 
                        meanValue=layer.mean, invStdValue=layer.inv_std, gamma=layer.gamma, beta=layer.beta)

                elif layerType == madmom.ml.nn.layers.ConvolutionalLayer:
                    activationFunc = MadmomActivationFuncToTensorFlow(layer.activation_fn)
                    kernel = ConvertMadmomKernelToTensorFlow(layer.weights)      
                    outputData = ConvLayer(outputData, GenerateName(nameDic, 'conv'), 
                        kernelValue=kernel, biasValue=layer.bias, activationFunc=activationFunc)

                elif layerType == madmom.ml.nn.layers.MaxPoolLayer:
                    ksize = layer.size
                    stride = layer.stride
                    outputData = tf.nn.max_pool(outputData, [1, ksize[0], ksize[1], 1], [1, stride[0], stride[1], 1], 'SAME')

                elif layerType == madmom.ml.nn.layers.StrideLayer:
                    outputData = StrideLayer(outputData, layer.block_size)

                elif layerType == madmom.ml.nn.layers.FeedForwardLayer:
                    activationFunc = MadmomActivationFuncToTensorFlow(layer.activation_fn)
                    outputData = FeedForwardLayer(outputData, GenerateName(nameDic, 'ff'), 
                        weightValue=layer.weights, biasValue=layer.bias, activationFunc=activationFunc)

                else:
                    print('error. not support layer type')
                    return False

            output = tf.reshape(outputData, [-1], name='output')

            sess.run([tf.global_variables_initializer()])

            saver = tf.train.Saver()
            saver.save(sess, outputFilePath)
            print('model save to', outputFilePath)

    return True

if __name__ == '__main__':
    # ConvertMadmomOnsetModelToTensorflow()

    # ConvertMadmomDownbeatsModelToTensorflow()
    print('model tool end')




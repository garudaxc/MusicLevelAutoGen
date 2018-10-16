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

def GetActiviationFn(func):

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
    
    tfActivationFunc = GetActiviationFn(madmomActivationFunc)
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
        outputFileDir = os.path.join(rootDir, 'madmom_to_tf')
        outputFileDir = os.path.join(outputFileDir, varScopeName)
        if not os.path.exists(outputFileDir):
            os.mkdir(outputFileDir)

        outputFilePath = os.path.join(outputFileDir, varScopeName + '.ckpt')
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


if __name__ == '__main__':
    ConvertMadmomDownbeatsModelToTensorflow()
    print('model tool end')




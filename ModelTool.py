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

def BuildDownbeatsModelGraph(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, weightsShape, biasShape, tfActivationFunc):
    with tf.variable_scope(variableScopeName):
        cells = []
        for i in range(numLayers * 2):
            cell = rnn.LSTMCell(numUnits, use_peepholes=usePeepholes)
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
        BuildDownbeatsModelGraph(variableScopeName, numLayers, batchSize, maxTime, numUnits, inputDim, usePeepholes, np.shape(madmomWeights), np.shape(madmomBias), tfActivationFunc)

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




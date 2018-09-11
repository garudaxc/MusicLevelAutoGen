import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python import pywrap_tensorflow

def TFVariableToShapeMap(ckptPath):
    reader = pywrap_tensorflow.NewCheckpointReader(ckptPath)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print('=======')
    for key in var_to_shape_map:
        print(key, var_to_shape_map[key])
    print('=======')
    return var_to_shape_map

def TFCurrentVariableList():
    nameList = [v.name for v in tf.trainable_variables()]
    print('=======')
    for name in nameList:
        print(name)
    print('=======')
    return nameList

def TFVariable(shape, dtype=tf.float32, name=None):
    # return tf.get_variable(name=name, initializer=tf.random_normal(shape, dtype=dtype))
    return tf.Variable(tf.random_normal(shape, dtype=dtype), name=name)

def Conv2dLayer(inputData, filterShape, name=None):
    filterTensor = TFVariable(shape=filterShape)
    convLayer = tf.nn.conv2d(inputData, filterTensor, [1, 1, 1, 1], "SAME", name=name)
    reluLayer = tf.nn.relu(convLayer)
    return reluLayer

def LinearLayer(inputData, dimA, dimB, nameWeight=None, nameBias=None):
    weight = TFVariable(shape=(dimA, dimB), name=nameWeight)
    bias = TFVariable(shape=[dimB], name=nameBias)
    linearLayer = tf.matmul(inputData, weight) + bias
    return linearLayer

def BuildCNN(inputData, batchSize, height, width, channel):
    featureNum = 16
    tempData = tf.reshape(inputData, shape=(batchSize, height, width, channel))
    tempData = Conv2dLayer(tempData, (3, 3, channel, featureNum), 'cnn0')
    tempData = Conv2dLayer(tempData, (3, 3, featureNum, featureNum), 'cnn1')
    cnnOutputData = tf.reshape(tempData, (batchSize * height, width * featureNum))
    linearOutputData = LinearLayer(cnnOutputData, width * featureNum, featureNum)
    linearOutputData = tf.reshape(linearOutputData, (batchSize, height, featureNum))
    return linearOutputData

def BuildModel(batchSize, maxTime, xDim, yDim, tensorDic):
    xData = tf.placeholder(tf.float32, shape=(batchSize, maxTime, xDim), name='xData')
    yData = tf.placeholder(tf.float32, shape=(batchSize, maxTime, yDim), name='yData')
    seqLen = tf.placeholder(tf.int32, shape=(batchSize), name='seqLen')
    learningRate = tf.placeholder(tf.float32, name='learningRate')

    cnnOutputData = BuildCNN(xData, batchSize, maxTime, xDim, 1)

class NoteDetectionModel():
    def __init__(self, batchSize, maxTime, numLayers, numUnits, xDim, yDim, timeMajor=False, useCudnn=False, useInitialStates=True):
        self.batchSize = batchSize
        self.maxTime = maxTime
        self.numLayers = numLayers
        self.numUnits = numUnits
        self.xDim = xDim
        self.yDim = yDim
        self.timeMajor = timeMajor
        self.useCudnn = useCudnn
        self.useInitialStates = useInitialStates
        self.tensorDic = {}
        print('batchSize', batchSize, 'maxTime', maxTime, 'numLayers', numLayers, 'numUnits', numUnits)
        print('xDim', xDim, 'yDim', yDim, 'timeMajor', timeMajor, 'useCudnn', useCudnn, 'useInitialStates', useInitialStates)
        self.cudnnLSTMName = 'note_cudnn_lstm'

    def ModelInfo(self):
        return self.batchSize, self.maxTime, self.numLayers, self.numUnits, self.xDim, self.yDim, self.timeMajor, self.useCudnn, self.useInitialStates

    def GetTensorDic(self):
        return self.tensorDic

    def ClassWeight(self, Y):
        # weight for each class, balance the sample
        classWeight = tf.reduce_sum(Y, axis=0) / tf.cast(tf.shape(Y)[0], tf.float32)
        classWeight = tf.maximum(classWeight, tf.ones_like(classWeight) * 0.001)
        classWeight = (1.0 / self.yDim) / classWeight
        classWeight = tf.reduce_sum(classWeight * Y, axis=1)
        return classWeight

    def LossOp(self, logits, Y):
        classWeight = self.ClassWeight(Y)
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        loss = tf.reduce_mean(crossEntropy * classWeight)
        return loss

    def TrainOp(self, loss, learningRate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)    
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        train = optimizer.minimize(loss)
        return train

    def AccuracyOp(self, logits, Y):
        correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def ClassifyInfoTensor(self, logits, Y):
        # classify debug info tensor
        batchSize = self.batchSize
        maxTime = self.maxTime
        classifyInfoArr = []
        for idx in range(self.yDim):
            tempYIdxArr = [idx] * (batchSize * maxTime)
            tempYIdxArr = tf.constant(tempYIdxArr, dtype=tf.int64, shape=[batchSize * maxTime])
            labelY = tf.equal(tempYIdxArr, tf.argmax(Y, axis=1))
            labelY = tf.cast(labelY, tf.int32)
            labelYCount = tf.reduce_sum(labelY)
            labelYCount = tf.reshape(labelYCount, [1])

            predictY = tf.equal(tempYIdxArr, tf.argmax(logits, axis=1))
            predictY = tf.cast(predictY, tf.int32)
            predictYCount = tf.reduce_sum(predictY)
            predictYCount = tf.reshape(predictYCount, [1])

            predictYTrueCount = labelY * predictY
            predictYTrueCount = tf.reduce_sum(predictYTrueCount)
            predictYTrueCount = tf.reshape(predictYTrueCount, [1])

            predictYFalseCount = predictYCount - predictYTrueCount
            notPredictCount = labelYCount - predictYTrueCount

            classifyInfo = tf.concat([labelYCount, predictYCount, predictYTrueCount, predictYFalseCount, notPredictCount], 0)
            classifyInfoArr.append([classifyInfo])

        classifyInfo = tf.concat(classifyInfoArr, 0)
        return classifyInfo

    def XAndSequenceLength(self):
        XShape = (self.batchSize, self.maxTime, self.xDim)
        if self.timeMajor:
            XShape = (self.maxTime, self.batchSize, self.xDim)

        X = tf.placeholder(dtype=tf.float32, shape=XShape, name='X')
        sequenceLength = tf.placeholder(tf.int32, [None], name='sequence_length')
        return X, sequenceLength

    def LSTMInputStates(self):
        numLayers = self.numLayers
        batchSize = self.batchSize
        numUnits = self.numUnits
        statesCount = numLayers * 2 * 2
        initialStates = tf.placeholder(dtype=tf.float32, shape=(batchSize * statesCount, numUnits), name='initial_states')
        initialStatesFW = None
        initialStatesBW = None
        if self.useInitialStates:
            initialStatesReshape = tf.reshape(initialStates, [statesCount, batchSize, numUnits])
            partitions = []
            for i in range(statesCount):
                partitions.append(i)
            allInputStates = tf.dynamic_partition(initialStatesReshape, partitions, statesCount)
            for i in range(len(allInputStates)):
                allInputStates[i] = tf.reshape(allInputStates[i], shape=[batchSize, numUnits])
            initialStatesFW = []
            initialStatesBW = []
            for i in range(numLayers):
                initialStatesFW.append(tf.nn.rnn_cell.LSTMStateTuple(allInputStates[i * 2], allInputStates[i * 2 + 1]))
                initialStatesBW.append(tf.nn.rnn_cell.LSTMStateTuple(allInputStates[i * 2 + numLayers * 2], allInputStates[i * 2 + 1 + numLayers * 2]))

        return initialStates, initialStatesFW, initialStatesBW

    def LSTMOuputStates(self, statesFW, statesBW):
        outputStatesArr = []
        for states in statesFW:
            for state in states:
                outputStatesArr.append(state)
        for states in statesBW:
            for state in states:
                outputStatesArr.append(state)

        outputStates = tf.concat(outputStatesArr, 0, name="output_states")
        return outputStates

    def LSTM(self, X, sequenceLength):
        numLayers = self.numLayers
        state_size = tf.nn.rnn_cell.LSTMStateTuple(0, 0)
        cells = []
        dropoutCells = []
        for i in range(numLayers * 2):
            cell = rnn.LSTMCell(self.numUnits, use_peepholes=True, forget_bias=1.0)
            cells.append(cell)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1.0 - self.dropout))
            dropoutCells.append(cell)

            state_size = cell.state_size

        if self.timeMajor:
            X = tf.transpose(X, perm=[1, 0, 2])

        initialStates, initialStatesFW, initialStatesBW = self.LSTMInputStates()

        outputs, statesFW, statesBW = rnn.stack_bidirectional_dynamic_rnn(
            dropoutCells[0:numLayers], dropoutCells[numLayers:], 
            X, sequence_length=sequenceLength, dtype=tf.float32,
            initial_states_fw=initialStatesFW)

        if self.timeMajor:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        outputStates = self.LSTMOuputStates(statesFW, statesBW)

        return outputs, initialStates, outputStates

    def CudnnLSTMInputStates(self, cudnnLSTM):
        stateShape = cudnnLSTM.state_shape(self.batchSize)
        initialStatesShape = [stateShape[0][0] * 2, stateShape[0][1], stateShape[0][2]]
        countPerState = stateShape[0][0]
        initialStates = tf.placeholder(dtype=tf.float32, shape=initialStatesShape, name='initial_states')
        if not self.useInitialStates:
            return initialStates, None

        partitions = []
        for i in range(initialStatesShape[0]):
            if i < countPerState:
                partitions.append(0)
            else:
                partitions.append(1)
        allInputStates = tf.dynamic_partition(initialStates, partitions, 2)
        for i in range(len(allInputStates)):
            allInputStates[i] = tf.reshape(allInputStates[i], shape=stateShape[0])

        return initialStates, (allInputStates[0], allInputStates[1])

    def CudnnLSTM(self, X):
        if not self.timeMajor:
            X = tf.transpose(X, perm=[1, 0, 2])

        direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
        print('CudnnLSTM direction', direction)
        
        cudnnLSTM = cudnn_rnn.CudnnLSTM(self.numLayers, self.numUnits, direction=direction, dropout=self.dropout, name=self.cudnnLSTMName)        
        initialStates, initialStatesTuple = self.CudnnLSTMInputStates(cudnnLSTM)
        outputs, (outputStatesH, outputStatesC) = cudnnLSTM(X, initial_state=initialStatesTuple, training=True)
        if not self.timeMajor:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        outputStates = tf.concat((outputStatesH, outputStatesC), 0, name="output_states")
        return outputs, initialStates, outputStates

    def LSTMToLogits(self, outputs):
        outlayerDim = tf.shape(outputs)[2]
        outputs = tf.reshape(outputs, [self.batchSize * self.maxTime, outlayerDim])
        logits = LinearLayer(outputs, 2 * self.numUnits, self.yDim, 'output_linear_w', 'output_linear_b')
        return logits

    def BuildGraph(self, dropout):
        self.dropout = dropout
        print('dropout', dropout)
        batchSize, maxTime, numLayers, numUnits, xDim, yDim, timeMajor, useCudnn, useInitialStates = self.ModelInfo()

        tensorDic = self.tensorDic

        X, sequenceLength = self.XAndSequenceLength()
        YShape = (batchSize, maxTime, yDim)
        if timeMajor:
            YShape = (maxTime, batchSize, yDim)
        Y = tf.placeholder(dtype=tf.float32, shape=YShape, name='Y')
        learningRate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        tensorDic['X'] = X
        tensorDic['Y'] = Y
        tensorDic['sequence_length'] = sequenceLength
        tensorDic['learning_rate'] = learningRate
        
        if useCudnn:
            outputs, initialStates, outputStates = self.CudnnLSTM(X)
        else:
            outputs, initialStates, outputStates = self.LSTM(X, sequenceLength)
        tensorDic['initial_states'] = initialStates
        tensorDic['output_states'] = outputStates
        
        logits = self.LSTMToLogits(outputs)
        Y = tf.reshape(Y, [batchSize * maxTime, yDim])
        
        loss_op = self.LossOp(logits, Y)
        tensorDic['loss_op'] = loss_op
        tensorDic['train_op'] = self.TrainOp(loss_op, learningRate)
        tensorDic['accuracy'] = self.AccuracyOp(logits, Y)
        tensorDic['classify_info'] = self.ClassifyInfoTensor(logits, Y)

        print('build rnn done')

    def Restore(self, sess, modelFilePath):
        if self.useCudnn:
            self.RestoreForCudnn(sess, modelFilePath)
            return

        graphFile = modelFilePath + '.meta'
        saver = tf.train.import_meta_graph(graphFile)
        saver.restore(sess, modelFile)

        tensorDic = self.tensorDic
        graph = tf.get_default_graph()
        tensorDic['X'] = graph.get_tensor_by_name('X:0')
        tensorDic['sequence_length'] = graph.get_tensor_by_name('sequence_length:0')
        tensorDic['predict_op'] = graph.get_tensor_by_name('predict_op:0')
        tensorDic['initial_states'] = graph.get_tensor_by_name('initial_states:0')
        tensorDic['output_states'] = graph.get_tensor_by_name('output_states:0')

        print('Restore done')

    def RestoreForCudnn(self, sess, modelFilePath):
        numUnits = self.numUnits
        numLayers = self.numLayers
        singleCell = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.numUnits)
        cellsFW = [singleCell() for _ in range(numLayers)]
        cellsBW = [singleCell() for _ in range(numLayers)]
        X, sequenceLength = self.XAndSequenceLength()
        initialStates, initialStatesFW, initialStatesBW = self.LSTMInputStates()
        with tf.variable_scope(self.cudnnLSTMName):
            outputs, outputStatesFW, outputStatesBW = rnn.stack_bidirectional_dynamic_rnn(
                cellsFW, cellsBW, X, sequence_length=sequenceLength, dtype=tf.float32, 
                initial_states_fw=initialStatesFW, initial_states_bw=initialStatesBW)

        outputStates = self.LSTMOuputStates(outputStatesFW, outputStatesBW)
        
        logits = self.LSTMToLogits(outputs)
        predictOp = tf.nn.softmax(logits, name='predict_op')
        tensorDic = self.tensorDic
        tensorDic['X'] = X
        tensorDic['sequence_length'] = sequenceLength
        tensorDic['predict_op'] = predictOp
        tensorDic['initial_states'] = initialStates
        tensorDic['output_states'] = outputStates

        saver = tf.train.Saver()
        saver.restore(sess, modelFilePath)

        print('RestoreForCudnn done')

    def InitialStatesZero(self):
        if 'initial_states' not in self.tensorDic:
            print('initial_states tensor not found')
            return []

        initialStates = self.tensorDic['initial_states']
        return np.zeros(initialStates.shape)



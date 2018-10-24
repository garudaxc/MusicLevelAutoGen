import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib import crf
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

def RenameVarName(checkpointDir, replaceFrom = None, replaceTo = None, prefix = None):
    checkpoint = tf.train.get_checkpoint_state(checkpointDir)
    with tf.Session() as sess:
        varList = tf.contrib.framework.list_variables(checkpointDir)
        for varName, _ in varList:
            variable = tf.contrib.framework.load_variable(checkpointDir, varName)
            newName = varName
            if replaceFrom is not None and replaceTo is not None:
                newName = newName.replace(replaceFrom, replaceTo)
            if prefix is not None:
                newName = prefix + newName

            print('Renaming %s to %s.' % (varName, newName))
            newVariable = tf.Variable(variable, name=newName)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint.model_checkpoint_path)

def TFVariable(shape, dtype=tf.float32, name=None):
    # return tf.get_variable(name=name, initializer=tf.random_normal(shape, dtype=dtype))
    return tf.Variable(tf.random_normal(shape, dtype=dtype), name=name)

def Conv2dLayer(inputData, filterShape, name=None):
    filterTensor = TFVariable(shape=filterShape)
    convLayer = tf.nn.conv2d(inputData, filterTensor, [1, 1, 1, 1], "SAME", name=name)
    reluLayer = tf.nn.relu(convLayer)
    return reluLayer

def LinearLayer(inputData, dimA, dimB, nameWeight=None, nameBias=None):
    weight = TFVariable(shape=[dimA, dimB], name=nameWeight)
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
    def __init__(self, variableScopeName, batchSize, maxTime, numLayers, numUnits, xDim, yDim, timeMajor=False, useCudnn=False, restoreCudnnWithGPUMode=False, useInitialStatesFW=True, useInitialStatesBW=False):
        self.variableScopeName = variableScopeName
        self.batchSize = batchSize
        self.maxTime = maxTime
        self.numLayers = numLayers
        self.numUnits = numUnits
        self.xDim = xDim
        self.yDim = yDim
        self.timeMajor = timeMajor
        self.useCudnn = useCudnn
        self.useInitialStatesFW = useInitialStatesFW
        self.useInitialStatesBW = useInitialStatesBW
        self.useCRF = False
        self.restoreCudnnWithGPUMode = restoreCudnnWithGPUMode
        self.tensorDic = {}
        print('batchSize', batchSize, 'maxTime', maxTime, 'numLayers', numLayers, 'numUnits', numUnits)
        print('xDim', xDim, 'yDim', yDim, 'timeMajor', timeMajor, 'useCudnn', useCudnn)
        print('useInitialStates', self.UseInitialStates(), 'useInitialStatesFW', self.useInitialStatesFW, 'useInitialStatesBW', self.useInitialStatesBW)
        self.cudnnLSTMName = 'note_cudnn_lstm'

    def UseInitialStates(self):
        return self.useInitialStatesFW or self.useInitialStatesBW

    def ModelInfo(self):
        return self.batchSize, self.maxTime, self.numLayers, self.numUnits, self.xDim, self.yDim, self.timeMajor, self.useCudnn

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
        # tf 1.5.0 or later, use v2 default
        # crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        loss = tf.reduce_mean(crossEntropy * classWeight)
        return loss

    def LossOpCRF(self, logits, Y, sequenceLength):
        logits = tf.reshape(logits, [self.batchSize, self.maxTime, self.yDim])
        Y = tf.reshape(Y, [self.batchSize, self.maxTime, self.yDim])
        Y = tf.argmax(Y, axis=2)
        logLikelihood, transitionParams = crf.crf_log_likelihood(logits, Y, sequenceLength)
        loss = tf.reduce_mean(-logLikelihood)
        return loss, transitionParams

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
        XShape = (self.batchSize, None, self.xDim)
        if self.timeMajor:
            XShape = (None, self.batchSize, self.xDim)

        X = tf.placeholder(dtype=tf.float32, shape=XShape, name='X')
        sequenceLength = tf.placeholder(tf.int32, [None], name='sequence_length')
        return X, sequenceLength

    def LSTMInputStates(self, initialStatesName):
        numLayers = self.numLayers
        batchSize = self.batchSize
        numUnits = self.numUnits
        statesCount = numLayers * 2 * 2
        initialStates = tf.placeholder(dtype=tf.float32, shape=(batchSize * statesCount, numUnits), name=initialStatesName)
        initialStatesFW = None
        initialStatesBW = None
        if self.UseInitialStates():
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

        if not self.useInitialStatesFW:
            initialStatesFW = None
        
        if not self.useInitialStatesBW:
            initialStatesBW = None

        return initialStates, initialStatesFW, initialStatesBW

    def LSTMOuputStates(self, statesFW, statesBW, outputStatesName):
        outputStatesArr = []
        for states in statesFW:
            for state in states:
                outputStatesArr.append(state)
        for states in statesBW:
            for state in states:
                outputStatesArr.append(state)

        outputStates = tf.concat(outputStatesArr, 0, name=outputStatesName)
        return outputStates

    def LSTM(self, X, sequenceLength, initialStatesName, outputStatesName, cells=None):
        numLayers = self.numLayers
        if cells is None:
            print('cells is None. create cells')
            cells = []
            dropoutCells = []
            for i in range(numLayers * 2):
                cell = rnn.LSTMCell(self.numUnits, use_peepholes=True, forget_bias=1.0)
                cells.append(cell)

                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1.0 - self.dropout))
                dropoutCells.append(cell)

            usedCells = dropoutCells
        else:
            print('cells is not None. reuse cells.')
            dropoutCells = cells
            usedCells = cells

        if self.timeMajor:
            X = tf.transpose(X, perm=[1, 0, 2])

        initialStates, initialStatesFW, initialStatesBW = self.LSTMInputStates(initialStatesName)

        outputs, statesFW, statesBW = rnn.stack_bidirectional_dynamic_rnn(
            usedCells[0:numLayers], usedCells[numLayers:], 
            X, sequence_length=sequenceLength, dtype=tf.float32,
            initial_states_fw=initialStatesFW, initial_states_bw=initialStatesBW)

        if self.timeMajor:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        outputStates = self.LSTMOuputStates(statesFW, statesBW, outputStatesName)

        return outputs, initialStates, outputStates, cells, dropoutCells

    def CudnnLSTMInputStates(self, cudnnLSTM, initialStatesName):
        stateShape = cudnnLSTM.state_shape(self.batchSize)
        initialStatesShape = [stateShape[0][0] * 2, stateShape[0][1], stateShape[0][2]]
        initialStates = tf.placeholder(dtype=tf.float32, shape=initialStatesShape, name=initialStatesName)
        if not self.UseInitialStates():
            return initialStates, None

        partitionShape = stateShape[0]
        partitionSize = stateShape[0][0]
        partitionNum = 2
        useBothDir = self.useInitialStatesFW and self.useInitialStatesBW
        print('partitionShape', partitionShape)
        if not useBothDir:
            partitionShape = [int(stateShape[0][0] / 2), stateShape[0][1], stateShape[0][2]]
            partitionSize = int(partitionSize / 2)
            partitionNum = (partitionNum * 2)
            print('not use both dir states. change partitionShape to', partitionShape)

        partitions = []
        for i in range(initialStatesShape[0]):
            val = i // partitionSize
            partitions.append(val)
        print('partitions', partitions)

        allInputStates = tf.dynamic_partition(initialStates, partitions, partitionNum)
        for i in range(len(allInputStates)):
            allInputStates[i] = tf.reshape(allInputStates[i], shape=partitionShape)

        if useBothDir:
            initialStatesH = allInputStates[0]
            initialStatesC = allInputStates[1]
        else:
            zeroA = tf.constant(0, dtype=tf.float32, shape=partitionShape)
            zeroB = tf.constant(0, dtype=tf.float32, shape=partitionShape)
            # tensorflow 和 nvidia 的文档没写清楚这里forward 和 backward的顺序...
            if self.useInitialStatesFW:
                initialStatesH = tf.concat([allInputStates[0], zeroA], 0)
                initialStatesC = tf.concat([allInputStates[2], zeroB], 0)
            else:
                initialStatesH = tf.concat([allInputStates[1], zeroA], 0)
                initialStatesC = tf.concat([allInputStates[3], zeroB], 0)

        print('initialStatesH', initialStatesH)
        print('initialStatesC', initialStatesC)
        return initialStates, (initialStatesH, initialStatesC)

    def CudnnLSTM(self, X, training=True):
        if not self.timeMajor:
            X = tf.transpose(X, perm=[1, 0, 2])

        direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
        print('CudnnLSTM direction', direction)

        dropout = 0
        if training:
            dropout = self.dropout
        cudnnLSTM = cudnn_rnn.CudnnLSTM(self.numLayers, self.numUnits, direction=direction, dropout=dropout, name=self.cudnnLSTMName)        
        initialStates, initialStatesTuple = self.CudnnLSTMInputStates(cudnnLSTM, 'initial_states')
        outputs, (outputStatesH, outputStatesC) = cudnnLSTM(X, initial_state=initialStatesTuple, training=training)
        if not self.timeMajor:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        outputStates = tf.concat((outputStatesH, outputStatesC), 0, name='output_states')
        return outputs, initialStates, outputStates

    def LSTMToLogits(self, outputs, weight=None, bias=None):
        outlayerDim = tf.shape(outputs)[2]
        outputs = tf.reshape(outputs, [-1, outlayerDim])

        if weight is None:
            weight = TFVariable(shape=[2 * self.numUnits, self.yDim], name='output_linear_w')

        if bias is None:
            bias = TFVariable(shape=[self.yDim], name='output_linear_b')

        logits = tf.matmul(outputs, weight) + bias
        return logits, weight, bias

    def BuildGraph(self, dropout):
        if not self.HasVariableScopeName():
            self.BuildGraphImp(dropout)
            return

        with tf.variable_scope(self.variableScopeName):
            self.BuildGraphImp(dropout)

    def BuildGraphImp(self, dropout):
        self.dropout = dropout
        print('dropout', dropout)
        batchSize, maxTime, numLayers, numUnits, xDim, yDim, timeMajor, useCudnn = self.ModelInfo()

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
            outputs, initialStates, outputStates, cells, dropoutCells = self.LSTM(X, sequenceLength, 'initial_states', 'output_states')
        tensorDic['initial_states'] = initialStates
        tensorDic['output_states'] = outputStates
        
        logits, weight, bias = self.LSTMToLogits(outputs)
        tensorDic['logits'] = logits
        Y = tf.reshape(Y, [batchSize * maxTime, yDim])
        
        lossOp = self.LossOp(logits, Y)
        tensorDic['loss_op'] = lossOp
        tensorDic['train_op'] = self.TrainOp(lossOp, learningRate)
        tensorDic['accuracy'] = self.AccuracyOp(logits, Y)
        tensorDic['classify_info'] = self.ClassifyInfoTensor(logits, Y)

        if self.useCRF:
            logitsCRF = tf.placeholder(tf.float32, shape=[self.batchSize * self.maxTime, self.yDim])
            lossOpCRF, transitionParams = self.LossOpCRF(logitsCRF, Y, sequenceLength)
            tensorDic['logits_crf'] = logitsCRF
            tensorDic['loss_op_crf'] = lossOpCRF
            tensorDic['transition_params'] = transitionParams

        if not useCudnn:
            # not drop out for predict
            outputsPredict, initialStatesPredict, outputStatesPredict, _, _ = self.LSTM(X, sequenceLength, 'initial_states_predict', 'output_states_predict', cells=cells)
            logitsPredict, _, _ = self.LSTMToLogits(outputsPredict, weight=weight, bias=bias)
            predictOp = tf.nn.softmax(logitsPredict, name='predict_op')
            
        print('build rnn done')

    def Restore(self, sess, modelFilePath):
        if self.useCudnn:
            self.RestoreForCudnn(sess, modelFilePath)
            return

        graphFilePath = modelFilePath + '.meta'
        saver = tf.train.import_meta_graph(graphFilePath)
        saver.restore(sess, modelFilePath)

        tensorDic = self.tensorDic
        graph = tf.get_default_graph()
        tensorDic['X'] = self.GetTensorByName(graph, 'X:0')
        tensorDic['sequence_length'] = self.GetTensorByName(graph, 'sequence_length:0')
        tensorDic['predict_op'] = self.GetTensorByName(graph, 'predict_op:0')
        tensorDic['initial_states'] = self.GetTensorByName(graph, 'initial_states_predict:0')
        tensorDic['output_states'] = self.GetTensorByName(graph, 'output_states_predict:0')

        print('Restore done')

    def RestoreForCudnn(self, sess, modelFilePath):
        if not self.HasVariableScopeName():
            self.RestoreForCudnnImp(sess, modelFilePath)
            return
    
        with tf.variable_scope(self.variableScopeName):
            self.RestoreForCudnnImp(sess, modelFilePath)

    def RestoreForCudnnImp(self, sess, modelFilePath):
        X, sequenceLength = self.XAndSequenceLength()
        if self.restoreCudnnWithGPUMode:
            outputs, initialStates, outputStates = self.CudnnLSTM(X, training=False)
        else:
            X, sequenceLength = self.XAndSequenceLength()
            numUnits = self.numUnits
            numLayers = self.numLayers
            singleCell = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.numUnits)
            cellsFW = [singleCell() for _ in range(numLayers)]
            cellsBW = [singleCell() for _ in range(numLayers)]

            initialStates, initialStatesFW, initialStatesBW = self.LSTMInputStates('initial_states')
            with tf.variable_scope(self.cudnnLSTMName):
                outputs, outputStatesFW, outputStatesBW = rnn.stack_bidirectional_dynamic_rnn(
                    cellsFW, cellsBW, X, sequence_length=sequenceLength, dtype=tf.float32, 
                    initial_states_fw=initialStatesFW, initial_states_bw=initialStatesBW)

            outputStates = self.LSTMOuputStates(outputStatesFW, outputStatesBW, 'output_states')
        
        logits, _, _ = self.LSTMToLogits(outputs)
        predictOp = tf.nn.softmax(logits, name='predict_op')
        tensorDic = self.tensorDic
        tensorDic['X'] = X
        tensorDic['sequence_length'] = sequenceLength
        tensorDic['predict_op'] = predictOp
        tensorDic['initial_states'] = initialStates
        tensorDic['output_states'] = outputStates

        if self.useCRF:
            transitionParams = TFVariable([self.yDim, self.yDim], name='transitions')
            logitsCRF = tf.reshape(logits, [self.batchSize, self.maxTime, self.yDim])
            decodeTags, bestScore = crf.crf_decode(logitsCRF, transitionParams, sequenceLength)
            tensorDic['decode_tags'] = decodeTags
        
        varList = None
        if self.HasVariableScopeName():
            varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.variableScopeName)
        saver = tf.train.Saver(var_list=varList)
        saver.restore(sess, modelFilePath)

        print('RestoreForCudnn done')

    def GetTensorByName(self, graph, name):
        return graph.get_tensor_by_name(self.variableScopeName + '/' + name)

    def HasVariableScopeName(self):
        return (self.variableScopeName is not None) and (len(self.variableScopeName) > 0)

    def InitialStatesZero(self):
        if 'initial_states' not in self.tensorDic:
            print('initial_states tensor not found')
            return []

        initialStates = self.tensorDic['initial_states']
        return np.zeros(initialStates.shape)
    



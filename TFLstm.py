import pickle
import os
import LevelInfo
import numpy as np
import myprocesser
import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import matplotlib.pyplot as plt
import util
import DownbeatTracking
import madmom
import shutil

rootDir = util.getRootDir()
trainDataDir = rootDir + 'MusicLevelAutoGen/train_data/'

runList = []
def run(r):
    runList.append(r)
    return r

LevelDifficutly = 2
numHidden = 26
batchSize = 16
numSteps = 512
numSteps = 128
inputDim = 314
featureExtractFunc = myprocesser.LoadAndProcessAudio
# outputDim = 2
learning_rate = 0.001
bSaveModel = True

epch = 10000

# for long note
songList = ['aiqingmaimai','ai', 'hellobaby', 'hongri', 'houhuiwuqi', 'huashuo', 'huozaicike', 'haodan']
songList = ['huashuo', 'haodan']


songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic']

songList = ['inthegame', 'remains', 'ilikeit', 'haodan', 'ai', 
'1987wbzhyjn', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'myall', 'unreal', 'faraway',
'fengniao', 'wangfei', 'wodemuyang', 'mrx', 'rageyourdream', 'redhot', 'yilububian', 'yiqiyaobai', 'yiranaini', 'yiyejingxi',
'hongri', 'huahuangshou', 'huashuo', 'ificoulefly', 'ineedmore', 'iwantmytearsback',  'rippingthebackside', 
 'whencaniseeyouagain', 'wingsofpiano', 'wocanggwz', 'ygrightnow']

songList = ['inthegame', 'remains', 'ilikeit', 'haodan', 'ai', 
'1987wbzhyjn', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'myall', 'unreal', 'faraway',
'fengniao', 'wangfei', 'wodemuyang', 'mrx', 'rageyourdream', 'redhot', 'yilububian', 'yiqiyaobai', 'yiranaini', 'yiyejingxi']


# 'mrq',  'huozaicike' 'ribuluo'

testing = False
if testing:
    numSteps = 64
    epch = 3
    songList = ['aiqingmaimai','ai']


def FrameIndex(time, msPerFrame):
    
    frameIndex = time // msPerFrame
    residual = time % msPerFrame
    residual = min(residual, abs(residual-msPerFrame))

    # if residual > 3:
    #     print('residual too large', residual)
    return int(frameIndex)
    

def FillCombineNoteAsLongNote(data, combineNote, msPerFrame):
    begin = combineNote[0][0]
    end = 0
    for note in combineNote:
        time = note[0]
        last = time + note[2]
        if time < begin:
            begin = time
        if last > end:
            end = last
    i0 = FrameIndex(begin, msPerFrame)
    i1 = FrameIndex(end, msPerFrame)
    data[i0:i1, 3] = 1.0

def FillNote(data, note, msPerFrame, isCombineNote):
    # 音符数据转化为sample data
    index = FrameIndex(note[0], msPerFrame)
    type = note[1]

    if type == LevelInfo.shortNote:
        data[index, 0] = 0.0 
        data[index, 1] = 1.0
    elif type == LevelInfo.slideNote:
        if (not isCombineNote):
            data[index, 0] = 0.0 
            data[index, 2] = 1.0
        else:
            data[index, 0] = 0.0 
            data[index, 3] = 1.0
    elif type == LevelInfo.longNote:
        last = FrameIndex(note[2], msPerFrame)
        data[index:index+last, 0] = 0.0 
        data[index:index+last, 4] = 1.0
    else:
        assert False


def ConvertLevelToLables(level, numframes, msMinInterval = 10):
    '''
    关卡数据转换训练数据
    关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    样本格式 non-note short slide slide-in-combine long    
    '''
    frames = [[1.0, 0.0, 0.0, 0.0, 0.0]] * numframes
    frames = np.array(frames)

    combineNote = 0
    notesInCombine = 0
    for note in level:
        time, type, val = note
        if type == LevelInfo.combineNode:             
            combineNote += 1
            for n in val:
                FillNote(frames, n, msMinInterval, True)
        else:        
            FillNote(frames, note, msMinInterval, False)

    # a = np.sum(frames, axis=0)
    # print('frames', a, 'combine note', combineNote, 'notes in combine', notesInCombine)

    return frames


def SamplesToSoftmax(samples):
    '''
    将样本数据转换为分类数据
    '''
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    s1 = 1 - np.amax(samples, axis=1)
    s1 = s1[:, np.newaxis]
    return np.hstack((s1, samples))


def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = rootDir + 'rm/'
    return path

def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.m4a' % (path, song, song)
    if not os.path.exists(pathname):
        pathname = '%s%s/%s.mp3' % (path, song, song)
    if not os.path.exists(pathname):
        pathname = '%s%s/%s.wav' % (path, song, song)

    return pathname

def MakeSongDataPathName(song, dataName, ext='.csv'):
    path = GetSamplePath()
    pathname = '%s%s/%s_%s%s' % (path, song, song, dataName, ext)
    return pathname

def MakeDirIfNotExists(dirPath):
    if os.path.exists(dirPath):
        return

    os.mkdir(dirPath)

def MakeSongSubDir(song, subDirName):
    songDir = MakeMp3Dir(song)
    subDir = songDir + subDirName
    MakeDirIfNotExists(songDir + subDirName)
    return subDir + '/'

def MakeMp3Dir(song):
    path = GetSamplePath()
    pathname = '%s%s/' % (path, song)
    if not os.path.exists(pathname):
        assert False
    return pathname

def MakeLevelPathname(song, difficulty=2):
    path = GetSamplePath()
    diff = ['ez', 'nm', 'hd']
    pathname = '%s%s/%s_4k_%s.imd' % (path, song, song, diff[difficulty])
    return pathname   

def PrepareTrainData(songList, batchSize = 32, loadTestData = True, featureFunc=myprocesser.LoadAndProcessAudio):   
    trainx = []
    trainy = []
    for song in songList:
        pathname = MakeMp3Pathname(song)
        inputData = featureFunc(pathname)
        trainx.append(inputData)
        numSample = len(inputData)

        if loadTestData:
            pathname = MakeLevelPathname(song, difficulty=LevelDifficutly)
            level = LevelInfo.LoadRhythmMasterLevel(pathname)

            targetData = ConvertLevelToLables(level, numSample)

            # a = targetData[:, [1, 2, 3]]
            # count = (a>0.1).nonzero()[0].shape[0]
            # print('count', count)
                
            trainy.append(targetData)

    trainx = np.vstack(trainx)
    if len(trainy) > 0:
        trainy = np.vstack(trainy)

    # if not useSoftmax:
    #     trainy = trainy[:,0]

    return trainx, trainy

# @run
def RhythmMasterLevelPorcess():
    # pathname = r'd:\librosa\RhythmMaster\hangengxman\hangengxman_4k_nm.imd'
    # level = LevelInfo.LoadRhythmMasterLevel(pathname)
    # print(type(level))
    # count = 0
    # for note in level:
    #     if note[1] == 3:
    #         print(count, type(note[2]))
    #         count += 1

    songList = ['feiyuedexin']
    songList = ['aiqingmaimai']
    x, y = PrepareTrainData(songList, 32, True)

    y = y[:,3]
    
    acceptThrehold = 0.9

    pathname = MakeMp3Pathname(songList[0])
    # y = y > acceptThrehold
    # LevelInfo.SaveSamplesToRegionFile(y, pathname, 'region')
    # return    

    notes = postprocess.TrainDataToLevelData(y, 0, acceptThrehold)
    notes = np.asarray(notes)
    notes[0]


    notes = notes[:,0]
    LevelInfo.SaveInstantValue(notes, pathname, '_slide')




class TrainDataBase():

    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        seqLen = np.array([self.numSteps] * self.batchSize)
        return x, y, seqLen

    def ShuffleBatch(self):
        s = np.arange(self.batchSize)
        np.random.shuffle(s)

        self._x = self._x[:, s]

        if not self._y[0] is None:
            self._y = self._y[:, s]

    def BuildBatch(self, x, y):
        count = x.shape[0]
        self.numBatches = count // (self.batchSize * self.numSteps)
        count = self.numBatches * (self.batchSize * self.numSteps)

        y = y[:count]
        yDim = y.shape[1]
        y = y.reshape(self.batchSize, -1, self.numSteps, yDim)
        self._y = y.transpose(1, 0, 2, 3)

        x = x[:count]
        xDim = x.shape[1]
        x = x.reshape(self.batchSize, -1, self.numSteps, xDim)
        self._x = x.transpose(1, 0, 2, 3)
        print('numbatches', self.numBatches, 'x shape', self._x.shape, 'y shape', self._y.shape)

###
# TrainDataDynShortNoteSinging
###
class TrainDataDynShortNoteSinging(TrainDataBase):
    lableDim = 2
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if (not y is None): 
            beat = np.max(y[:, 1:4], axis=1)
            
            long = np.max(y[:, 4:5], axis=1)
            for i in range(1, len(long)):
                if long[i-1] == 0 and long[i] > 0:
                    beat[i] = 1

            y = SamplesToSoftmax(beat)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 

    def GetModelPathName():
        return 'd:/work/model_shortnote_singing.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_short_singing.raw'
    
    def GenerateLevel(predicts, pathname):      
                       
        pre = predicts[:, 1]
        # pre = postprocess.PurifyInstanceSample(pre)
        # picked = postprocess.PickInstanceSample(pre)

        # sam = np.zeros_like(pre)
        # sam[picked] = 1
        
        notes = postprocess.TrainDataToLevelData(pre, 10, 0.90, 0)
        notes = np.asarray(notes)
        notes[0]

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst_singing')



class TrainDataDynShortNoteBeat(TrainDataBase):
    lableDim = 2
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if (not y is None): 
            beat = np.max(y[:, 1:4], axis=1)
            
            long = np.max(y[:, 4:5], axis=1)
            for i in range(1, len(long)):
                if long[i-1] == 0 and long[i] > 0:
                    beat[i] = 1

            y = SamplesToSoftmax(beat)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 

    def GetModelPathName():
        return rootDir + 'model_shortnote_beat/model_shortnote_beat.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_short_beat.raw'
    
    def GenerateLevel(predicts, pathname):     
                       
        short = predicts[:, 1]
        short = postprocess.PickInstanceSample(short, count=400)
        notes = postprocess.TrainDataToLevelData(short, 10, 0.1, 0)

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst_beat')


class TrainDataDynSinging(TrainDataBase):
    lableDim = 3
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if (not y is None):
            y = y[:, 0:self.lableDim]
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 

    def GetModelPathName():
        return rootDir + 'model_singing/model_singing.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_singing.raw'
    
    def GenerateLevel(predicts, pathname):
        return False


###
# TrainDataDynLongNote
###
class TrainDataDynLongNote(TrainDataBase):
    lableDim = 4
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        if (not y is None):
            
            # beat = np.max(y[:, 3:5], axis=1)
            # y = SamplesToSoftmax(beat)
            y = np.argmax(y, axis=1)
            ydata = np.zeros((len(y), self.lableDim))
            for i in range(len(y)):
                idx = y[i]
                if idx == 0:
                    ydata[i][0] = 1.0
                elif idx < 3:
                    ydata[i][1] = 1.0
                elif i > 0 and (y[i - 1] == y[i]):
                    ydata[i][3] = 1.0
                else:
                    ydata[i][2] = 1.0

            y = np.array(ydata)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 


    def GetModelPathName():
        return rootDir + 'model/model_longnote.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_long.raw'    
    
    def GenerateLevel(predicts, pathname):     
              
        # #for long note
        # print('evaluate shape', evaluate.shape)
        # predicts = evaluate[:,1] > acceptThrehold
        # LevelInfo.SaveSamplesToRegionFile(predicts, pathname, '_region')
        # return

        # predicts = postprocess.pick(evaluate, kernelSize=11)
        print('predicts', predicts.shape)

        # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')
        
        acceptThrehold = 0.8
        pred = predicts[:,1]
        notes = postprocess.TrainDataToLevelData(pred, 10, acceptThrehold, 0)
        print('gen notes number', len(notes))
        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_region')

        # LevelInfo.SaveInstantValue(notes, pathname, '_region')


def BuildDynamicRnn(X, Y, seqlen, learningRate, TrainData):
    result = {}

    weights = tf.Variable(tf.random_normal(shape=[2*numHidden, TrainData.lableDim]))
    bais = tf.Variable(tf.random_normal(shape=[TrainData.lableDim]))
    
    numLayers = 3
    cells = []
    dropoutCell = []

    state_size = tf.nn.rnn_cell.LSTMStateTuple(0, 0)
    for i in range(numLayers * 2):
        c = rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)
        cells.append(c)
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.6)
        dropoutCell.append(c)
        state_size = c.state_size

    statesCount = numLayers * 2 * 2
    initial_states = tf.placeholder(dtype=tf.float32, shape=(batchSize * statesCount, c.state_size.c), name='initial_states')
    result['initial_states'] = initial_states
    initial_states_re = tf.reshape(initial_states, [statesCount, batchSize, c.state_size.c])
    partitions = []
    for i in range(statesCount):
        partitions.append(i)
    allInputStates = tf.dynamic_partition(initial_states_re, partitions, statesCount)
    for i in range(len(allInputStates)):
        allInputStates[i] = tf.reshape(allInputStates[i], shape=[batchSize, c.state_size.c])
    initialStatesFW = []
    initialStatesBW = []
    for i in range(numLayers):
        initialStatesFW.append(tf.nn.rnn_cell.LSTMStateTuple(allInputStates[i * 2], allInputStates[i * 2 + 1]))
        initialStatesBW.append(tf.nn.rnn_cell.LSTMStateTuple(allInputStates[i * 2 + numLayers * 2], allInputStates[i * 2 + 1 + numLayers * 2]))
    
    # output, (states_fw), (states_bw) = rnn.stack_bidirectional_dynamic_rnn(
    #     dropoutCell[0:numLayers], dropoutCell[numLayers:], 
    #     X, sequence_length=seqlen, dtype=tf.float32,
    #     initial_states_fw=initialStatesFW, initial_states_bw=initialStatesBW)
    output, (states_fw), (states_bw) = rnn.stack_bidirectional_dynamic_rnn(
        dropoutCell[0:numLayers], dropoutCell[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32,
        initial_states_fw=initialStatesFW)

    allOutputStates = []
    for states in states_fw:
        for state in states:
            allOutputStates.append(state)
    for states in states_bw:
        for state in states:
            allOutputStates.append(state)

    batchStates = tf.concat(allOutputStates, 0, name="batch_states")
    result['batch_states'] = batchStates
 
    outlayerDim = tf.shape(output)[2]
    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])

    logits = tf.matmul(output, weights) + bais

    Y = tf.reshape(Y, [batchSize*numSteps, TrainData.lableDim])

    # weight for each class, balance the sample
    class_weight = tf.reduce_sum(Y, axis=0) / tf.cast(tf.shape(Y)[0], tf.float32)
    class_weight = tf.maximum(class_weight, tf.ones_like(class_weight) * 0.001)
    class_weight = (1.0 / TrainData.lableDim) / class_weight
    result['class_weight'] = class_weight
    class_weight = tf.reduce_sum(class_weight * Y, axis=1)

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

    loss_op = tf.reduce_mean(crossEntropy * class_weight)
    result['loss_op'] = loss_op

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)
    result['train_op'] = train_op

    correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    result['accuracy'] = accuracy

    # classify debug info tensor
    classifyInfoArr = []
    for idx in range(TrainData.lableDim):
        tempYIdxArr = [idx] * (batchSize * numSteps)
        tempYIdxArr = tf.constant(tempYIdxArr, dtype=tf.int64, shape=[batchSize*numSteps])
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
    result['classify_info'] = classifyInfo

    # prediction without dropout
    output, (states_fw_predict), (states_bw_predict) = rnn.stack_bidirectional_dynamic_rnn(
        cells[0:numLayers], cells[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32, initial_states_fw=initialStatesFW)

    allOutputStatesPredict = []
    for states in states_fw_predict:
        for state in states:
            allOutputStatesPredict.append(state)
    for states in states_bw_predict:
        for state in states:
            allOutputStatesPredict.append(state)
    batchStatesPredict = tf.concat(allOutputStatesPredict, 0, name="batch_states_predict")
    result['batch_states_predict'] = batchStatesPredict

    # output = tf.reshape(output, [batchSize*maxSeqLen, outlayerDim])
    
    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])
    logits = tf.matmul(output, weights) + bais

    predict_op = tf.nn.softmax(logits, name='predict_op')
    result['predict_op'] = predict_op

    print('build rnn done')
    return result

   
def SaveModel(sess, filename, saveMeta = False):
    saver = tf.train.Saver()
    saver.save(sess, filename, write_meta_graph=saveMeta)
    print('model saved')


def EvaluateWithModel(modelFile, song, rawFile, TrainData, featureFunc=myprocesser.LoadAndProcessAudio):
    # 加载训练好的模型，将结果保存到二进制文件
    predicts = RunModel(modelFile, MakeMp3Pathname(song), TrainData, featureFunc)
    with open(rawFile, 'wb') as file:
        pickle.dump(predicts, file)
        print('raw file saved', predicts.shape)

    return predicts

def RunModel(modelFile, songFile, TrainData, featureFunc=myprocesser.LoadAndProcessAudio):    
    predictGraph = tf.Graph()
    graphFile = modelFile + '.meta'

    with predictGraph.as_default():
        saver = tf.train.import_meta_graph(graphFile)
        with tf.Session() as sess:
            
            saver.restore(sess, modelFile)
            print('model loaded')

            predict_op = tf.get_default_graph().get_tensor_by_name("predict_op:0")
            print('predict_op', predict_op)         
            X = tf.get_default_graph().get_tensor_by_name('X:0')
            seqLenHolder = tf.get_default_graph().get_tensor_by_name('seqLen:0')
            
            reuseState = True
            if reuseState:
                initial_states = tf.get_default_graph().get_tensor_by_name('initial_states:0')
                # batch_states = tf.get_default_graph().get_tensor_by_name('batch_states_predict:0')
                batch_states = tf.get_default_graph().get_tensor_by_name('batch_states:0')
                initialStatesZero = np.array([[0.0] * batch_states.shape[1]] * batch_states.shape[0])
                currentInitialStates = initialStatesZero

            testx = featureFunc(songFile)

            data = TrainData(batchSize, numSteps, testx)
            
            evaluate = []
            for i in range(data.numBatches):
                xData, _, seqLen = data.GetBatch(i)
                if not reuseState:
                    t = sess.run(predict_op, feed_dict={X:xData, seqLenHolder:seqLen})
                else:
                    t, batchStates = sess.run([predict_op, batch_states], feed_dict={X:xData, seqLenHolder:seqLen, initial_states: currentInitialStates})
                    currentInitialStates = batchStates
                
                t = t[0:numSteps,:]
                evaluate.append(t)


            predicts = np.stack(evaluate).reshape(-1, TrainData.lableDim)

    return predicts

def AlignNoteWithBPMAndET(notes, frameInterval, bpm, et):
    '''
    notes 10 ms
    et ms
    '''
    beatInterval = 60.0 / bpm
    beatInterval *= 1000
    posPerbeat = 8
    posInterval = beatInterval / posPerbeat
    secondPerBeat = 60.0 / bpm
    noteTimes = []
    alignedPos = []
    maxNotePerBeat = 2
    posScale = posInterval * (posPerbeat / maxNotePerBeat)
    for i in range(len(notes)):
        if notes[i] < 1:
            continue

        timeInMS = i * frameInterval - et
        pos = round(timeInMS / posScale)
        addPos = pos
        # if pos % maxNotePerBeat == 0:
        #     addPos = pos
        # else:
        #     lenAlignedPos = len(alignedPos)
        #     if lenAlignedPos > 0 and (pos - alignedPos[-1]) < maxNotePerBeat:
        #         subPos = (timeInMS - noteTimes[-1]) / posScale
        #         if subPos > 1.5:
        #             subPos = 2
        #             addPos = alignedPos[-1] + subPos
        #         else:
        #             addPos = pos
        #     else:
        #         addPos = addPos

        if len(alignedPos) > 0:
            if addPos <= alignedPos[-1]:
                continue

        alignedPos.append(addPos)
        noteTimes.append(timeInMS)

    halfBeatInfoDic = {}
    for idx in range(len(alignedPos)):
        pos = alignedPos[idx]
        if pos % maxNotePerBeat != 0:
            hasPre = (idx > 0) and (pos - alignedPos[idx - 1] == 1)
            hasPost = (idx < len(alignedPos) -1) and (alignedPos[idx + 1] - pos == 1)
            count = 0
            if hasPre:
                count += 1
            if hasPost:
                count += 1
            halfBeatInfoDic[pos] = (pos, hasPre, hasPost, count)

    # DownbeatTracking.SaveInstantValue(alignedPos, MakeMp3Pathname('jinjuebianjingxian'), '_alignpos')
    newNotes = [0.0] * len(notes)
    for i in range(len(alignedPos)):
        pos = alignedPos[i]
        idx = int((pos * posScale + et + posInterval / 2) / frameInterval)
        if idx >= len(notes):
            continue

        if pos % maxNotePerBeat != 0:
            curHalfBeat = halfBeatInfoDic[pos]
            if (pos - maxNotePerBeat not in halfBeatInfoDic) and (pos + maxNotePerBeat not in halfBeatInfoDic):
                # if curHalfBeat[3] != 2:
                halfBeatInfoDic.pop(pos)
                continue
            elif curHalfBeat[3] == 0:
                threshold = 4
                tempCount = 0
                for tempIdx in range(-(threshold - 1), threshold):
                    if pos + maxNotePerBeat * tempIdx not in halfBeatInfoDic:
                        tempCount = 0
                        continue
                    
                    halfBeat = halfBeatInfoDic[pos + maxNotePerBeat * tempIdx]
                    if halfBeat[3] == 0:
                        tempCount += 1
                    
                    if tempCount == threshold:
                        break

                if tempCount < threshold:
                    halfBeatInfoDic.pop(pos)
                    continue

            elif not curHalfBeat[2]:
                if pos - maxNotePerBeat not in halfBeatInfoDic:
                    halfBeatInfoDic.pop(pos)
                    continue
                else:
                    beat = halfBeatInfoDic[pos - maxNotePerBeat]
                    if beat[3] != 2:
                        halfBeatInfoDic.pop(pos)
                        continue        
            else:
                if not curHalfBeat[1] and (pos - maxNotePerBeat not in halfBeatInfoDic):
                    halfBeatInfoDic.pop(pos)
                    continue

                if (pos - maxNotePerBeat in halfBeatInfoDic) and (pos - 2 * maxNotePerBeat in halfBeatInfoDic):
                    beatA = halfBeatInfoDic[pos - maxNotePerBeat]
                    beatB = halfBeatInfoDic[pos - 2 * maxNotePerBeat]
                    if beatA[3] != 0 and beatB[3] != 0:
                        halfBeatInfoDic.pop(pos)
                        continue
        newNotes[idx] = 1

    return np.array(newNotes)

def GenerateLevelImp(songFilePath, duration, bpm, et, shortPredicts, longPredicts, levelFilePath, templateFilePath, saveDebugFile = False):
    predicts = shortPredicts
    pathname = songFilePath
    singingActivation = predicts[:, 1]
    if saveDebugFile:
        DownbeatTracking.SaveInstantValue(singingActivation, pathname, '_result_singing')
        if len(predicts[0]) > 2:
            DownbeatTracking.SaveInstantValue(predicts[:, 2], pathname, '_result_bg')
        if len(predicts[0]) > 3:
            DownbeatTracking.SaveInstantValue(predicts[:, 3], pathname, '_result_dur')
        DownbeatTracking.SaveInstantValue(predicts[:, 0], pathname, '_result_no_label')
    short = DownbeatTracking.PickOnsetFromFile(pathname, bpm, duration, None, saveDebugFile)
    dis_time = 60 / bpm / 8
    singingPicker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=0.7, smooth=0.0, pre_max=dis_time, post_max=dis_time, fps=100)
    singingTimes = singingPicker(singingActivation)

    if saveDebugFile:
        DownbeatTracking.SaveInstantValue(singingTimes, pathname, '_result_singing_pick')

    singingTimes = singingTimes * 100
    singingTimes = singingTimes.astype(int)
    frameCount = len(short)
    singing = np.array([0] * frameCount)
    singing[singingTimes] = 1
    idxOffsetPre = int(60 / bpm * 1 * 100 * 0.9)
    idxOffsetPost = int(60 / bpm * 1 * 100 * 0.9)
    for singIdx in range(0, frameCount):
        if (singing[singIdx] < 1):
            continue
        
        idxStart = max(singIdx - idxOffsetPre, 0)
        idxEnd = min(singIdx + idxOffsetPost, frameCount)
        for idx in range(idxStart, idxEnd):
            short[idx] = 0

    checkShortCount = 0
    for val in short:
        if val > 0:
            checkShortCount += 1
    print('remove some short by singing. remain ', checkShortCount)
    short[singingTimes] = 1
    short = AlignNoteWithBPMAndET(short, 10, bpm, et)

    long = longPredicts[:, 3]
    levelNotes = postprocess.ProcessSampleToIdolLevel2(long, short, bpm)

    LevelInfo.GenerateIdolLevel(levelFilePath, levelNotes, bpm, et, duration, templateFilePath)

def AutoGenerateLevel(songFilePath, shortModelPath, longModelPath, levelFilePath):
    longPredicts = RunModel(longModelPath, songFilePath, TrainDataDynLongNote)
    shortPredicts = RunModel(shortModelPath, songFilePath, TrainDataDynSinging)
    duration, bpm, et = DownbeatTracking.CalcMusicInfoFromFile(songFilePath, -1, -1, False)
    duration = int(duration * 1000)
    et = int(et * 1000)
    GenerateLevelImp(songFilePath, duration, bpm, et, shortPredicts, longPredicts, levelFilePath, 'idol_template.xml', False)

# @run
def AutoGenerateLevelTool():
    configFilePath = 'config.txt'
    if not os.path.exists(configFilePath):
        print('configFilePath config.txt not found')
        return False

    print('find config.txt succeed')
    levelFileDir = ''
    songFileArr = []
    with open(configFilePath, 'r') as file:
        idx = 0
        for line in file:
            line = line.replace('\r', '\n')
            line = line.replace('\n', '')
            if idx == 0:
                levelFileDir = line
            else:
                if len(line) > 0:
                    songFileArr.append(line)
            idx += 1

    if not os.path.exists(levelFileDir):
        print('levelFileDir not exist')
        return False
    
    print('check levelFileDir ' + levelFileDir + 'end')
    for songFilePath in songFileArr:
        if not os.path.exists(songFilePath):
            print('songFilePath not exist' + songFilePath)
            return False

    print('check song list end')
    singModelPath = 'model/short/model_singing.ckpt'
    longModelPath = 'model/long/model_longnote.ckpt'
    for songFilePath in songFileArr:
        print('generate ' + songFilePath)
        songFileName = os.path.basename(songFilePath).split('.')[0]
        levelFilePath = os.path.join(levelFileDir, songFileName + '.xml')
        AutoGenerateLevel(songFilePath, singModelPath, longModelPath, levelFilePath)

    print(' ')
    print('all song level generate end ==========================')

    return True

@run
def GenerateLevel():
    print('gen level')

    debugBPM = -1
    debugET = -1
    song = ['PleaseDontGo']
    song = ['jilejingtu']
    song = ['aLIEz']
    song = ['bboombboom']
    song = ['1987wbzhyjn']
    song = ['mrq']
    song = ['ribuluo']
    song = ['xiagelukoujian']
    song = ['CheapThrills']
    song = ['dainiqulvxing']
    song = ['liuxingyu']
    song = ['Emmanuel - Corazon de Melao']
    # song = ['shaonianzhongguo']
    # song = ['danshengonghai']
    # song = ['sanguolian_2']
    # song = ['foxishaonv']
    # debugBPM = 128 
    # debugET = 768
    # song = ['xiagelukoujian']
    # debugBPM = 98
    # debugET = 2959
    # song = ['caocao']
    # song = ['dear my lady']
    # song = ['tian kong zhi cheng']
    # song = ['kanong']
    # song = ['qinghuaci']
    # song = ['zuichibi']
    # song = ['buchaobuyonghuaqian']
    # song = ['hxdssan']
    # song = ['zhongguohua']
    # song = ['100576SEXY_LOVE']
    # song = ['test_123woaini']
    # debugBPM = 90
    # debugET = 2737
    # song = ['test_aidezhudage']
    # debugBPM = 128
    # debugET = 8052
    # song = ['test_SuperStar']
    # debugBPM = 92
    # debugET = 476
    # song = ['test_zuichudemengxiang']
    # debugET = 605
    # debugBPM = 164
    # song = ['biaobai']

    # 
    # song = ['100580']
    # debugET = 100
    # song = ['ISI']
    # debugET = 15020
    # song = ['Luv Letter']
    # debugET = 29543
    # song = ['jianyunzhe']
    # debugET = 682
    song = ['jinjuebianjingxian']
    debugET = 6694
    # song = ['ouxiangwanwansui']
    # debugET = 571
    # song = ['blue planet']
    # debugET = 1615
    # song = ['xingzhixing']
    # debugET = 2360
    # song = ['yanghuadaqiao']
    # debugET = 170
    # song = ['Better Than One']
    # debugET = 260
    # song = ['DAY BY DAY']
    # debugET = 8210
    # song = ['Fátima']
    # debugBPM = 179
    # debugET = 670
    # song = ['PDD']
    # debugET = 147
    # song = ['Sakura Tears']
    # debugET = 356
    # song = ['SWEET TALKER']
    # debugET = 3750
    # song = ['ruqi']
    # debugET = 2253
    # song = ['shisanhaoxingqiwu']
    # debugET = 1145
    # song = ['shuxiu__']
    # debugET = 195
    # song = ['xiangrikuideyueding']
    # debugBPM = 158
    # debugET = 115

    # postprocess.ProcessSampleToIdolLevel(song[0])
    # return

    pathname = MakeMp3Pathname(song[0])
    print(pathname)

    useOnsetForShort = True
    if True:
        # gen raw data

        TrainData = TrainDataDynLongNote
        rawFile = TrainData.RawDataFileName(song[0])
        modelFile = TrainData.GetModelPathName()
        predicts = EvaluateWithModel(modelFile, song[0], rawFile, TrainData)
        DownbeatTracking.SaveInstantValue(predicts[:, 1], pathname, '_long_short')   
        DownbeatTracking.SaveInstantValue(predicts[:, 2], pathname, '_long_start')   
        DownbeatTracking.SaveInstantValue(predicts[:, 3], pathname, '_long_dur')   
        print('predicts shape', predicts.shape)

        # TrainData.GenerateLevel(predicts, pathname)

        if not useOnsetForShort:
            TrainData = TrainDataDynShortNoteBeat
            rawFile = TrainData.RawDataFileName(song[0])
            modelFile = TrainData.GetModelPathName()
            predicts = EvaluateWithModel(modelFile, song[0], rawFile, TrainData)  
            print('predicts shape', predicts.shape) 

        # TrainData.GenerateLevel(predicts, pathname)

        if useOnsetForShort:
            TrainData = TrainDataDynSinging
            rawFile = TrainData.RawDataFileName(song[0])
            modelFile = TrainData.GetModelPathName()
            predicts = EvaluateWithModel(modelFile, song[0], rawFile, TrainData, featureExtractFunc)  
            print('predicts shape', predicts.shape) 

        print('calc bpm')
        DownbeatTracking.CalcMusicInfoFromFile(pathname, debugET, debugBPM)

    if useOnsetForShort:
        # levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (song[0])
        levelEditorRoot = rootDir + 'LevelEditorForPlayer_8.0/LevelEditor_ForPlayer_8.0/'
        levelFile = '%sclient/Assets/LevelDesign/%s.xml' % (levelEditorRoot, song[0])
        duration, bpm, et = LevelInfo.LoadMusicInfo(pathname)
        
        rawFileLong = TrainDataDynLongNote.RawDataFileName(song[0])
        rawFileSinging = TrainDataDynSinging.RawDataFileName(song[0])
        with open(rawFileSinging, 'rb') as file:
            predicts = pickle.load(file)

        with open(rawFileLong, 'rb') as file:
            longPredicts = pickle.load(file)

        GenerateLevelImp(pathname, duration, bpm, et, predicts, longPredicts, levelFile, rootDir + 'data/idol_template.xml', True)
    

def LoadRawData(useSoftmax = True):
    
    songList = ['hangengxman']
    songList = ['jilejingtu']

    with open('d:/work/evaluate_data.raw', 'rb') as file:
        evaluate = pickle.load(file)
        print(type(evaluate))
        
        acceptThrehold = 0.4
        pathname = MakeMp3Pathname(songList[0])
        #for long note
        print('evaluate shape', evaluate.shape)
        predicts = evaluate[:,1] > acceptThrehold
        LevelInfo.SaveSamplesToRegionFile(predicts, pathname, '_region')
        return

        predicts = postprocess.pick(evaluate)

    # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')

        if useSoftmax:
            predicts = predicts[:,1]

        acceptThrehold = 0.7
        notes = postprocess.TrainDataToLevelData(predicts, 0, acceptThrehold)
        notes = np.asarray(notes)
        notes[0]

        pathname = MakeMp3Pathname(songList[0])
        duration, bpm, entertime = LoadMusicInfo(pathname)
        levelNotes = postprocess.ConvertToLevelNote(notes, bpm, entertime)
        levelFilename = 'd:/work/%s.xml' % (songList[0])
        LevelInfo.GenerateIdolLevel(levelFilename, levelNotes, bpm, entertime, duration)

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_predict')


# @run
def _Main():
    trainForSinging = True
    # testx, testy = PrepareTrainData(songList, batchSize)
    if trainForSinging:
        TrainData = TrainDataDynSinging
        trainx, trainy, trainSong, validateX, validateY, validateSong = LoadTrainAndValidateData()
        ValidateData = TrainDataDynSinging
        # offsetX = [[0.0] * inputDim] * int(numSteps / 2)
        # tempLabel = [0.0] * TrainData.lableDim
        # tempLabel[0] = 1.0
        # offsetY = [tempLabel] * int(numSteps / 2)
        # trainxOffset = np.concatenate((offsetX, trainx))
        # trainyOffset = np.concatenate((offsetY, trainy))
        # TrainDataOffset = TrainDataDynSinging
        # offsetData = TrainDataOffset(batchSize, numSteps, trainxOffset, trainyOffset)
    else:
        TrainData = TrainDataDynLongNote
        trainx, trainy = PrepareTrainDataFromPack(featureFile)
    
    modelPath = TrainData.GetModelPathName()    
    print('trainx frame count: ', len(trainx))
    data = TrainData(batchSize, numSteps, trainx, trainy)
    numBatches = data.numBatches
    validateData = None
    if trainForSinging:
        validateData = ValidateData(batchSize, numSteps, validateX, validateY)

    seqlenHolder = tf.placeholder(tf.int32, [None], name='seqLen')
    X = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, TrainData.lableDim), name='Y')
    learningRate = tf.placeholder(dtype=tf.float32, name='learn_rate')

    result = BuildDynamicRnn(X, Y, seqlenHolder, learningRate, TrainData)

    maxAcc = 0.0
    minLoss = 10000.0
    minTrainLoss = 10000.0
    notIncreaseCount = 0
    currentLearningRate = learning_rate
    learningRateFined = False

    learningRateList =[
        [learning_rate, 10],
        [learning_rate / 2, 15],
        [learning_rate / 4, 20],
        [learning_rate / 10, 40],
        [learning_rate / 20, 100],
        [learning_rate / 100, 500],
    ]
    currentLearningRateIdx = 0

    trainOpType = 0

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(tf.global_variables_initializer())
        SaveModel(sess, modelPath, saveMeta=True)

        initial_states = result['initial_states']
        batch_states = result['batch_states']
        initialStatesZero = np.array([[0.0] * batch_states.shape[1]] * batch_states.shape[0])

        for j in range(epch):
            loss = []
            acc = []
            classify = []
            trainLoss = []
            trainAcc = []
            trainClassify = []

            curData = data
            # if j % 2 == 1:
            #     curData = offsetData

            if trainOpType == 0:
                print('train_op')
                trainOp = result['train_op']
                lossOp = result['loss_op']
                accuracyOp = result['accuracy']
                classifyInfoOp = result['classify_info']
            
            currentInitialStates = initialStatesZero
            for i in range(curData.numBatches):
                xData, yData, seqLen = curData.GetBatch(i)
                t, l, a, classifyInfo, batchStates = sess.run([trainOp, lossOp, accuracyOp, classifyInfoOp, batch_states], 
                feed_dict={X:xData, Y:yData, seqlenHolder:seqLen, initial_states:currentInitialStates, learningRate:currentLearningRate})
                trainLoss.append(l)
                trainAcc.append(a)
                trainClassify.append(classifyInfo)
                currentInitialStates = batchStates            
            
            currentInitialStates = initialStatesZero
            if validateData is not None:
                for i in range(validateData.numBatches):
                    xData, yData, seqLen = validateData.GetBatch(i)
                    l, a, classifyInfo, batchStates = sess.run([lossOp, accuracyOp, classifyInfoOp, batch_states], 
                    feed_dict={X:xData, Y:yData, seqlenHolder:seqLen, initial_states:currentInitialStates})
                    loss.append(l)
                    acc.append(a)
                    classify.append(classifyInfo)
                    currentInitialStates = batchStates
                lossValue = sum(loss) / len(loss)
                accValue = sum(acc) / len(acc)
                classifyValue = sum(classify)

            trainLossValue = sum(trainLoss) / len(trainLoss)
            trainAccValue = sum(trainAcc) / len(trainAcc)
            trainClassifyValue = sum(trainClassify)

            notIncreaseCount += 1
            # if accValue > maxAcc:
            #     maxAcc = accValue
            #     notIncreaseCount = 0
            #     # save checkpoint
            #     print('save checkpoint')
            #     SaveModel(sess, modelPath)
            
            # if lossValue < minLoss:
            #     minLoss = lossValue
            #     notIncreaseCount = 0
            #     print('save checkpoint')
            #     SaveModel(sess, modelPath)

            if trainLossValue < minTrainLoss:
                minTrainLoss = trainLossValue
                notIncreaseCount = 0
                print('save checkpoint')
                SaveModel(sess, modelPath)

            if notIncreaseCount > learningRateList[currentLearningRateIdx][1]:
                currentLearningRateIdx += 1
                if currentLearningRateIdx >= len(learningRateList):
                    print('stop learning')
                    break
                else:
                    currentLearningRate = learningRateList[currentLearningRateIdx][0]
                    print('change learning rate', currentLearningRate)
                    notIncreaseCount = 0

            # if notIncreaseCount > 10 and not learningRateFined:
            #     currentLearningRate = currentLearningRate / 2
            #     print('change learning rate', currentLearningRate)
            #     notIncreaseCount = 0
            #     learningRateFined = True

            # if notIncreaseCount > 15 and learningRateFined:
            #     print('stop learning')
            #     break

            print('epch', j, 'train', 'loss', trainLossValue, 'accuracy', trainAccValue)
            if validateData is not None:
                print('epch', j, 'validate', 'loss', lossValue, 'accuracy', accValue)
                print(np.concatenate((trainClassifyValue, classifyValue)))
            else:
                print(trainClassifyValue)
            print('epch', j, 'not increase', notIncreaseCount)
            
            data.ShuffleBatch()

# @run
def SaveShortNoteFile(pathname=r'd:\librosa\RhythmMaster\breakfree\breakfree_4k_nm.imd'):
    # 读取节奏大师关卡，生成文件
    duration = LevelInfo.ReadRhythmMasterLevelTime(pathname)

    level = LevelInfo.LoadRhythmMasterLevel(pathname)

    numSample = duration // 10

    targetData = ConvertLevelToLables(level, numSample)    
    beat = np.max(targetData[:, 1:4], axis=1)

    # beat[:] = 0
    long = np.max(targetData[:, 4:5], axis=1)
    for i in range(1, len(long)):
        if long[i-1] == 0 and long[i] > 0:
            beat[i] = 1
    
    note = postprocess.TrainDataToLevelData(beat, 10, 0.1)
    note = note[:,0]
    print('note count', len(note))
    LevelInfo.SaveInstantValue(note, pathname, '_origshort')
    # LevelInfo.SaveInstantValue(note, pathname, '_origlong')


# @run
def GenShortNoteFromRhythmLevel():
    
    path = util.GetSamplePath()

    dirs = os.listdir(path)
    for name in dirs:
        dirname = path + name
        if not os.path.isdir(dirname):
            continue
        
        filename = util.MakeLevelPathname(name, 1)
        if os.path.exists(filename):
            SaveShortNoteFile(filename)
            print(filename, 'saved')

        filename = util.MakeLevelPathname(name, 2)
        if os.path.exists(filename):
            SaveShortNoteFile(filename)
            print(filename, 'saved')


def ReadTrainningRegion(file):

    regions = ([], [])
    for line in file:
        time, type, duration = line.split(',')
        time = float(time) * 1000
        duration = float(duration) * 1000
        type = int(type)
        regions[type].append((time, duration))
        # print(type, time, duration)
    
    return regions
        
# @run
def LoadMarkedTrainningLable():
    # 加载标记信息
    path = '/Users/xuchao/Documents/python/MusicLevelAutoGen/train'
    path = 'D:/librosa/MusicLevelAutoGen/train'
    path = rootDir + 'MusicLevelAutoGen/train'
    filelist = os.listdir(path)
    lables = ([], [])
    lableDuration = [0, 0]
    for file in filelist:
        if os.path.splitext(file)[1] != '.csv':
            continue

        pathname = path + '/' + file
        file = os.path.splitext(file)[0]
        song, level = file.split('_')
        # print(song, level)

        with open(pathname, 'r') as f:
            regions = ReadTrainningRegion(f)

        for i in range(2):
            if len(regions[i]) == 0:
                continue
            lables[i].append((song, level, regions[i]))
            for _, dur in regions[i]:
                lableDuration[i] += dur // 1000

    print(lableDuration[0])
    print(lableDuration[1])

    return lables

# featureFile = r'D:\librosa\MusicLevelAutoGen\train\beat_train_feature.raw'
# featureFile = r'D:\librosa\MusicLevelAutoGen\train\singing_train_feature.raw'
featureFile = rootDir + 'data/long_train_feature.raw'

# @run
def LoadAllTrainningLable():
    # 加载所有的标记段落，保守数据到二进制文件
    metaData = LoadMarkedTrainningLable()
    metaData = metaData[0] + metaData[1]

    print('labled song', len(metaData))

    featureData = []
    lableData = []
    index = 0
    for song, _, regions in metaData:
        print('load', index, song)
        index += 1
        pathname = MakeMp3Pathname(song)
        features = myprocesser.LoadAndProcessAudio(pathname)
        numSample = len(features)
        
        pathname = MakeLevelPathname(song, difficulty=1)
        print(pathname)
        level = LevelInfo.LoadRhythmMasterLevel(pathname)
        lable = ConvertLevelToLables(level, numSample)

        for start, length in regions:
            start = int(start // 10)
            length = int(length // 10)
            featureData.append(features[start:start+length])
            lableData.append(lable[start:start+length])
        
    featureData = np.vstack(featureData)
    lableData = np.vstack(lableData)
    print('featureData', featureData.shape, 'lableData', lableData.shape)

    with open(featureFile, 'wb') as file:
        pickle.dump(featureData, file)
        pickle.dump(lableData, file)
        print('raw file saved', featureData.shape)


midiDir = trainDataDir + 'midifiles/'
midiTrainListPath = trainDataDir + 'midi_train_list.csv'
midiValidateListPath = trainDataDir + 'midi_validate_list.csv'
# @run
def CheckValidMidi():
    fileList = os.listdir(midiDir)
    resFilePath = midiDir + 'result.csv'
    with open(resFilePath, 'w', 1, 'utf-8') as resFile:
        for name in fileList:
            arr = name.split('.')
            if len(arr) < 2:
                continue     
            if arr[1] == 'midi':
                filePath = os.path.join(midiDir, name)
                notes = LevelInfo.LoadMidi(filePath)
                if len(notes) > 0:
                    resFile.write(arr[0] + '\n')

    return True

def LoadMidiFileList(filePath):
    fileList = []
    with open(filePath, 'r', 1, 'utf-8') as f:
        for line in f:
            line = line.replace('\r', '\n')
            line = line.replace('\n', '')
            midiFileName, offset, song = line.split(',')
            fileList.append((midiFileName, float(offset), song))

    return fileList

def LoadMidiTrainAndValidateFileList():
    trainList = LoadMidiFileList(midiTrainListPath)
    validateList = LoadMidiFileList(midiValidateListPath)
    return trainList, validateList

def SaveFeaturesAndLabels(dataFilePath, features, labels):
    with open(dataFilePath, 'wb') as file:
        pickle.dump(features, file)
        pickle.dump(labels, file)
        print('raw file saved', features.shape, labels.shape)

# @run
def GenerateMidiTrainData():
    trainList, validateList = LoadMidiTrainAndValidateFileList()
    fileList = np.concatenate((trainList, validateList))
    for midiFileName, offset, song in fileList:
        print('process song:', song)
        offset = float(offset)
        midiFilePath = midiDir + midiFileName + '.midi'
        midiNotes = LevelInfo.LoadMidi(midiFilePath)
        if len(midiNotes) <= 0:
            continue
        for note in midiNotes:
            note[0] += offset
        
        dataFilePath = MakeSongDataPathName(song, 'midi_train')
        with open(dataFilePath, 'w') as dataFile:
            for note in midiNotes:
                line = ''
                for val in note:
                    line = line + str(val) + ','
                line = line[:-1]
                dataFile.write(line + '\n')

        singingStartPath = MakeMp3Pathname(song)
        singingStartTime = midiNotes[:, 0]
        LevelInfo.SaveInstantValue(singingStartTime, singingStartPath, '_singing_start')
        # DownbeatTracking.CalcMusicInfoFromFile(singingStartPath)

def LoadMidiTranData(filePath):
    notes = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.replace('\r', '\n')
            line = line.replace('\n', '')
            start, length, pitch = line.split(',')
            notes.append([float(start), float(length), float(pitch)])

    return np.array(notes)    

def MarkOnsetTimeWithMidiTime(onsetTimeArr, midiTimeArr, bpm):
    disTime = (60 / bpm / 2) * 0.9
    bgTimes = []
    for onsetTime in onsetTimeArr:
        minDis = 100000.0
        minIdx = 0
        idx = 0
        for midiTime in midiTimeArr:
            dis = abs(midiTime - onsetTime)
            if minDis >= dis:
                minDis = dis
                minIdx = idx
            else:
                break
            idx += 1

        if minDis > disTime:
            bgTimes.append(onsetTime)
        

    return midiTimeArr, np.array(bgTimes)

def PickOnsetForSingingTrain(filePath, bpm):
    onsetProcessor = madmom.features.onsets.CNNOnsetProcessor()
    onsetActivation = onsetProcessor(filePath)
    DownbeatTracking.SaveInstantValue(onsetActivation, filePath, '_midi_activation')

    threhold = 0.7
    dis_time = 60 / bpm / 4
    picker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=threhold, smooth=0.0, 
    pre_max=dis_time, post_max=dis_time, fps=100)
    onsettime = picker(onsetActivation)
    return onsettime

# @run
def MarkMidiLabel():
    trainList, validateList = LoadMidiTrainAndValidateFileList()
    fileList = np.concatenate((trainList, validateList))
    for midiFileName, offset, song in fileList:
        print('process song:', song)
        filePath = MakeMp3Pathname(song)

        duration, bpm, et = LevelInfo.LoadMusicInfo(filePath)
        onsettime = PickOnsetForSingingTrain(filePath, bpm)
        midiNotesData = LoadMidiTranData(MakeSongDataPathName(song, 'midi_train'))
        midiNotes = midiNotesData[:, 0]
        midiDuration = midiNotesData[:, 1]
        

        singTimes, bgTimes = MarkOnsetTimeWithMidiTime(onsettime, midiNotes, bpm)
        DownbeatTracking.SaveInstantValue(singTimes, filePath, '_midi_singing')
        DownbeatTracking.SaveInstantValue(bgTimes, filePath, '_midi_bg')       
        DownbeatTracking.SaveInstantValue(midiDuration, filePath, '_midi_duration')

def GenerateFeatureAndLabelByTimes(songFilePath, singingTimes, bgTimes, durations = None):
    msPerFrame = 10
    features = featureExtractFunc(songFilePath)
    numSample = len(features)
    labels = [[1.0, 0.0, 0.0, 0.0, 0.0]] * numSample
    for t in bgTimes:
        frameIndex = FrameIndex(t * 1000, msPerFrame)
        labels[frameIndex] = [0.0, 0.0, 1.0, 0.0, 0.0]
    for t in singingTimes:
        frameIndex = FrameIndex(t * 1000, msPerFrame)
        labels[frameIndex] = [0.0, 1.0, 0.0, 0.0, 0.0]

    if durations is not None:
        for idx in range(0, len(singingTimes)):
            frameIndexStart = FrameIndex(singingTimes[idx] * 1000, msPerFrame)
            frameIndexEnd = FrameIndex((singingTimes[idx] + durations[idx]) * 1000, msPerFrame)
            for subIdx in range(frameIndexStart + 1, frameIndexEnd):
                labels[subIdx] = [0.0, 0.0, 0.0, 1.0, 0.0]

    return np.array(features), np.array(labels)

# @run
def GenerateMarkedMidiFeature():
    xDatas = []
    yDatas = []

    outputCsvCount = 0
    trainList, validateList = LoadMidiTrainAndValidateFileList()
    fileList = np.concatenate((trainList, validateList))
    for midiFileName, offset, song in fileList:
        print('process song:', song)
        pathname = MakeMp3Pathname(song)
        singingTimes = DownbeatTracking.LoadInstantValue(pathname, '_midi_singing')
        bgTimes = DownbeatTracking.LoadInstantValue(pathname, '_midi_bg')
        # durations = DownbeatTracking.LoadInstantValue(pathname, '_midi_duration')

        features, labels = GenerateFeatureAndLabelByTimes(pathname, singingTimes, bgTimes)

        if outputCsvCount < 3:
            SaveFeatures(MakeSongDataPathName(song, '_feature'), features)
            outputCsvCount = outputCsvCount + 1
        SaveFeaturesAndLabels(MakeSongDataPathName(song, 'feature', '.raw'), features, labels)

        # sing = np.array(labels)
        # sing = sing[:, 1]
        # DownbeatTracking.SaveInstantValue(sing, pathname, '_s_label')
        # bg = np.array(labels)
        # bg = bg[:, 2]
        # DownbeatTracking.SaveInstantValue(bg, pathname, '_bg_label')
        # durs = np.array(labels)
        # durs = durs[:, 3]
        # DownbeatTracking.SaveInstantValue(durs, pathname, '_dur_label')

    # pureBgSongs = ['bgm_palace_memories', 'bgm_canon', 'bgm_mobile_suit', 'bgm_wuyueyu']
    # for song in pureBgSongs:
    #     pathname = MakeMp3Pathname(song)
    #     # DownbeatTracking.CalcMusicInfoFromFile(pathname)
    #     duration, bpm, et = LevelInfo.LoadMusicInfo(pathname)
    #     bgTimes = PickOnsetForSingingTrain(pathname, bpm)
    #     DownbeatTracking.SaveInstantValue(bgTimes, pathname, '_midi_bg')    
    #     features, labels = GenerateFeatureAndLabelByTimes(pathname, [], bgTimes)
    #     xDatas.append(features)
    #     yDatas.append(labels)

def LoadPureBGMFileList(loadOrigin=True):
    pureBGMDir = rootDir + 'purebgm/'
    names = os.listdir(pureBGMDir)
    fileList = []
    for name in names:
        song, ext = name.split('.')
        if ext == 'mp3' or ext == 'm4a':
            isOrigin = name.find('_vol') < 0
            if loadOrigin == isOrigin:
                fileList.append((song, ext))

    return fileList, pureBGMDir

# @run
def GeneratePureBGMFeature():
    fileList, pureBGMDir = LoadPureBGMFileList()
    for song, ext in fileList:
        print('process song:', song)
        pathname = pureBGMDir + song + '.' + ext
        bgTimes = PickOnsetForSingingTrain(pathname, 120)
        DownbeatTracking.SaveInstantValue(bgTimes, pathname, '_midi_bg')    
        features, labels = GenerateFeatureAndLabelByTimes(pathname, [], bgTimes)
        SaveFeaturesAndLabels(pureBGMDir + song + '_feature.raw', features, labels)

def LoadTrainAndValidateData(withSongOrder = False):
    trainList, validateList = LoadMidiTrainAndValidateFileList()
    trainX = []
    trainY = []
    trainSong = []
    validateX = []
    validateY = []
    validateSong = []

    for midiFileName, offset, song in trainList:
        dataFilePath = MakeSongDataPathName(song, 'feature', '.raw')
        with open(dataFilePath, 'rb') as file:
            features = pickle.load(file)
            labels = pickle.load(file)
            trainX.append(features)
            trainY.append(labels)
            trainSong.append(song)

    for midiFileName, offset, song in validateList:
        dataFilePath = MakeSongDataPathName(song, 'feature', '.raw')
        with open(dataFilePath, 'rb') as file:
            features = pickle.load(file)
            labels = pickle.load(file)
            validateX.append(features)
            validateY.append(labels)
            validateSong.append(song)

    levelTrainList = LoadLevelFileList()
    for song, offset in levelTrainList:
        dataFilePath = MakeSongDataPathName(song, 'feature', '.raw')
        with open(dataFilePath, 'rb') as file:
            features = pickle.load(file)
            labels = pickle.load(file)
            trainX.append(features)
            trainY.append(labels)
            trainSong.append(song)

    # pureBGMfileList, pureBGMDir = LoadPureBGMFileList()
    # for song, ext in pureBGMfileList:
    #     dataFilePath = pureBGMDir + song + '_feature.raw'
    #     with open(dataFilePath, 'rb') as file:
    #         features = pickle.load(file)
    #         labels = pickle.load(file)
    #         trainX.append(features)
    #         trainY.append(labels)
    #         trainSong.append(song)

    if not withSongOrder:
        trainX = np.vstack(trainX)
        trainY = np.vstack(trainY)
        validateX = np.vstack(validateX)
        validateY = np.vstack(validateY)

    return trainX, trainY, trainSong, validateX, validateY, validateSong
    
rmListPath = rootDir + 'MusicLevelAutoGen/midi/rm_train_list.csv'
# @run
def GenerateRhythmMasterTrainData():
    startTimeDic = {}
    songRegionDic = {}
    with open(rmListPath, 'r') as f:
        for line in f:
            singingStartTime = []
            line = line.replace('\r', '\n')
            line = line.replace('\n', '')
            song, start, noteType, duration = line.split(',')
            print('process song rm:', song)
            start = float(start)
            duration = float(duration)
            levelPath = MakeLevelPathname(song, difficulty=1)
            level = LevelInfo.LoadRhythmMasterLevel(levelPath)
            for note in level:
                time, type, val = note
                if type == LevelInfo.combineNode:             
                    for n in val:
                        subTime, subType, subVal = n
                        timeval = subTime / 1000.0
                        if timeval < start or timeval > start + duration:
                            continue
                        singingStartTime.append(timeval)
                else:
                    timeval = time / 1000.0
                    if timeval < start or timeval > start + duration:
                        continue
                    singingStartTime.append(timeval)

            if song in startTimeDic:
                arr = startTimeDic[song]
                arr = arr + singingStartTime
                startTimeDic[song] = arr
                songRegionDic[song].append((start, duration))
            else:
                startTimeDic[song] = singingStartTime
                songRegionDic[song] = [(start, duration)]

    xDatas = []
    yDatas = []
    for song in startTimeDic:
        print('process song feature:', song)
        songFilePath = MakeMp3Pathname(song)
        singingStartTime = startTimeDic[song]
        singingStartTime = np.array(singingStartTime)
        singingStartTime = np.unique(singingStartTime)
        DownbeatTracking.CalcMusicInfoFromFile(songFilePath)
        duration, bpm, et = LevelInfo.LoadMusicInfo(songFilePath)
        onsetTimes = PickOnsetForSingingTrain(songFilePath, bpm)
        singingTimes, bgTimes = MarkOnsetTimeWithMidiTime(onsetTimes, singingStartTime, bpm)
        
        features, labels = GenerateFeatureAndLabelByTimes(songFilePath, singingTimes, bgTimes)

        regions = songRegionDic[song]
        tempBgTimes = []
        trainFeatures = np.array([[0.0] * inputDim] * len(features))
        for start, length in regions:
            for bgTime in bgTimes:
                if bgTime >= start and bgTime < start + length:
                    tempBgTimes.append(bgTime)

            start = int(start * 100)
            length = int(length * 100)
            xDatas.append(features[start:start+length])
            yDatas.append(labels[start:start+length])

            trainFeatures[start:start+length] = features[start:start+length]

        bgTimes = tempBgTimes
        LevelInfo.SaveInstantValue(singingTimes, songFilePath, '_rm_singing')
        LevelInfo.SaveInstantValue(bgTimes, songFilePath, '_rm_bg')
        SaveFeatures(MakeSongDataPathName(song, '_feature'), trainFeatures)

    xDatas = np.vstack(xDatas)
    yDatas = np.vstack(yDatas)

    dataFilePath = rootDir + 'data/rm_singing_bg.raw'
    with open(dataFilePath, 'wb') as file:
        pickle.dump(xDatas, file)
        pickle.dump(yDatas, file)
        print('raw file saved', xDatas.shape, yDatas.shape)

def LoadLevelFileList():
    filePath = trainDataDir + 'level_train_list.csv'
    fileList = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.replace('\r', '\n')
            line = line.replace('\n', '')
            song, offset = line.split(',')
            fileList.append((song, float(offset)))

    return fileList

# @run
def GenerateLevelMarkTrainData():
    levelMarkFileDir = trainDataDir + 'level_mark_singing/'
    fileList = LoadLevelFileList()
    for song, offset in fileList:
        f = song + '.xml'
        print('process ', f)
        pathname = levelMarkFileDir + f
        notes = LevelInfo.LoadIdolInfo(pathname)
        bpm, et = LevelInfo.LoadBpmET(pathname)
        et = et * 1000.0
        notes.sort()
        for i in range(len(notes)):
            notes[i] = notes[i] + offset

        songFilePath = MakeMp3Pathname(song)
        DownbeatTracking.CalcMusicInfoFromFile(songFilePath, debugET=et, debugBPM=bpm)
        duration, bpm, et = LevelInfo.LoadMusicInfo(songFilePath)
        onsetTimes = PickOnsetForSingingTrain(songFilePath, bpm)
        singingTimes, bgTimes = MarkOnsetTimeWithMidiTime(onsetTimes, notes, bpm)

        features, labels = GenerateFeatureAndLabelByTimes(songFilePath, singingTimes, bgTimes)

        LevelInfo.SaveInstantValue(singingTimes, songFilePath, '_level_singing')
        LevelInfo.SaveInstantValue(bgTimes, songFilePath, '_level_bg')
        # SaveFeatures(MakeSongDataPathName(song, '_feature'), features)

        SaveFeaturesAndLabels(MakeSongDataPathName(song, 'feature', '.raw'), features, labels)

def SaveFeatures(filePath, features):
    with open(filePath, 'w') as file:
        for fea in features:
            line = ''
            for val in fea:
                line = line + str(val) + ','
            line = line[:-1]
            file.write(line + '\n')


def PrepareTrainDataFromPack(filePath):
    
    with open(filePath, 'rb') as file:
        xdata = pickle.load(file)
        ydata = pickle.load(file)
        print(xdata.shape, ydata.shape)
    
    return xdata, ydata

    

if __name__ == '__main__':
    
    for fun in runList:
        fun()



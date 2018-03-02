import pickle
import os
import LevelInfo
import numpy as np
import lstm.myprocesser
from lstm import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn
from lstm import postprocess
import time


runList = []
def run(r):
    runList.append(r)
    return r

LevelDifficutly = 1
numHidden = 26
batchSize = 16
numSteps = 512
numSteps = 128
inputDim = 314
outputDim = 2
learning_rate = 0.002
bSaveModel = True
ModelFile = 'd:/work/model.ckpt'

epch = 300


# for long note
songList = ['aiqingmaimai','ai', 'hellobaby', 'hongri', 'houhuiwuqi', 'huashuo', 'huozaicike', 'haodan']
songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic']
songList = ['huashuo', 'haodan']
songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 
'abracadabra', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'huozaicike', 'wangfei', 'wodemuyang']

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

def FillNote(data, note, msPerFrame):
    # 音符数据转化为sample data
    index = FrameIndex(note[0], msPerFrame)
    type = note[1]

    if type == LevelInfo.slideNote:
        data[index] = [0.0, 0.0, 1.0, 0.0]
    elif type == LevelInfo.longNote:
        last = FrameIndex(note[2], msPerFrame)
        data[index:index+last, 3] = 1.0
        # print('long note index %d last %d' % (index, last))
    else:        
        data[index] = [0.0, 1.0, 0.0, 0.0]


def ConvertLevelToLables(level, numframes, msMinInterval = 10):
    '''
    关卡数据转换训练数据
    关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    尝试每个样本三个输出，代表是否有该类型音符，（点击、滑动、长按）
    '''
    frames = [[1.0, 0.0, 0.0, 0.0]] * numframes
    frames = np.array(frames)

    maxResidual = 0
    for note in level:
        time, type, val = note
        if type == LevelInfo.combineNode:             
            for n in val:
                FillNote(frames, n, msMinInterval)
            continue
        
        FillNote(frames, note, msMinInterval)

    return frames


def SamplesToSoftmax(samples):
    '''
    将样本数据转换为分类数据
    '''
    assert samples.ndim == 1
    s1 = 1 - samples
    return np.stack((s1, samples), axis=1)



def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
    return path


def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.mp3' % (path, song, song)
    return pathname

def MakeLevelPathname(song, difficulty=2):
    path = GetSamplePath()
    diff = ['ez', 'nm', 'hd']
    pathname = '%s%s/%s_4k_%s.imd' % (path, song, song, diff[difficulty])
    return pathname   


def PrepareTrainData(songList, batchSize = 32, loadTestData = True):   
    trainx = []
    trainy = []
    for song in songList:
        pathname = MakeMp3Pathname(song)
        inputData = lstm.myprocesser.LoadAndProcessAudio(pathname)
        trainx.append(inputData)
        numSample = len(inputData)

        if loadTestData:
            pathname = MakeLevelPathname(song, difficulty=LevelDifficutly)
            level = LevelInfo.LoadRhythmMasterLevel(pathname)

            targetData = ConvertLevelToLables(level, numSample)
                
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


# class TrainData():
#     def __init__(self, x, y, batchSize, numSteps):
#         assert x.ndim == 2
#         self.batchSize = batchSize
#         self.numSteps = numSteps

#         count = x.shape[0]
#         self.numBatches = count // (self.batchSize * self.numSteps)
#         print('numbatches', self.numBatches)

#         count = self.numBatches * (self.batchSize * self.numSteps)
#         x = x[:count]
#         y = y[:count]

#         xDim = x.shape[1]
#         yDim = y.shape[1]

#         # 重组数据，数据项 shape=(step, batchsize, inputsize)
#         x = x.reshape(self.batchSize, self.numBatches, self.numSteps, xDim)
#         self._x = x.transpose(1, 2, 0, 3)

#         y = y.reshape(self.batchSize, self.numBatches, self.numSteps, yDim)
#         self._y = y.transpose(1, 2, 0, 3)
#         print('y shape', self._y.shape)


#     def GetBatch(self, n):
#         x = self._x[n]
#         y = self._y[n]
#         return x, y



class TrainDataDynShortNote():
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if y != None and len(y) != 0:
            y = y[:, 1]
            y = SamplesToSoftmax(y)

            y = y[:count]
            yDim = y.shape[1]
            y = y.reshape(self.batchSize, -1, self.numSteps, yDim)
            self._y = y.transpose(1, 0, 2, 3)
            print('y shape', self._y.shape)

            self.numBatches = count // (self.batchSize * self.numSteps)
            print('numbatches', self.numBatches)

            count = self.numBatches * (self.batchSize * self.numSteps)

            x = x[:count]
            xDim = x.shape[1]
            x = x.reshape(self.batchSize, -1, self.numSteps, xDim)
            self._x = x.transpose(1, 0, 2, 3)
            print('x shape', self._x.shape)

        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 


    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        seqLen = np.array([self.numSteps] * self.batchSize)
        return x, y, seqLen

    
    def GenerateLevel(self, sess, prediction, X, seqLenHolder, pathname):      
        
        evaluate = []
        for i in range(self.numBatches):
            xData, _, seqLen = self.GetBatch(i)
            t = sess.run(prediction, feed_dict={X:xData, seqLenHolder:seqLen})
            t = t[0:numSteps,:]
            evaluate.append(t)

        evaluate = np.stack(evaluate)
        evaluate = evaluate.reshape(-1, outputDim)
        print('evaluate', evaluate.shape)
        
        acceptThrehold = 0.2
        # #for long note
        # print('evaluate shape', evaluate.shape)
        # predicts = evaluate[:,1] > acceptThrehold
        # LevelInfo.SaveSamplesToRegionFile(predicts, pathname, '_region')
        # return

        predicts = postprocess.pick(evaluate, kernelSize=11)
        print('predicts', predicts.shape)

        # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')

        predicts = predicts[:,1]

        notes = postprocess.TrainDataToLevelData(predicts, 10, acceptThrehold, 0)
        print('gen notes number', len(notes))
        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst')

        # LevelInfo.SaveInstantValue(notes, pathname, '_region')
    



class TrainDataDynLongNote():
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        if (not y is None):
            y = y[:, 3]

            ySum = np.sum(y)
            print('y sample count', len(y), 'y sum', ySum)

            y = SamplesToSoftmax(y)

            self.numBatches = count // (self.batchSize * self.numSteps)
            print('numbatches', self.numBatches)
            count = self.numBatches * (self.batchSize * self.numSteps)

            y = y[:count]
            yDim = y.shape[1]
            y = y.reshape(self.batchSize, -1, self.numSteps, yDim)
            self._y = y.transpose(1, 0, 2, 3)
            print('y shape', self._y.shape)

            x = x[:count]
            xDim = x.shape[1]
            x = x.reshape(self.batchSize, -1, self.numSteps, xDim)
            self._x = x.transpose(1, 0, 2, 3)
            print('x shape', self._x.shape)

        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 


    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        seqLen = np.array([self.numSteps] * self.batchSize)
        return x, y, seqLen

    
    def GenerateLevel(self, sess, prediction, X, seqLenHolder, pathname):      
        
        evaluate = []
        for i in range(self.numBatches):
            xData, _, seqLen = self.GetBatch(i)
            t = sess.run(prediction, feed_dict={X:xData, seqLenHolder:seqLen})
            t = t[0:numSteps,:]
            evaluate.append(t)

        evaluate = np.stack(evaluate)
        predicts = evaluate.reshape(-1, outputDim)
        print('evaluate', evaluate.shape)
        
        acceptThrehold = 0.3
        # #for long note
        # print('evaluate shape', evaluate.shape)
        # predicts = evaluate[:,1] > acceptThrehold
        # LevelInfo.SaveSamplesToRegionFile(predicts, pathname, '_region')
        # return

        # predicts = postprocess.pick(evaluate, kernelSize=11)
        print('predicts', predicts.shape)

        # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')

        predicts = predicts[:,1]

        notes = postprocess.TrainDataToLevelData(predicts, 10, acceptThrehold, 0)
        print('gen notes number', len(notes))
        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_region')

        # LevelInfo.SaveInstantValue(notes, pathname, '_region')
    



TrainData = TrainDataDynLongNote



def BuildDynamicRnn(X, Y, seqlen, learningRate):
    t = time.time()

    weights = tf.Variable(tf.random_normal(shape=[2*numHidden, outputDim]))
    bais = tf.Variable(tf.random_normal(shape=[outputDim]))
    
    numLayers = 3
    cells = []
    dropoutCell = []
    for i in range(numLayers * 2):
        c = rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)
        cells.append(c)
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.8)
        dropoutCell.append(c)
    
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        dropoutCell[0:numLayers], dropoutCell[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    print('output', output)
    outlayerDim = tf.shape(output)[2]

    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])
    print('output', output)

    logits = tf.matmul(output, weights) + bais

    Y = tf.reshape(Y, [batchSize*numSteps, outputDim])

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_op = tf.reduce_mean(crossEntropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)
    print('train op', time.time() - t)    

    correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # prediction without dropout
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        cells[0:numLayers], cells[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    # output = tf.reshape(output, [batchSize*maxSeqLen, outlayerDim])
    
    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])
    logits = tf.matmul(output, weights) + bais

    prediction = tf.nn.softmax(logits, name='prediction')

    print('ready to train')

    return train_op, loss_op, accuracy, prediction

   

def SaveModel(sess, saveMeta = False):
    saver = tf.train.Saver()
    saver.save(sess, ModelFile, write_meta_graph=saveMeta)
    print('model saved')


@run
def GenerateLevel():

    print('gen level')

    song = ['bboombboom']
    song = ['jilejingtu']
    pathname = MakeMp3Pathname(song[0])
    
    saver = tf.train.import_meta_graph("d:/work/model.ckpt.meta")

    with tf.Session() as sess:
        
        saver.restore(sess, ModelFile)
        print('model loaded')

        prediction = tf.get_default_graph().get_tensor_by_name("prediction:0")
        print('prediction', prediction)         
        X = tf.get_default_graph().get_tensor_by_name('X:0')
        seqLenHolder = tf.get_default_graph().get_tensor_by_name('seqLen:0')
    
        testx, _ = PrepareTrainData(song, batchSize, loadTestData = False)

        data = TrainData(batchSize, numSteps, testx)

        data.GenerateLevel(sess, prediction, X, seqLenHolder, pathname)
    

# @run
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


def LoadMusicInfo(filename):
    '''
    读取歌曲的长度,bpm,entertime等信息
    '''
    # filename = r'd:\librosa\RhythmMaster\jilejingtu\jilejingtu.mp3'
    dir = os.path.dirname(filename) + os.path.sep
    filename = dir + 'info.txt'
    with open(filename, 'r') as file:
        value = [float(s.split('=')[1]) for s in file.readlines()]
        
        # duration, bpm, entertime
        value[0] = int(value[0] * 1000)
        value[2] = int(value[2] * 1000)
        # print(value)
        return tuple(value)


# @run
def _Main():
    
    testx, testy = PrepareTrainData(songList, batchSize)
    print('test shape', testx.shape)
    
    data = TrainData(batchSize, numSteps, testx, testy)
    numBatches = data.numBatches
    print('numbatchs', numBatches)
    
    seqlenHolder = tf.placeholder(tf.int32, [None], name='seqLen')
    X = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, outputDim), name='Y')
    learningRate = tf.placeholder(dtype=tf.float32, name='learn_rate')

    train_op, loss_op, accuracy, prediction = BuildDynamicRnn(X, Y, seqlenHolder, learningRate)

    init = tf.global_variables_initializer()

    maxAcc = 0.0
    notIncreaseCount = 0
    currentLearningRate = learning_rate
    learningRateFined = False
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        SaveModel(sess, saveMeta=True)

        for j in range(epch):
            loss = []
            acc = []

            for i in range(numBatches):
                xData, yData, seqLen = data.GetBatch(i)

                if i % 6 == 0:
                    l, a = sess.run([loss_op, accuracy], feed_dict={X:xData, Y:yData, seqlenHolder:seqLen})
                    loss.append(l)
                    acc.append(a)
                else:
                    t = sess.run(train_op, feed_dict={X:xData, Y:yData, seqlenHolder:seqLen, learningRate:currentLearningRate})
                
            lossValue = sum(loss) / len(loss)
            accValue = sum(acc) / len(acc)

            notIncreaseCount += 1
            if accValue > maxAcc:
                maxAcc = accValue
                notIncreaseCount = 0
                # save checkpoint
                print('save checkpoint')
                SaveModel(sess)
            
            if notIncreaseCount > 10 and not learningRateFined:
                currentLearningRate = currentLearningRate / 2
                print('change learning rate', currentLearningRate)
                notIncreaseCount = 0
                learningRateFined = True

            if notIncreaseCount > 15 and learningRateFined:
                print('stop learning')
                break
            
            print('epch', j, 'loss', lossValue, 'accuracy', accValue, 'not increase', notIncreaseCount)

        # saver = tf.train.Saver()
        # saver.restore(sess, ModelFile)
        # print('checkpoint loaded')
        # GenerateLevel(sess, prediction, X, seqlenHolder)


if __name__ == '__main__':
    
    for fun in runList:
        fun()


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
batchSize = 8
numSteps = 512
numSteps = 128
inputDim = 314
outputDim = 2
learning_rate = 0.0006
bSaveModel = True
ModelFile = 'd:/work/model.ckpt'

epch = 300


# for long note
songList = ['aiqingmaimai','ai', 'hellobaby', 'hongri', 'houhuiwuqi', 'huashuo', 'huozaicike', 'haodan']
songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic']
songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 
'abracadabra', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'huozaicike', 'wangfei', 'wodemuyang']
# songList = ['4minuteshm', 'hangengxman']

testing = False
if testing:
    numSteps = 64
    epch = 3
    songList = ['aiqingmaimai','ai']



def FillLongNote(data, combineNote):
    begin = combineNote[0][0]
    end = 0
    for note in combineNote:
        time = note[0]
        last = time + note[2]
        if time < begin:
            begin = time
        if last > end:
            end = last
    data[[begin//10, end//10], 3] = 1.0

def FillNote(data, note):
    # 音符数据转化为sample data
    index = note[0] // 10
    type = note[1]

    if type == LevelInfo.slideNote:
        data[index] = [0.0, 0.0, 1.0, 0.0]
    elif type == LevelInfo.longNote:
        last = note[2] // 10
        # data[index:index+last, 3] = 1.0
        # 尝试标记长音符的开头和结尾
        data[[index, index+last], 3] = 1.0
    else:        
        data[index] = [0.0, 1.0, 0.0, 0.0]

def SamplesToSoftmax(samples):
    '''
    将样本数据转换为分类数据
    '''
    s1 = 1 - samples
    return np.stack((s1, samples), axis=1)


def LevelDataToTrainData(LevelData, numSamples):
    '''
    关卡数据转换训练数据
    关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    尝试每个样本三个输出，代表是否有该类型音符，（点击、滑动、长按）
    '''
    result = [[1.0, 0.0, 0.0, 0.0]] * numSamples
    result = np.array(result)
    for l in LevelData:
        # time in ms, sample rate is 100 samples per second
        type = l[1]
        if type == 3:
            notes = l[2]
            FillLongNote(result, notes)
            # for n in notes:
            #     FillNote(result, n)
            continue

        FillNote(result, l)

    return result


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

            targetData = LevelDataToTrainData(level, numSample)
                
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


class TrainData():
    def __init__(self, x, y, batchSize, numSteps):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        self.numBatches = count // (self.batchSize * self.numSteps)
        print('numbatches', self.numBatches)

        count = self.numBatches * (self.batchSize * self.numSteps)
        x = x[:count]
        y = y[:count]

        xDim = x.shape[1]
        yDim = y.shape[1]

        # 重组数据，数据项 shape=(step, batchsize, inputsize)
        x = x.reshape(self.batchSize, self.numBatches, self.numSteps, xDim)
        self._x = x.transpose(1, 2, 0, 3)

        y = y.reshape(self.batchSize, self.numBatches, self.numSteps, yDim)
        self._y = y.transpose(1, 2, 0, 3)
        print('y shape', self._y.shape)


    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        return x, y

class TrainDataDyn():
    def __init__(self, x, y, batchSize, numSteps):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        self.numBatches = count // (self.batchSize * self.numSteps)
        print('numbatches', self.numBatches)

        count = self.numBatches * (self.batchSize * self.numSteps)
        x = x[:count]
        y = y[:count]

        xDim = x.shape[1]
        yDim = y.shape[1]

        x = x.reshape(self.batchSize, -1, self.numSteps, xDim)
        self._x = x.transpose(1, 0, 2, 3)
        print('x shape', self._x.shape)

        y = y.reshape(self.batchSize, -1, self.numSteps, yDim)
        self._y = y.transpose(1, 0, 2, 3)
        print('y shape', self._y.shape)


    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        seqLen = np.array([self.numSteps] * self.batchSize)
        return x, y, seqLen




def BuildNetwork(X, Y):
    t = time.time()

    x = tf.unstack(X, axis=0)
    print('unstack', time.time() - t)

    # Define lstm cells with tensorflow
    # Forward direction cell

    numLayers = 3
    cells = []
    dropoutCell = []
    for i in range(numLayers * 2):
        c = rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)
        cells.append(c)
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.9)
        dropoutCell.append(c)
    
    output, _, _ = rnn.stack_bidirectional_rnn(dropoutCell[0:numLayers], dropoutCell[numLayers:], x, dtype=tf.float32)
    print('stack_bidirectional_rnn', time.time() - t)
    
    weights = tf.Variable(tf.random_normal(shape=[2 * numHidden, outputDim]))
    bais = tf.Variable(tf.random_normal(shape=[outputDim]))

    logits = [tf.matmul(o, weights) + bais for o in output]
    logits = tf.stack(logits)
    # print(logits)

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_op = tf.reduce_mean(crossEntropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)
    print('train op', time.time() - t)    

    correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=2), tf.argmax(Y, axis=2))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # prediction without dropout
    output, _, _ = rnn.stack_bidirectional_rnn(cells[0:numLayers], cells[numLayers:], x, dtype=tf.float32)
    logits = [tf.matmul(o, weights) + bais for o in output]
    logits = tf.stack(logits)
    prediction = tf.nn.softmax(logits, name='prediction')

    print('ready to train')

    return train_op, loss_op, accuracy, prediction


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


# @run
def Test():

    state_size = 10
    num_layers = 3
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    print(cell)
    return

    x = np.arange(1, 25)
    x = x.reshape(24, 1)
    x = np.hstack((x, x, x))
    print(x.shape)

    x = x.reshape(3, 4, 2, 3)
    x = x.transpose(1, 2, 0, 3)   
    x = x.transpose(2, 0, 1, 3)   
    x = x.reshape(24, 3)

    
    with open('d:/work/evaluate_data.raw', 'wb') as file:
        pickle.dump(x, file)

    # x = x.reshape(3, 2 * 4, 3)
    # x = x.transpose(1, 0, 2)

    # x = x.reshape(4, 2, 3, 3)
   


def SaveModel(sess, saveMeta = False):
    saver = tf.train.Saver()
    saver.save(sess, ModelFile, write_meta_graph=saveMeta)
    print('model saved')

@run
def LoadModel():
    
    saver = tf.train.import_meta_graph("d:/work/model.ckpt.meta")

    with tf.Session() as sess:
        
        saver.restore(sess, ModelFile)
        print('model loaded')

        predict = tf.get_default_graph().get_tensor_by_name("prediction:0")
        print('tensor',predict)
         
        X = tf.get_default_graph().get_tensor_by_name('X:0')
        seqLen = tf.get_default_graph().get_tensor_by_name('seqLen:0')

        GenerateLevel(sess, predict, X, seqLen)
        


def GenerateLevel(sess, prediction, X, seqLenHolder):
    
    print('gen level')

    song = ['bboombboom']
    song = ['jilejingtu']
    testx, _ = PrepareTrainData(song, batchSize, loadTestData = False)
        
    count = len(testx)

    testx = testx[:-(count%numSteps)]
    testx = testx.reshape(-1, 1, numSteps, inputDim)
    testx = np.repeat(testx, batchSize, axis=1)
    print('testx', testx.shape)

    numBatches = len(testx)
    print('numbatch', numBatches) 

    seqLen = [numSteps] * batchSize
    
    evaluate = []
    for i in range(numBatches):
        xData = testx[i]
        t = sess.run(prediction, feed_dict={X:xData, seqLenHolder:seqLen})
        t = t[0:numSteps,:]
        evaluate.append(t)

    evaluate = np.stack(evaluate)
    evaluate = evaluate.reshape(-1, outputDim)
    print('evaluate', evaluate.shape)

    
    acceptThrehold = 0.2
    pathname = MakeMp3Pathname(song[0])
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

    # print('pick long note')
    testy = testy[:, 1]
    testy = SamplesToSoftmax(testy)
    
    data = TrainDataDyn(testx, testy, batchSize, numSteps)
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

        saver = tf.train.Saver()
        saver.restore(sess, ModelFile)
        print('checkpoint loaded')
        GenerateLevel(sess, prediction, X, seqlenHolder)


if __name__ == '__main__':
    
    for fun in runList:
        fun()



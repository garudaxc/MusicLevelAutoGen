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


numHidden = 24
batchSize = 6
numSteps = 256
# numSteps = 128
inputDim = 314
outputDim = 2
learning_rate = 0.001


def FillNote(data, note):
    # 音符数据转化为sample data
    index = note[0] // 10
    type = note[1]

    if type == LevelInfo.slideNote:
        data[index] = [0.0, 0.0, 1.0, 0.0]
    elif type == LevelInfo.longNote:
        last = note[2] // 10
        data[index:index+last, 3] = 1.0
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
            for n in notes:
                FillNote(result, n)
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

def MakeLevelPathname(song, difficulty=1):
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
            pathname = MakeLevelPathname(song)
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
    y = y > acceptThrehold
    LevelInfo.SaveSamplesToRegionFile(y, pathname, 'region')
    return    

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
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.6)
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



# @run
def TestBuildTime():
    
    X = tf.placeholder(dtype=tf.float32, shape=(10, batchSize, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(10, batchSize, outputDim), name='Y')
    train_op, loss_op, accuracy, prediction = BuildNetwork(X, Y)


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
   


def SaveModel(sess):
    saver = tf.train.Saver()
    saver.save(sess, 'd:/work/model.ckpt')
    print('model saved')

# @run
def LoadModel():
    
    saver = saver = tf.train.import_meta_graph("d:/work/model.ckpt.meta")

    with tf.Session() as sess:
        
        saver.restore(sess, 'd:/work/model.ckpt')
        print('model loaded')

        predict = tf.get_default_graph().get_tensor_by_name("prediction:0")
        print('tensor',predict)
         
        X = tf.get_default_graph().get_tensor_by_name('X:0')
        # X = tf.get_default_graph().get_operation_by_name('X')

        GenerateLevel(sess, predict, X)
        


# @run
def GenerateLevelTest(sess=None, train_op=None):
    useSoftmax = True

    songList = ['4minuteshm']
    testx, testy = PrepareTrainData(songList, batchSize)
    print('testy', testy.shape)


    testx = np.repeat(testx, batchSize, axis=0)
    count = len(testx)
    yu = count % (batchSize * numSteps)
    testx = testx[0:-yu]

    testx = testx.reshape(-1, numSteps, batchSize, inputDim)
    print('num batches', len(testx))

    b = testx[2]
    print('batch shape', b.shape)

    print('step 0, batch 0 1', b[0, 0, 0:2], b[0, 1, 0:2])
    
    print('step 1 2, batch 0', b[1, 0, 0:2], b[2, 0, 0:2])    

    return

    
    data = TrainData(testx, testy, batchSize, numSteps)

    print('numbatch', data.numBatches)
    
    X = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, inputDim))
    Y = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, outputDim))
    prediction = BuildNetwork(X, Y)

    if sess == None:
        sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())

    evaluate = []
    for i in range(data.numBatches):
        xData, yData = data.GetBatch(i)
        t = sess.run(prediction, feed_dict={X:xData, Y:yData})
        evaluate.append(t)

    evaluate = np.array(evaluate)
    print('result', evaluate.shape)

    evaluate = evaluate.transpose(2, 0, 1, 3)
    predicts = np.reshape(evaluate, (-1, 2))
    
    predicts = postprocess.pick(predicts)

    postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')

    if useSoftmax:
        predicts = predicts[:,1]

    acceptThrehold = 0.5
    notes = postprocess.TrainDataToLevelData(predicts, 0, acceptThrehold)
    notes = np.asarray(notes)
    notes[0]
    notes = notes[:,0]

    return notes
    

def GenerateLevel(sess, prediction, X):
    
    print('gen level')

    useSoftmax = True
    saveRawData = True

    songList = ['4minuteshm']
    songList = ['hangengxman']
    songList = ['jilejingtu']
    testx, _ = PrepareTrainData(songList, batchSize, loadTestData = False)
    # print('testy', testy.shape)    

    testx = np.repeat(testx, batchSize, axis=0)
    count = len(testx)
    yu = count % (batchSize * numSteps)
    testx = testx[0:-yu]

    testx = testx.reshape(-1, numSteps, batchSize, inputDim)
    numBatches = len(testx)

    print('numbatch', numBatches)  
    # if sess == None:
    #     sess = tf.Session()
    
    evaluate = []
    for i in range(numBatches):
        xData = testx[i]
        t = sess.run(prediction, feed_dict={X:xData})
        evaluate.append(t)

    evaluate = np.array(evaluate)
    evaluate = evaluate.reshape(-1, batchSize, 2)
    evaluate = evaluate[:, 0, :]
    
    if saveRawData:
        with open('d:/work/evaluate_data.raw', 'wb') as file:
            pickle.dump(evaluate, file)


    acceptThrehold = 0.6
    pathname = MakeMp3Pathname(songList[0])
    #for long note
    print('evaluate shape', evaluate.shape)
    predicts = evaluate[:,1] > acceptThrehold
    LevelInfo.SaveSamplesToRegionFile(prediction, pathname, '_region')
    return


    predicts = postprocess.pick(evaluate)

    # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')

    if useSoftmax:
        predicts = predicts[:,1]

    acceptThrehold = 0.6
    notes = postprocess.TrainDataToLevelData(predicts, 0, acceptThrehold)
    notes = np.asarray(notes)
    notes[0]
    notes = notes[:,0]

    LevelInfo.SaveInstantValue(notes, pathname, '_predict')
    

# @run
def LoadRawData(useSoftmax = True):
    
    songList = ['hangengxman']
    songList = ['jilejingtu']

    with open('d:/work/evaluate_data.raw', 'rb') as file:
        evaluate = pickle.load(file)
        print(type(evaluate))
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
    # filename = r'd:\librosa\RhythmMaster\jilejingtu\jilejingtu.mp3'
    dir = os.path.dirname(filename) + os.path.sep
    filename = dir + 'info.txt'
    with open(filename, 'r') as file:
        value = [float(s.split('=')[1]) for s in file.readlines()]
        
        # duration, bpm, entertime
        value[0] = int(value[0] * 1000)
        value[2] = int(value[2] * 1000)
        print(value)
        return tuple(value)


@run
def Run():
    
    useSoftmax = True

    songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic', 'aiqingkele']
    # for long note
    songList = ['aiqingmaimai','ai', 'hellobaby', 'hongri', 'houhuiwuqi', 'huashuo', 'huozaicike', 'haodan']
    # songList = ['4minuteshm', 'hangengxman']

    testx, testy = PrepareTrainData(songList, batchSize)
    print('test shape', testx.shape)

    print('pick long note')
    testy = testy[:, 3]
    testy = SamplesToSoftmax(testy)
    
    data = TrainData(testx, testy, batchSize, numSteps)
    numBatches = data.numBatches
    print('numbatchs', numBatches)
    
    X = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, outputDim), name='Y')
    train_op, loss_op, accuracy, prediction = BuildNetwork(X, Y)

    # Initialize the variables (i.e. assign their default value)

    init = tf.global_variables_initializer()

    maxAcc = 0.0
    notIncreaseCount = 0
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        epch = 200

        for j in range(epch):
            loss = []
            acc = []

            for i in range(numBatches):
                xData, yData = data.GetBatch(i)

                if i % 6 == 0:
                    l, a = sess.run([loss_op, accuracy], feed_dict={X:xData, Y:yData})
                    loss.append(l)
                    acc.append(a)
                else:
                    t = sess.run(train_op, feed_dict={X:xData, Y:yData})
    
            lossValue = sum(loss) / len(loss)
            accValue = sum(acc) / len(acc)

            notIncreaseCount += 1
            if accValue > maxAcc:
                maxAcc = accValue
                notIncreaseCount = 0
            
            if notIncreaseCount > 15:
                break            
            print('epch', j, 'loss', lossValue, 'accuracy', accValue, 'not increase', notIncreaseCount)

        SaveModel(sess)
        GenerateLevel(sess, prediction, X)


if __name__ == '__main__':
    # Test()
    # BuildNetwork()

    # Run()
    for fun in runList:
        fun()




import pickle
import os
import LevelInfo
import numpy as np
import lstm.myprocesser
from lstm import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn




numHidden = 20
batchSize = 4
numSteps = 256
inputDim = 314
outputDim = 2
learning_rate = 0.005



def FillNote(data, note):
    index = note[0] // 10
    type = note[1]

    data[index, 0] = 1.0
    if type == 1:
        data[index, 1] = 1.0
    if type == 2:
        last = note[2] // 10
        data[index:index+last, 2] = 1.0


def LevelDataToTrainData(LevelData, numSamples):
    # 关卡数据转换训练数据
    # 关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    # 尝试每个样本三个输出，代表是否有该类型音符，（点击、滑动、长按）
    result = [[0.0, 0.0, 0.0]] * numSamples
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

    result = result[:,0:1]
    return result

def LevelToTrainDataSoftmax(levelData, numSamples):
    result = LevelDataToTrainData(levelData, numSamples)
    r = []
    dim = result.shape[1]
    for i in range(numSamples):
        if result[i, 0] == 1.0:
            r.append([0.0, 1.0])
        else:
            r.append([1.0, 0.0])
    r = np.array(r)
    return r


def LoadLevelData(pathname, numSample):
    #加载心动关卡，转为训练数据
    notes = LevelInfo.LoadIdolInfo(pathname)
    print('load form %s with %d notes' % (pathname, len(notes)))

    lastNote = notes[-1]
    if lastNote * 100 > numSample:
        print('level lenth %f not match sample count %d' % (lastNote, numSample))
    
    sample = [0.0] * numSample
    for t in notes:
        index = int(np.floor(t * 100 + 0.5))
        sample[index] = 1.0

    return sample

def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
    return path


def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.mp3' % (path, song, song)
    return pathname

def MakeLevelPathname(song, difficulty=0):
    path = GetSamplePath()
    pathname = '%s%s/%s_4k_nm.imd' % (path, song, song)
    return pathname   


def PrepareTrainData(songList, batchSize = 32, useSoftmax = False):
   
    trainx = []
    trainy = []
    for song in songList:
        pathname = MakeMp3Pathname(song)
        inputData = lstm.myprocesser.LoadAndProcessAudio(pathname)
        trainx.append(inputData)
        numSample = len(inputData)
        pathname = MakeLevelPathname(song)
        level = LevelInfo.LoadRhythmMasterLevel(pathname)

        if useSoftmax:
            targetData = LevelToTrainDataSoftmax(level, numSample)
        else:
            targetData = LevelDataToTrainData(level, numSample)
        trainy.append(targetData)

    trainx = np.vstack(trainx)
    trainy = np.vstack(trainy)

    # if not useSoftmax:
    #     trainy = trainy[:,0]

    n = len(trainx) % batchSize
    trainx = trainx[:-n]
    trainy = trainy[:-n]

    return trainx, trainy


class TrainData():
    def __init__(self, x, y, batchSize, numSteps):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        self.numBatches = count // (self.batchSize * self.numSteps)
        print(self.numBatches)

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


    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        return x, y


def BuildNetwork(X, Y):
    x = tf.unstack(X, axis=0)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = [
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)]

    # Backward direction cell
    lstm_bw_cell = [
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0),
        rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)]

    # Get lstm cell output
    
    output, _, _ = rnn.stack_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    
    weights = tf.Variable(tf.random_normal(shape=[2 * numHidden, outputDim]))
    bais = tf.Variable(tf.random_normal(shape=[outputDim]))

    logits = [tf.matmul(o, weights) + bais for o in output]
    logits = tf.stack(logits)
    # print(logits)

    prediction = tf.nn.softmax(logits)
    print(prediction)

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    print(crossEntropy)
    loss_op = tf.reduce_mean(crossEntropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    print(train_op)

    return train_op, loss_op

    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #     logits=logits, labels=Y))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss_op)

    # print(out[0])


def Test():
    timestep = 128
    num_input = 314
    num_hidden = 10
    batch_size = 8

    x = tf.placeholder(dtype=tf.float32, shape=(timestep, batch_size, num_input))
    x = tf.unstack(x, axis=0)


    print(x)
    return

    # x = tf.Variable(validate_shape=(batch_size, num_input), dtype=tf.float32)
    # x = tf.random_normal(shape=(batch_size, num_input))

    x = [x]

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)

    # Get lstm cell output
    
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    print(outputs)
    print(outputs[0])

    weight = tf.Variable(tf.random_normal(shape=(2*num_hidden, 2), dtype=tf.float32))
    out = tf.matmul(outputs[-1], weight)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        y = sess.run(out)
        print(y)

        y = sess.run(out)
        print(y)

        
def Run():
    
    useSoftmax = True

    songList = ['4minuteshm', '2differenttears', 'abracadabra']
    testx, testy = PrepareTrainData(songList, batchSize, useSoftmax)
    
    data = TrainData(testx, testy, batchSize, numSteps)
    numBatches = data.numBatches
    print('numbatchs', numBatches)    
    
    X = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, inputDim))
    Y = tf.placeholder(dtype=tf.float32, shape=(numSteps, batchSize, outputDim))
    train_op, loss_op = BuildNetwork(X, Y)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        epch = 100

        for j in range(epch):
            loss = 0

            for i in range(numBatches):
                xData, yData = data.GetBatch(i)

                if i % 10 == 0:
                    l = sess.run(loss_op, feed_dict={X:xData, Y:yData})
                    loss = loss + l
                else:
                    sess.run(train_op, feed_dict={X:xData, Y:yData})
    
            print('epch', j, 'loss', loss)

if __name__ == '__main__':
    # Test()
    # BuildNetwork()

    Run()


    # x, y = PrepareTrainData(['4minuteshm'], batchSize = 32, useSoftmax=True)
    # print(x.shape, y.shape)

    # a = TrainData(x, y, batcSize, numSteps)

    # x, y = a.GetBatch(1)
    # print(x.shape)
    # print(y.shape)


    

    # x = np.arange(1, 25)
    # x = x.reshape(24, 1)
    # x = np.hstack((x, x, x))

    # x = x.reshape(3, 4, 2, 3)
    # x = x.transpose(1, 2, 0, 3)   

    # x = x.reshape(3, 2 * 4, 3)
    # x = x.transpose(1, 0, 2)

    # x = x.reshape(4, 2, 3, 3)
    # print(x)



import pickle
import os
import LevelInfo
import numpy as np
import lstm.myprocesser
from lstm import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn




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

    trainx = trainx.reshape(trainx.shape[0], 1, trainx.shape[1])

    return trainx, trainy


def BuildNetwork(x):
    
    num_hidden = 25

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)

    # Get lstm cell output
    
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)

    print(outputs)


def Test():
    timestep = 1
    num_input = 15
    num_hidden = 10
    batch_size = 5

    x = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_input))
    x = tf.Variable(validate_shape=(batch_size, num_input), dtype=tf.float32)
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

    weight = tf.Variable(validate_shape=(2*num_hidden, 2), dtype=tf.float32)
    out = tf.matmul(outputs[-1], weight)


    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        y = sess.run(out)
        print(y)


        

def Run():
    batch_size = 512
    useSoftmax = False
    units = 30
    epochs = 120
    
    songList = ['4minuteshm']
    pathname = MakeMp3Pathname(songList[0])
    testx, testy = PrepareTrainData(songList, batch_size, useSoftmax)



if __name__ == '__main__':
    Test()
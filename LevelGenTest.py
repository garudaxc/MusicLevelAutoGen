# coding=UTF-8
import pickle
import os
import LevelInfo
import numpy as np
import lstm.myprocesser
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error


def LoadInputData(pathname):    
    d = None
    with open(pathname, 'rb') as file:
        d = pickle.load(file)
    
    print('read file ', pathname, type(d), d.shape)
    return d


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

    if not useSoftmax:
        trainy = trainy[:,0]

    n = len(trainx) % batchSize
    trainx = trainx[:-n]
    trainy = trainy[:-n]

    trainx = trainx.reshape(trainx.shape[0], 1, trainx.shape[1])

    return trainx, trainy


def BuildLSTM(data_dim = 16, batch_size = 64, fitSize = 1, useSoftmax = False):
    
    timesteps = 1
    units = 30
    stateful = False

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful,
                ), batch_input_shape=(batch_size, timesteps, data_dim)))

    # model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful,
    #             ), input_shape=(timesteps, data_dim)))

    #model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful)))
    # model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful)))
    model.add(Bidirectional(LSTM(units, stateful=stateful)))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))

    if useSoftmax:
        opti = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opti = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.add(Dense(fitSize, activation='softmax')) 
        model.compile(loss='binary_crossentropy',
                    optimizer='adam')
        # model.compile(loss='binary_crossentropy',
        #             optimizer='rmsprop')
        print('build network with softmax activation')
    else:
        model.add(Dense(fitSize, activation='sigmoid'))

        #            metrics=['accuracy']
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])
        #model.compile(loss='mean_squared_error', optimizer='sgd')

    return model
    

def LevelDataToTrainData(LevelData, numSamples):
    # 关卡数据转换训练数据
    # 关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    # 尝试每个样本三个输出，代表是否有该类型音符，（点击、滑动、长按）
    result = [[0.0, 0.0, 0.0]] * numSamples
    result = np.array(result)
    for l in LevelData:
        # time in ms, sample rate is 100 samples per second
        index = int(float(l[0]) / 10.0 + 0.5)
        type = l[1]

        result[index, 0] = 1.0
        if type == 3:
            continue

        result[index, type] = 1.0
        # if type == 2:
        #     last = int(float(l[2]) / 10.0)
        #     result[index:index+last, 2] = 1.0

    return result

def LevelToTrainDataSoftmax(levelData, numSamples):
    result = LevelDataToTrainData(levelData, numSamples)
    r = []
    for i in range(numSamples):
        if result[i, 0] == 1.0:
            r.append([0.0, 1.0])
        else:
            r.append([1.0, 0.0])
    r = np.array(r)
    return r


def TrainDataToLevelData(data, timeOffset):
    threhold = 0.8
    notes = []
    longNote = False
    last = 0
    start = 0  
    for i in range(len(data)):                
        d = data[i]
        t = i * 10 + timeOffset

        if not isinstance(d, np.ndarray):
            if (d > threhold):
                notes.append((t, 0, 0))
            continue

        if (d[0] > threhold):
            notes.append((t, 0, 0))

        if (len(d) == 1):
            continue

        if (d[1] > threhold):
            notes.append((t, 1, 0))
        if (d[2] > threhold):
            if not longNote:
                start = t
            longNote = True
            last += 10            
        else:
            if longNote:
                if last <= 10:
                    print('long note last too short')
                else:
                    notes.append((start, 2, last))
            longNote = False
            start = 0
            last = 0
    
    print('got number of notes ', len(notes))
    return notes

def SaveResult(prid, target, time, pathname):
    # pathname = '/Users/xuchao/Documents/python/MusicLevelAutoGen/result.txt'
    # if os.name == 'nt':
    #     pathname = r'D:\librosa\result.log'
    with open(pathname, 'w') as file:
        n = 0
        for i in zip(prid, target):
            t = time + n * 10
            minite = t // 60000
            sec = (t % 60000) // 1000
            milli = t % 1000

            file.write('%d:%d:%d, %s , %s\n' % (minite, sec, milli, i[0], i[1]))
            n += 1

    print('saved ', pathname)

def Test():
    
    songList = ['tianyuanriji', '2differenttears', 'abracadabra']

    # tianyuanriji 2differenttears 4minuteshm abracadabra

    batch_size = 256
    useSoftmax = False

    trainx, trainy = PrepareTrainData(songList, batch_size, useSoftmax)
    
    songList = ['4minuteshm']
    pathname = MakeMp3Pathname(songList[0])
    testx, testy = PrepareTrainData(songList, batch_size, useSoftmax)

    epochs = 100

    model = BuildLSTM(data_dim = 314, batch_size = batch_size, fitSize = 1, useSoftmax = useSoftmax)
    
    #model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, validation_split=0.2, shuffle = False)
    for i in range(epochs):
        model.fit(trainx, trainy, batch_size = batch_size, epochs = 1, validation_data=(testx, testy), shuffle = True)
        model.reset_states()
        print('epoch ', i)

    
    predicts = model.predict(testx, batch_size=batch_size)
    predicts = np.array(predicts)
    test_y = np.array(testy)
    SaveResult(predicts, testy, 0, r'D:\librosa\result2.log')
    notes = TrainDataToLevelData(predicts, 0)
    LevelInfo.SaveNote(notes, pathname, '_predict')
    # rmse = np.sqrt(mean_squared_error(predicts, test_y))
    # print('Test RMSE: %.3f' % rmse)

def DummyLSTMTest():
    timesteps = 1
    units = 25
    stateful = False

    fitSize = 2
    data_dim = 5

    batchsize = 32

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    # model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful,
    #             ), batch_input_shape=(batch_size, timesteps, data_dim)))

    
    model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful,
                ), input_shape=(timesteps, data_dim)))
    model.add(Bidirectional(LSTM(units, stateful=stateful)))

    model.add(Dense(fitSize, activation='softmax')) 
    model.compile(loss='binary_crossentropy',
                optimizer='Adam')
    # model.compile(loss='binary_crossentropy',
    #             optimizer='rmsprop')
    print('build network with softmax activation')


    numSamples = 500 * batchsize + 16
    samples = np.random.randn(numSamples, timesteps, data_dim)
    trainy = np.random.randn(numSamples, fitSize)

    model.fit(samples, trainy, batch_size=32, epochs=1, shuffle=False)


if __name__ == '__main__':                                                                                                                                                                
    Test()
    # DummyLSTMTest()

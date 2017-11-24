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

from lstm import postprocess


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

    model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=stateful)))
    model.add(Bidirectional(LSTM(units, stateful=stateful)))
    #model.add(Dropout(0.5))

    if useSoftmax:
        opti = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opti = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.add(Dense(fitSize, activation='softmax')) 
        model.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['mae', 'acc'])
        # model.compile(loss='binary_crossentropy',
        #             optimizer='rmsprop', metrics=['mae', 'acc'])
        print('build network with softmax activation')
    else:
        model.add(Dense(fitSize, activation='sigmoid'))

        #            metrics=['accuracy']
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])
        #model.compile(loss='mean_squared_error', optimizer='sgd')

    return model
    
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


def Test():
    
    songList = ['tianyuanriji', '2differenttears', 'abracadabra', 'hewojiaowangba', 'huashuo', 'ygrightnow', 'haodan', 'ineedmore', 'hangengxman']
    songList = ['tianyuanriji', '2differenttears', 'hewojiaowangba', 'huashuo']
    #songList = ['tianyuanriji']

    # tianyuanriji 2differenttears 4minuteshm abracadabra

    batch_size = 512
    useSoftmax = False
    units = 30
    epochs = 100

    trainx, trainy = PrepareTrainData(songList, batch_size, useSoftmax)
    
    songList = ['4minuteshm']
    pathname = MakeMp3Pathname(songList[0])
    testx, testy = PrepareTrainData(songList, batch_size, useSoftmax)

    fitsize = trainy.shape[1]

    model = BuildLSTM(data_dim = 314, batch_size = batch_size, fitSize = fitsize, useSoftmax = useSoftmax)
    
    stopCallback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
    
    # model.fit(trainx, trainy, batch_size = batch_size, epochs = epochs, validation_data=(testx, testy), shuffle = True, callbacks=[stopCallback])
    #model.fit(trainx, trainy, batch_size = batch_size, epochs = epochs, validation_data=(testx, testy), shuffle = True)

    for i in range(epochs):
        model.fit(trainx, trainy, batch_size = batch_size, epochs = 1, validation_data=(testx, testy), shuffle = True, callbacks=[stopCallback])
        model.reset_states()
        print('epoch ', i)

    keras.models.save_model(model, 'd:/mymodel.mdl')

    Evaluate(model)
    
    model = keras.models.load_model('d:/mymodel.mdl')
    
    Evaluate(model)


def Evaluate(model):
    useSoftmax = False
    batch_size = 512
    acceptThrehold = 0.2

    songList = ['4minuteshm']
    pathname = MakeMp3Pathname(songList[0])
    testx, testy = PrepareTrainData(songList, batch_size, useSoftmax)
    
    predicts = model.predict(testx, batch_size=batch_size)
    predicts = np.array(predicts)

    predicts = postprocess.pick(predicts)

    postprocess.SaveResult(predicts, testy, 0, r'D:\librosa\result.log')

    if useSoftmax:
        predicts = predicts[:,1]

    notes = postprocess.TrainDataToLevelData(predicts, 0, acceptThrehold)
    LevelInfo.SaveNote(notes, pathname, '_predict')



def Evaluate2():    
    model = keras.models.load_model('d:/mymodel.mdl')    

    useSoftmax = False
    batch_size = 512
    acceptThrehold = 0.2

    filename = r'd:\leveledtior\client\Assets\resources\audio\bgm\4minuteshm.m4a'

    trainx = lstm.myprocesser.LoadAndProcessAudio(filename)
    n = len(trainx) % batch_size
    trainx = trainx[:-n]

    trainx = trainx.reshape(trainx.shape[0], 1, trainx.shape[1])
    
    predicts = model.predict(trainx, batch_size=batch_size)
    predicts = np.array(predicts)

    predicts = postprocess.pick(predicts)

    if useSoftmax:
        predicts = predicts[:,1]

    notes = postprocess.TrainDataToLevelData(predicts, 0, acceptThrehold)    

    bpm = 124.973963758
    et = 1937
    musicTime = 128730
    levelNotes = postprocess.ConvertToLevelNote(notes, bpm, et)
    LevelInfo.GenerateIdolLevel('d:/test.xml', levelNotes, bpm, et, musicTime)


if __name__ == '__main__':                                                                                                                                                                
    # Test()
        
    # model = keras.models.load_model('d:/mymodel.mdl')    
    # Evaluate(model)
    Evaluate2()
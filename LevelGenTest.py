# coding=UTF-8
import pickle
import LevelInfo
import numpy as np
import lstm.myprocesser
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
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


def BuildLSTM(data_dim = 16, batch_size = 32, fitSize = 1, useSoftmax = False):
    
    timesteps = 1
    num_classes = 1
    units = 25

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, stateful=True, go_backwards=True,
                batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(units, return_sequences=True, stateful=True, go_backwards=True))
    model.add(LSTM(units, stateful=True, go_backwards=True))

    if useSoftmax:
        opti = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opti = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.add(Dense(fitSize, activation='softmax')) 
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam')
        # model.compile(loss='binary_crossentropy',
        #             optimizer='rmsprop')
        print('build network with softmax activation')
    else:
        model.add(Dense(fitSize))

        #            metrics=['accuracy']
        model.compile(loss='mean_squared_error', optimizer='adam')
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
    threhold = 0.1
    notes = []
    longNote = False
    last = 0
    start = 0  
    for i in range(len(data)):
        d = data[i]
        t = i * 10 + timeOffset
        if (d[0] > threhold):
            notes.append((t, 0, 0))
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

def SaveResult(prid, target):
    pathname = '/Users/xuchao/Documents/python/MusicLevelAutoGen/result.txt'
    with open(pathname, 'w') as file:
        for i in zip(prid, target):
            file.write('%s , %s\n' % i)
    print('saved ', pathname)

def Test():
    
    #data = LoadInputData('d:/test.pk')

    path = '/Users/xuchao/Documents/rhythmMaster/'
    songName = 'aiqingfadeguang'

    pathname = '%s%s/%s.mp3' % (path, songName, songName)
    inputData = lstm.myprocesser.LoadAndProcessAudio(pathname)

    numSample = len(inputData)
    print('number of samples ', numSample)

    pathname = '%s%s/%s_4k_nm.imd' % (path, songName, songName)
    level = LevelInfo.LoadRhythmMasterLevel(pathname)
    # targetData = LevelDataToTrainData(level, numSample)
    targetData = LevelToTrainDataSoftmax(level, numSample)

    batch_size = 1
    inputData = np.array(inputData)
    inputData = inputData.reshape(inputData.shape[0], 1, inputData.shape[1])
    
    separate = int(len(inputData) * 0.7)

    train_x = inputData[:separate]
    train_y = targetData[:separate]
    # train_y = train_y[:,0]

    test_x = inputData[separate:]
    test_y = targetData[separate:]

    print(test_x.shape)
    print(test_y.shape)

    epochs = 10

    model = BuildLSTM(data_dim = 314, batch_size = batch_size, fitSize = 2, useSoftmax = True)
    
    model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, validation_split=0.2, shuffle = False)
    # for i in range(epochs):
    #     model.fit(train_x, train_y, batch_size = batch_size, epochs = 1, shuffle = False)
    #     model.reset_states()
    #     print('epoch ', i)

    predicts = []
    for i in range(test_x.shape[0]):
        t = test_x[i].reshape(1, 1, inputData.shape[2])
        y = model.predict(t)
        #print(y[0], test_y[i])
        predicts.append(y[0])

    predicts = np.array(predicts)
    test_y = np.array(test_y)

    SaveResult(predicts, test_y)
    return

    notes = TrainDataToLevelData(predicts, separate * 10)
    LevelInfo.SaveNote(notes, pathname, '_predict')
    rmse = np.sqrt(mean_squared_error(predicts, test_y))
    print('Test RMSE: %.3f' % rmse)

if __name__ == '__main__':                                                                                                                                                                
    Test()

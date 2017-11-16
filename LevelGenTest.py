
import pickle

import LevelInfo
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def LoadInputData(pathname):    
    d = None
    with open(pathname, 'rb') as file:
        d = pickle.load(file)
    
    print('read file ', pathname, type(d), d.shape)
    return d


def LoadLevelData(pathname, numSample):
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


def BuildLSTM(data_dim = 16, batch_size = 32):
    
    timesteps = 1
    num_classes = 1
    units = 25

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, stateful=True, go_backwards = True,
                batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(units, return_sequences=True, stateful=True, go_backwards = True))
    model.add(LSTM(units, stateful=True, go_backwards = True))
    model.add(Dense(1))

    #, activation='softmax'

    # model.compile(loss='binary_crossentropy',
    #             optimizer='rmsprop',
    #             metrics=['accuracy'])


    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    

def Test():
    
    data = LoadInputData('d:/test.pk')
    numSample = len(data)

    notes = LoadLevelData(r'd:\librosa\炫舞自动关卡生成\level\idol_100052.xml', numSample)

    batch_size = 1
    data = np.array(data)
    data = data.reshape(data.shape[0], 1, data.shape[1])
    
    start = 1560
    train_x = data[start:start + 32 * 183]
    train_y = notes[start:start + 32 * 183]

    print(train_x[start])



    test_x = data[10000:12000]
    test_y = notes[10000:12000]
    # test_x = data[10000:10032]
    # test_y = notes[10000:10032]

    epochs=5      

    model = BuildLSTM(data_dim = 314, batch_size = batch_size)

    model.fit(train_x, train_y, batch_size = batch_size, epochs = 3, shuffle = False)
    # for i in range(epochs):
    #     model.fit(train_x, train_y, batch_size = batch_size, epochs = 1, shuffle = False)
    #     model.reset_states()
    #     print('epoch ', i)

    #t = test_x[0].reshape(1, 1, test_x.shape[2])
    for i in range(test_x.shape[0]):
        t = test_x[i].reshape(1, 1, data.shape[2])
        y = model.predict(t)
        print(y, test_y[i])


Test()


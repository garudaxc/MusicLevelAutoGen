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
import TFLstm


levelDifficulty = 1
numHidden = 26
batchSize = 128
numSteps = 512
numSteps = 256
inputDim = 314
outputDim = 2
learning_rate = 0.001
bSaveModel = True

maxSeqLen = 10

epch = 200

songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic', 'aiqingkele']

runList = []
def run(r):
    runList.append(r)
    return r


def CheckTrainningDataValid():
    '''
    扫描训练数据是否有效
    每个节拍被分为12等分，音符需要落在等分点上
    '''

    path = 'd:/librosa/RhythmMaster'
    files = os.listdir(path)
    dirs = []
    for f in files:
        if not os.path.isdir(path + '/' + f):
            continue
        filename = path + '/' + f + '/' + f + '.mp3'
        if not os.path.exists(filename):
            continue
        m = MusicTrainData(f)

    songList = ['haodan']
    songList = ['accentier']
    songList = ['reanimate']
    songList = ['huijiaxiang']
    songList = ['hongri']
    # songList = ['faraway']

    songName = songName
    mp3name = TFLstm.MakeMp3Pathname(songName)
    duration, bpm, entertime = TFLstm.LoadMusicInfo(mp3name)
    msPerBeat = 60000.0 / bpm
    msPerStep = msPerBeat / 12.0     # 32 step per bar, 8 step per beat
    step = round(msPerStep / 10)       # train step
    fps = 1000.0 / (msPerStep / step)
    # numframes = len(self.signals)

    pathname = TFLstm.MakeLevelPathname(songName, difficulty=1)
    if not os.path.exists(pathname):
        return
    level = LevelInfo.LoadRhythmMasterLevel(pathname)
    msPerFrame = 1000.0 / fps
    # print('msPerFrame', msPerFrame)
    
    maxResidual = 0
    for note in level:
        time, type, _ = note
        if type == LevelInfo.combineNode:
            continue
        
        frameIndex = time // msPerFrame
        residual = time % msPerFrame
        residual = min(residual, abs(residual-msPerFrame))
        if residual > maxResidual:
            maxResidual = residual
        # print('residual', residual)

    print('max residual', maxResidual)


def FrameIndex(time, msPerFrame):
    
    frameIndex = time // msPerFrame
    residual = time % msPerFrame
    residual = min(residual, abs(residual-msPerFrame))

    if residual > 3:
        print('residual too large', residual)
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
        # data[index:index+last, 3] = 1.0
    else:        
        data[index] = [0.0, 1.0, 0.0, 0.0]


def ConvertLevelToLables(level, numframes, msMinInterval):
    '''
    将level数据转换成训练数据
    音符的时间应该与step对齐    
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

        # frameIndex = time // msMinInterval
        # residual = time % msMinInterval
        # residual = min(residual, abs(residual-msMinInterval))
        # if residual > maxResidual:
        #     maxResidual = residual
        # print('residual', residual)

    # print('max residual', maxResidual)    

    return frames, maxResidual


def LableProcesserShortNote(lables):
    newLable = lables[:,[1, 2]]
    arg = newLable.argmax(1)
    # 取每一项中的较大元素
    newLable = newLable[range(len(newLable)), arg]
    newLable = TFLstm.SamplesToSoftmax(newLable)
    return newLable


class MusicTrainData():

    def __init__(self, songName, loadLables=True):
        self.songName = songName
        self.lables = []
        mp3name = TFLstm.MakeMp3Pathname(songName)
        self.duration, self.bpm, self.entertime = TFLstm.LoadMusicInfo(mp3name)
        msPerBeat = 60000.0 / self.bpm
        self.msMinInterval = msPerBeat / 12.0     # 12 step per beat
        self.step = round(self.msMinInterval / 10)       # train step per minimum note interval
        assert self.step <= maxSeqLen

        self.fps = 1000.0 / (self.msMinInterval / self.step)
        self.signals = lstm.myprocesser.LoadAndProcessAudio(mp3name, self.fps)
        numframes = len(self.signals)

        if numframes % self.step != 0:
            self.signals = self.signals[0:-(numframes%self.step)]       #取整
        numFeature = self.signals.shape[1]
        self.signals = np.reshape(self.signals, (-1, self.step, numFeature))
        numSeqs = len(self.signals)
        # print('numSeqs', numSeqs, 'duration', self.duration, 'interval', self.msMinInterval, 'duration/interval', self.duration/self.msMinInterval)

        # pad to max seq length
        toExpend = maxSeqLen - self.step
        self.signals = np.concatenate((self.signals, np.zeros((numSeqs, toExpend, numFeature))), axis=1)
        
        # print('song', songName)
        if loadLables:
            pathname = TFLstm.MakeLevelPathname(songName, difficulty=levelDifficulty)
            if not os.path.exists(pathname):
                return
            level = LevelInfo.LoadRhythmMasterLevel(pathname)
            self.lables, maxResidual = ConvertLevelToLables(level, numSeqs, self.msMinInterval)
            self.lables = LableProcesserShortNote(self.lables)
            if True: # for test
                lables = self.lables[:,1]
                notes = postprocess.TrainDataToLevelData(lables, self.msMinInterval, threhold=0.8, timeOffset=0)
                notes = notes[:,0]
                LevelInfo.SaveInstantValue(notes, pathname, '_inst')

            if maxResidual > 3:
                print(songName, 'step', self.step, 'maxResidual', maxResidual)
            # print('lables shape', self.lables.shape)
    
        # print('step', self.step, 'fps', self.fps)
        
    @property
    def numSeqs(self):
        return len(self.signals)

    def BeginBatch(self, batchSize):
        self.batchSize = batchSize
        self.currentBatch = 0
        self.numBatch = self.numSeqs // batchSize

        return self.numBatch
    
    def GetNextBatch(self):
        if self.currentBatch == self.numBatch:
            return None

        index = self.currentBatch * self.batchSize
        samples = self.signals[index:index+self.batchSize]
        seqlens = np.full(batchSize, self.step)

        lables = None
        if len(self.lables) > 0:
            lables  = self.lables[index:index+self.batchSize]        

        self.currentBatch += 1
        return (samples, lables, seqlens)


# @run
def TestPrepareData():

    if True:
        song = 'fancy'
        song = 'abracadabra'
        song = 'aiqingmaimai'

        m = MusicTrainData(song)
        return

    # 扫描最大的residual
    if True:
        path = 'd:/librosa/RhythmMaster'
        
        songList = []
        # songList = ['inthegame', 'isthisall', 'huashuo', '2differenttears', 'abracadabra', 'tictic', 'aiqingkele']
        if len(songList) == 0:
            songList = os.listdir(path)
        dirs = []
        for f in songList:
            if not os.path.isdir(path + '/' + f):
                continue
            filename = path + '/' + f + '/' + f + '.mp3'
            if not os.path.exists(filename):
                continue
            m = MusicTrainData(f)

        return

    # haodan

    songList = ['aiqingmaimai']
    songList = ['haodan']
    songList = ['accentier']
    songList = ['reanimate']
    songList = ['huijiaxiang']
    songList = ['hongri']
    songList = ['ineffabilis']
    songList = ['turanhxn']
    # songList = ['faraway']
    fileanme = TFLstm.MakeLevelPathname(songList[0], difficulty=1)
    level = LevelInfo.LoadRhythmMasterLevel(fileanme)
    mp3file = TFLstm.MakeMp3Pathname(songList[0])
    duration, bpm, entertime = TFLstm.LoadMusicInfo(mp3file)

    print('notes')

    start = 0

    step = 60000.0 / (bpm * 12)
    print('bpm', bpm, 'step', step)

    # print('result', start % step)

    for note in level:        
        time, type, _ = note
        if start == 0:
            start = time
            print('start', time)
            continue
        if type == 2:
            continue
        time = time - start
        residual = time % step
        residual = min(residual, abs(residual-step))
        if residual > 1:
            print('residual', residual)

    print('step', step)



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
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.9)
        dropoutCell.append(c)
    
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        dropoutCell[0:numLayers], dropoutCell[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    outlayerDim = tf.shape(output)[2]
    print('outlayer dim', outlayerDim)
    index = tf.range(0, batchSize) * maxSeqLen + (seqlen - 1)

    output = tf.reshape(output, [batchSize*maxSeqLen, outlayerDim])
    output = tf.gather(output, index)

    print('stack_bidirectional_dynamic_rnn', time.time() - t)    

    logits = tf.matmul(output, weights) + bais

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_op = tf.reduce_mean(crossEntropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)
    print('train op', time.time() - t)    

    correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return train_op, loss_op, accuracy
    
    # prediction without dropout
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        cells[0:numLayers], cells[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    output = tf.reshape(output, [batchSize*maxSeqLen, outlayerDim])
    output = tf.gather(output, index)
    logits = tf.matmul(output, weights) + bais

    prediction = tf.nn.softmax(logits, name='prediction')

    print('ready to train')

    return train_op, loss_op, accuracy, prediction


def LoadTrainData(songList):
    samplesList = []
    lablesList = []
    seqLenList = []
    for song in songList:
        data = MusicTrainData(song)
        numBatches = data.BeginBatch(batchSize)
        for i in range(numBatches):
            s, l, seq = data.GetNextBatch()
            samplesList.append(s)
            lablesList.append(l)
            seqLenList.append(seq)
    
    return samplesList, lablesList, seqLenList


def GenerateLevel(sess, prediction, X, seqLen):
    print('gen level')

    song = ['jilejingtu']
    data = MusicTrainData(song[0], loadLables=False)
    numBatches = data.BeginBatch(batchSize)

    evaluate = []
    for i in range(numBatches):
        s, _, seq = data.GetNextBatch()
        t = sess.run(prediction, feed_dict={X:s, seqLen:seq})
        evaluate.append(t)

    evaluate = np.array(evaluate)
    evaluate = np.reshape(evaluate, (-1, outputDim))

    print(evaluate[0:20])

    saveRawData = True
    if saveRawData:
        with open('d:/work/evaluate_data.raw', 'wb') as file:
            pickle.dump(evaluate, file)

    acceptThrehold = 0.6
    pathname = TFLstm.MakeMp3Pathname(song[0])
    predicts = postprocess.pick(evaluate)

    postprocess.SaveResult(predicts, data.msMinInterval, 0, r'D:\work\result.log')

    predicts = predicts[:,1]
    notes = postprocess.TrainDataToLevelData(predicts, data.msMinInterval, acceptThrehold, timeOffset=0)
    print('gen notes number', len(notes))

    notes = notes[:,0]
    LevelInfo.SaveInstantValue(notes, pathname, '_inst')

@run
def LoadRawData():
    
    song = ['jilejingtu']
    data = MusicTrainData(song[0], loadLables=False)
    pathname = TFLstm.MakeMp3Pathname(song[0])
    with open('d:/work/evaluate_data.raw', 'rb') as file:
        evaluate = pickle.load(file)
        evaluate = np.reshape(evaluate, (-1, outputDim))

        print('len', len(evaluate), 'time', (len(evaluate) * data.msMinInterval / 1000.0))
        
        predicts = evaluate
        predicts = postprocess.pick(evaluate, kernelSize=3)

        # postprocess.SaveResult(predicts, data.msMinInterval, 0, r'D:\work\result.log')

        predicts = predicts[:,1]
        acceptThrehold = 0.4
        notes = postprocess.TrainDataToLevelData(predicts, data.msMinInterval, acceptThrehold, timeOffset=0)
        print('gen notes number', len(notes))

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst')

# @run
def LoadModel():    
    saver = tf.train.import_meta_graph("d:/work/model.ckpt.meta")

    with tf.Session() as sess:        
        saver.restore(sess, 'd:/work/model.ckpt')
        print('model loaded')

        predict = tf.get_default_graph().get_tensor_by_name("prediction:0")         
        X = tf.get_default_graph().get_tensor_by_name('X:0')
        seqLen = tf.get_default_graph().get_tensor_by_name('seqLen:0')
        print('seqLen',seqLen)

        GenerateLevel(sess, predict, X, seqLen)

# @run
def _Main():

    samplesList, lablesList, seqLenList = LoadTrainData(songList)
    numBatches = len(samplesList)

    seqlenHolder = tf.placeholder(tf.int32, [None], name='seqLen')
    X = tf.placeholder(dtype=tf.float32, shape=(batchSize, maxSeqLen, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(batchSize, outputDim), name='Y')
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
        if bSaveModel:
            print('begin save model')
            TFLstm.SaveModel(sess, saveMeta=True)

        for j in range(epch):
            loss = []
            acc = []

            for i in range(numBatches):                
                sample, lable, seqlen = samplesList[i], lablesList[i], seqLenList[i]

                if i % 6 == 0:
                    l, a = sess.run([loss_op, accuracy], feed_dict={X:sample, Y:lable, seqlenHolder:seqlen})
                    loss.append(l)
                    acc.append(a)
                else:
                    t = sess.run(train_op, feed_dict={X:sample, Y:lable, seqlenHolder:seqlen, learningRate:currentLearningRate})
    
            lossValue = sum(loss) / len(loss)
            accValue = sum(acc) / len(acc)

            notIncreaseCount += 1
            if accValue > maxAcc:
                maxAcc = accValue
                notIncreaseCount = 0
                # save checkpoint
                print('save checkpoint')
                TFLstm.SaveModel(sess)
            
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
        saver.restore(sess, 'd:/work/model.ckpt')
        print('checkpoint loaded')
        GenerateLevel(sess, prediction, X, seqlenHolder)


# @run
def TestBuildRnn():    
    song = 'abracadabra'
    song = 'aiqingmaimai'
    song = '2differenttears'
    song = '1987wbzhyjn'

    m = MusicTrainData(song)
    numBatches = m.BeginBatch(batchSize)
    print('num batches', numBatches)

    return



if __name__ == '__main__':
    for fun in runList:
        fun()

import pickle
import os
import LevelInfo
import numpy as np
import myprocesser
import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import matplotlib.pyplot as plt
import util
import DownbeatTracking
import shutil

rootDir = util.getRootDir()

runList = []
def run(r):
    runList.append(r)
    return r

LevelDifficutly = 2
numHidden = 26
batchSize = 16
numSteps = 512
numSteps = 128
inputDim = 314
# outputDim = 2
learning_rate = 0.001
bSaveModel = True

epch = 300

# for long note
songList = ['aiqingmaimai','ai', 'hellobaby', 'hongri', 'houhuiwuqi', 'huashuo', 'huozaicike', 'haodan']
songList = ['huashuo', 'haodan']


songList = ['inthegame', 'isthisall', 'huashuo', 'haodan', '2differenttears', 'abracadabra', 'tictic']

songList = ['inthegame', 'remains', 'ilikeit', 'haodan', 'ai', 
'1987wbzhyjn', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'myall', 'unreal', 'faraway',
'fengniao', 'wangfei', 'wodemuyang', 'mrx', 'rageyourdream', 'redhot', 'yilububian', 'yiqiyaobai', 'yiranaini', 'yiyejingxi',
'hongri', 'huahuangshou', 'huashuo', 'ificoulefly', 'ineedmore', 'iwantmytearsback',  'rippingthebackside', 
 'whencaniseeyouagain', 'wingsofpiano', 'wocanggwz', 'ygrightnow']

songList = ['inthegame', 'remains', 'ilikeit', 'haodan', 'ai', 
'1987wbzhyjn', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'myall', 'unreal', 'faraway',
'fengniao', 'wangfei', 'wodemuyang', 'mrx', 'rageyourdream', 'redhot', 'yilububian', 'yiqiyaobai', 'yiranaini', 'yiyejingxi']


# 'mrq',  'huozaicike' 'ribuluo'

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

def FillNote(data, note, msPerFrame, isCombineNote):
    # 音符数据转化为sample data
    index = FrameIndex(note[0], msPerFrame)
    type = note[1]

    if type == LevelInfo.shortNote: 
        data[index, 1] = 1.0
    elif type == LevelInfo.slideNote:
        if (not isCombineNote):
            data[index, 2] = 1.0
        else:
            data[index, 3] = 1.0
    elif type == LevelInfo.longNote:
        last = FrameIndex(note[2], msPerFrame)
        data[index:index+last, 4] = 1.0
    else:
        assert False


def ConvertLevelToLables(level, numframes, msMinInterval = 10):
    '''
    关卡数据转换训练数据
    关卡数据（时刻，类型，时长），训练数据为每个采样点是否有值
    样本格式 non-note short slide slide-in-combine long    
    '''
    frames = [[1.0, 0.0, 0.0, 0.0, 0.0]] * numframes
    frames = np.array(frames)

    combineNote = 0
    notesInCombine = 0
    for note in level:
        time, type, val = note
        if type == LevelInfo.combineNode:             
            combineNote += 1
            for n in val:
                FillNote(frames, n, msMinInterval, True)
        else:        
            FillNote(frames, note, msMinInterval, False)

    # a = np.sum(frames, axis=0)
    # print('frames', a, 'combine note', combineNote, 'notes in combine', notesInCombine)

    return frames


def SamplesToSoftmax(samples):
    '''
    将样本数据转换为分类数据
    '''
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    s1 = 1 - np.amax(samples, axis=1)
    s1 = s1[:, np.newaxis]
    return np.hstack((s1, samples))


def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = rootDir + 'rm/'
    return path

def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.m4a' % (path, song, song)
    if not os.path.exists(pathname):
        pathname = '%s%s/%s.mp3' % (path, song, song)

    return pathname

def MakeMp3Dir(song):
    path = GetSamplePath()
    pathname = '%s%s/' % (path, song)
    if not os.path.exists(pathname):
        assert False
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
        inputData = myprocesser.LoadAndProcessAudio(pathname)
        trainx.append(inputData)
        numSample = len(inputData)

        if loadTestData:
            pathname = MakeLevelPathname(song, difficulty=LevelDifficutly)
            level = LevelInfo.LoadRhythmMasterLevel(pathname)

            targetData = ConvertLevelToLables(level, numSample)

            # a = targetData[:, [1, 2, 3]]
            # count = (a>0.1).nonzero()[0].shape[0]
            # print('count', count)
                
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




class TrainDataBase():

    def GetBatch(self, n):
        x = self._x[n]
        y = self._y[n]
        seqLen = np.array([self.numSteps] * self.batchSize)
        return x, y, seqLen

    def ShuffleBatch(self):
        s = np.arange(self.batchSize)
        np.random.shuffle(s)

        self._x = self._x[:, s]

        if not self._y[0] is None:
            self._y = self._y[:, s]

    def BuildBatch(self, x, y):
        count = x.shape[0]
        self.numBatches = count // (self.batchSize * self.numSteps)
        count = self.numBatches * (self.batchSize * self.numSteps)

        y = y[:count]
        yDim = y.shape[1]
        y = y.reshape(self.batchSize, -1, self.numSteps, yDim)
        self._y = y.transpose(1, 0, 2, 3)

        x = x[:count]
        xDim = x.shape[1]
        x = x.reshape(self.batchSize, -1, self.numSteps, xDim)
        self._x = x.transpose(1, 0, 2, 3)
        print('numbatches', self.numBatches, 'x shape', self._x.shape, 'y shape', self._y.shape)

###
# TrainDataDynShortNoteSinging
###
class TrainDataDynShortNoteSinging(TrainDataBase):
    lableDim = 2
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if (not y is None): 
            beat = np.max(y[:, 1:4], axis=1)
            
            long = np.max(y[:, 4:5], axis=1)
            for i in range(1, len(long)):
                if long[i-1] == 0 and long[i] > 0:
                    beat[i] = 1

            y = SamplesToSoftmax(beat)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 

    def GetModelPathName():
        return 'd:/work/model_shortnote_singing.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_short_singing.raw'
    
    def GenerateLevel(predicts, pathname):      
                       
        pre = predicts[:, 1]
        # pre = postprocess.PurifyInstanceSample(pre)
        # picked = postprocess.PickInstanceSample(pre)

        # sam = np.zeros_like(pre)
        # sam[picked] = 1
        
        notes = postprocess.TrainDataToLevelData(pre, 10, 0.90, 0)
        notes = np.asarray(notes)
        notes[0]

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst_singing')



class TrainDataDynShortNoteBeat(TrainDataBase):
    lableDim = 2
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]

        if (not y is None): 
            beat = np.max(y[:, 1:4], axis=1)
            
            long = np.max(y[:, 4:5], axis=1)
            for i in range(1, len(long)):
                if long[i-1] == 0 and long[i] > 0:
                    beat[i] = 1

            y = SamplesToSoftmax(beat)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 

    def GetModelPathName():
        return rootDir + 'model_shortnote_beat/model_shortnote_beat.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_short_beat.raw'
    
    def GenerateLevel(predicts, pathname):     
                       
        short = predicts[:, 1]
        short = postprocess.PickInstanceSample(short, count=400)
        notes = postprocess.TrainDataToLevelData(short, 10, 0.1, 0)

        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_inst_beat')


###
# TrainDataDynLongNote
###
class TrainDataDynLongNote(TrainDataBase):
    lableDim = 2
    def __init__(self, batchSize, numSteps, x, y=None):
        assert x.ndim == 2
        self.batchSize = batchSize
        self.numSteps = numSteps

        count = x.shape[0]
        if (not y is None):
            
            beat = np.max(y[:, 3:5], axis=1)
            y = SamplesToSoftmax(beat)
            print('y shape', y.shape, 'y sample count', len(y), 'y sum', np.sum(y, axis=0))

            self.BuildBatch(x, y)
        else:
            self._y = [None] * count

            x = x[:-(count%numSteps)]
            x = x.reshape(-1, 1, numSteps, inputDim)
            self._x = np.repeat(x, batchSize, axis=1)
            print('x shape', self._x.shape)

            self.numBatches = len(x)
            print('numbatch', self.numBatches) 


    def GetModelPathName():
        return rootDir + 'model/model_longnote.ckpt'

    def RawDataFileName(song):
        path = MakeMp3Dir(song)
        return path + 'evaluate_data_long.raw'    
    
    def GenerateLevel(predicts, pathname):     
              
        # #for long note
        # print('evaluate shape', evaluate.shape)
        # predicts = evaluate[:,1] > acceptThrehold
        # LevelInfo.SaveSamplesToRegionFile(predicts, pathname, '_region')
        # return

        # predicts = postprocess.pick(evaluate, kernelSize=11)
        print('predicts', predicts.shape)

        # postprocess.SaveResult(predicts, testy, 0, r'D:\work\result.log')
        
        acceptThrehold = 0.8
        pred = predicts[:,1]
        notes = postprocess.TrainDataToLevelData(pred, 10, acceptThrehold, 0)
        print('gen notes number', len(notes))
        notes = notes[:,0]
        LevelInfo.SaveInstantValue(notes, pathname, '_region')

        # LevelInfo.SaveInstantValue(notes, pathname, '_region')


def BuildDynamicRnn(X, Y, seqlen, learningRate, TrainData):
    result = {}

    weights = tf.Variable(tf.random_normal(shape=[2*numHidden, TrainData.lableDim]))
    bais = tf.Variable(tf.random_normal(shape=[TrainData.lableDim]))
    
    numLayers = 3
    cells = []
    dropoutCell = []
    for i in range(numLayers * 2):
        c = rnn.LSTMCell(numHidden, use_peepholes=True, forget_bias=1.0)
        cells.append(c)
        c = tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=0.6)
        dropoutCell.append(c)
    
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        dropoutCell[0:numLayers], dropoutCell[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    outlayerDim = tf.shape(output)[2]
    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])

    logits = tf.matmul(output, weights) + bais

    Y = tf.reshape(Y, [batchSize*numSteps, TrainData.lableDim])

    # weight for each class, balance the sample
    class_weight = tf.reduce_sum(Y, axis=0) / tf.cast(tf.shape(Y)[0], tf.float32)
    class_weight = tf.maximum(class_weight, tf.ones_like(class_weight) * 0.001)
    class_weight = (1.0 / TrainData.lableDim) / class_weight
    result['class_weight'] = class_weight
    class_weight = tf.reduce_sum(class_weight * Y, axis=1)

    # # Define loss and optimizer
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

    loss_op = tf.reduce_mean(crossEntropy * class_weight)
    result['loss_op'] = loss_op

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)
    result['train_op'] = train_op

    correct = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    result['accuracy'] = accuracy
    
    # prediction without dropout
    output, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        cells[0:numLayers], cells[numLayers:], 
        X, sequence_length=seqlen, dtype=tf.float32)

    # output = tf.reshape(output, [batchSize*maxSeqLen, outlayerDim])
    
    output = tf.reshape(output, [batchSize*numSteps, outlayerDim])
    logits = tf.matmul(output, weights) + bais

    predict_op = tf.nn.softmax(logits, name='predict_op')
    result['predict_op'] = predict_op

    print('build rnn done')
    return result

   
def SaveModel(sess, filename, saveMeta = False):
    saver = tf.train.Saver()
    saver.save(sess, filename, write_meta_graph=saveMeta)
    print('model saved')


def EvaluateWithModel(modelFile, song, rawFile, TrainData):
    # 加载训练好的模型，将结果保存到二进制文件
    
    graphFile = modelFile + '.meta'
    saver = tf.train.import_meta_graph(graphFile)

    with tf.Session() as sess:
        
        saver.restore(sess, modelFile)
        print('model loaded')

        predict_op = tf.get_default_graph().get_tensor_by_name("predict_op:0")
        print('predict_op', predict_op)         
        X = tf.get_default_graph().get_tensor_by_name('X:0')
        seqLenHolder = tf.get_default_graph().get_tensor_by_name('seqLen:0')
    
        testx, _ = PrepareTrainData(song, batchSize, loadTestData = False)

        data = TrainData(batchSize, numSteps, testx)
        
        evaluate = []
        for i in range(data.numBatches):
            xData, _, seqLen = data.GetBatch(i)
            t = sess.run(predict_op, feed_dict={X:xData, seqLenHolder:seqLen})
            t = t[0:numSteps,:]
            evaluate.append(t)

        predicts = np.stack(evaluate).reshape(-1, TrainData.lableDim)

        with open(rawFile, 'wb') as file:
            pickle.dump(predicts, file)
            print('raw file saved', predicts.shape)

    return predicts


@run
def GenerateLevel():
    print('gen level')

    debugET = -1
    song = ['PleaseDontGo']
    song = ['jilejingtu']
    song = ['aLIEz']
    song = ['bboombboom']
    song = ['1987wbzhyjn']
    song = ['mrq']
    song = ['ribuluo']
    song = ['xiagelukoujian']
    song = ['CheapThrills']
    song = ['dainiqulvxing']
    # song = ['foxishaonv']
    # song = ['dear my lady']
    # song = ['tian kong zhi cheng']
    # song = ['kanong']
    # song = ['qinghuaci']
    # song = ['zuichibi']]

    # 
    song = ['100580']
    debugET = 100
    song = ['ISI']
    debugET = 15020
    song = ['Luv Letter']
    debugET = 29543
    song = ['jianyunzhe']
    debugET = 682
    song = ['jinjuebianjingxian']
    debugET = 6694
    song = ['ouxiangwanwansui']
    debugET = 571
    song = ['shanghaihongchaguan']
    debugET = 1353
    song = ['xingzhixing']
    debugET = 2360
    song = ['yanghuadaqiao']
    debugET = 170

    # postprocess.ProcessSampleToIdolLevel(song[0])
    # return

    pathname = MakeMp3Pathname(song[0])
    print(pathname)


    # DownbeatTracking.ChromaFeature(pathname)
    # return

    useOnsetForShort = True
    if True:
        # gen raw data

        # TrainData = TrainDataDynLongNote
        # rawFile = TrainData.RawDataFileName(song[0])
        # modelFile = TrainData.GetModelPathName()
        # predicts = EvaluateWithModel(modelFile, song, rawFile, TrainData)   
        # print('predicts shape', predicts.shape)

        # TrainData.GenerateLevel(predicts, pathname)

        # if not useOnsetForShort:
        #     TrainData = TrainDataDynShortNoteBeat
        #     rawFile = TrainData.RawDataFileName(song[0])
        #     modelFile = TrainData.GetModelPathName()
        #     predicts = EvaluateWithModel(modelFile, song, rawFile, TrainData)  
        #     print('predicts shape', predicts.shape) 

        # TrainData.GenerateLevel(predicts, pathname)

        print('calc bpm')
        # DownbeatTracking.CalcMusicInfoFromFile(pathname, debugET)

    if not useOnsetForShort:
        # levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (song[0])
        levelEditorRoot = rootDir + 'LevelEditorForPlayer_8.0/LevelEditor_ForPlayer_8.0/'
        levelFile = '%sclient/Assets/LevelDesign/%s.xml' % (levelEditorRoot, song[0])
        duration, bpm, et = LevelInfo.LoadMusicInfo(pathname)
        
        rawFileLong = TrainDataDynLongNote.RawDataFileName(song[0])
        rawFileShort = TrainDataDynShortNoteBeat.RawDataFileName(song[0])
        #levelNotes = postprocess.ProcessSampleToIdolLevel(rawFileLong, rawFileShort)
        with open(rawFileShort, 'rb') as file:
            predicts = pickle.load(file)

        short = predicts[:, 1]
        short = DownbeatTracking.PickOnsetFromFile(pathname, bpm, duration, onsets=short)
        levelNotes = postprocess.ProcessSampleToIdolLevel2(rawFileLong, short)

        LevelInfo.GenerateIdolLevel(levelFile, levelNotes, bpm, et, duration)

    if useOnsetForShort:
        # levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (song[0])
        levelEditorRoot = rootDir + 'LevelEditorForPlayer_8.0/LevelEditor_ForPlayer_8.0/'
        levelFile = '%sclient/Assets/LevelDesign/%s.xml' % (levelEditorRoot, song[0])
        duration, bpm, et = LevelInfo.LoadMusicInfo(pathname)

        # postprocess.SaveDownbeat(bpm, et/1000.0, duration/1000.0, pathname)
        # return
        
        rawFileLong = TrainDataDynLongNote.RawDataFileName(song[0])
        short = DownbeatTracking.PickOnsetFromFile(pathname, bpm, duration)

        levelNotes = postprocess.ProcessSampleToIdolLevel2(rawFileLong, short)

        LevelInfo.GenerateIdolLevel(levelFile, levelNotes, bpm, et, duration)
    

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


# @run
def _Main():    
    # testx, testy = PrepareTrainData(songList, batchSize)
    TrainData = TrainDataDynLongNote
    # TrainData = TrainDataDynShortNoteBeat

    testx, testy = PrepareTrainDataFromPack()
    
    data = TrainData(batchSize, numSteps, testx, testy)
    numBatches = data.numBatches
    
    seqlenHolder = tf.placeholder(tf.int32, [None], name='seqLen')
    X = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, inputDim), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(batchSize, numSteps, TrainData.lableDim), name='Y')
    learningRate = tf.placeholder(dtype=tf.float32, name='learn_rate')

    result = BuildDynamicRnn(X, Y, seqlenHolder, learningRate, TrainData)

    maxAcc = 0.0
    notIncreaseCount = 0
    currentLearningRate = learning_rate
    learningRateFined = False
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(tf.global_variables_initializer())
        SaveModel(sess, TrainData.GetModelPathName(), saveMeta=True)

        for j in range(epch):
            loss = []
            acc = []

            for i in range(numBatches):
                xData, yData, seqLen = data.GetBatch(i)

                if i % 6 == 0:
                    l, a = sess.run([result['loss_op'], result['accuracy']], 
                    feed_dict={X:xData, Y:yData, seqlenHolder:seqLen})
                    
                    loss.append(l)
                    acc.append(a)
                else:
                    t = sess.run(result['train_op'], feed_dict={X:xData, Y:yData, seqlenHolder:seqLen, learningRate:currentLearningRate})
                
            lossValue = sum(loss) / len(loss)
            accValue = sum(acc) / len(acc)

            notIncreaseCount += 1
            if accValue > maxAcc:
                maxAcc = accValue
                notIncreaseCount = 0
                # save checkpoint
                print('save checkpoint')
                SaveModel(sess, TrainData.GetModelPathName())
            
            if notIncreaseCount > 10:
                print('stop learning')
                break

            # if notIncreaseCount > 10 and not learningRateFined:
            #     currentLearningRate = currentLearningRate / 2
            #     print('change learning rate', currentLearningRate)
            #     notIncreaseCount = 0
            #     learningRateFined = True

            # if notIncreaseCount > 15 and learningRateFined:
            #     print('stop learning')
            #     break
            
            print('epch', j, 'loss', lossValue, 'accuracy', accValue, 'not increase', notIncreaseCount)
            
            data.ShuffleBatch()

# @run
def SaveShortNoteFile(pathname=r'd:\librosa\RhythmMaster\breakfree\breakfree_4k_nm.imd'):
    # 读取节奏大师关卡，生成文件
    duration = LevelInfo.ReadRhythmMasterLevelTime(pathname)

    level = LevelInfo.LoadRhythmMasterLevel(pathname)

    numSample = duration // 10

    targetData = ConvertLevelToLables(level, numSample)    
    beat = np.max(targetData[:, 1:4], axis=1)

    # beat[:] = 0
    long = np.max(targetData[:, 4:5], axis=1)
    for i in range(1, len(long)):
        if long[i-1] == 0 and long[i] > 0:
            beat[i] = 1
    
    note = postprocess.TrainDataToLevelData(beat, 10, 0.1)
    note = note[:,0]
    print('note count', len(note))
    LevelInfo.SaveInstantValue(note, pathname, '_origshort')
    # LevelInfo.SaveInstantValue(note, pathname, '_origlong')


# @run
def GenShortNoteFromRhythmLevel():
    
    path = util.GetSamplePath()

    dirs = os.listdir(path)
    for name in dirs:
        dirname = path + name
        if not os.path.isdir(dirname):
            continue
        
        filename = util.MakeLevelPathname(name, 1)
        if os.path.exists(filename):
            SaveShortNoteFile(filename)
            print(filename, 'saved')

        filename = util.MakeLevelPathname(name, 2)
        if os.path.exists(filename):
            SaveShortNoteFile(filename)
            print(filename, 'saved')


def ReadTrainningRegion(file):

    regions = ([], [])
    for line in file:
        time, type, duration = line.split(',')
        time = float(time) * 1000
        duration = float(duration) * 1000
        type = int(type)
        regions[type].append((time, duration))
        # print(type, time, duration)
    
    return regions
        
# @run
def LoadMarkedTrainningLable():
    # 加载标记信息
    path = '/Users/xuchao/Documents/python/MusicLevelAutoGen/train'
    path = 'D:/librosa/MusicLevelAutoGen/train'
    path = rootDir + 'MusicLevelAutoGen/train'
    filelist = os.listdir(path)
    lables = ([], [])
    lableDuration = [0, 0]
    for file in filelist:
        if os.path.splitext(file)[1] != '.csv':
            continue

        pathname = path + '/' + file
        file = os.path.splitext(file)[0]
        song, level = file.split('_')
        # print(song, level)

        with open(pathname, 'r') as f:
            regions = ReadTrainningRegion(f)

        for i in range(2):
            if len(regions[i]) == 0:
                continue
            lables[i].append((song, level, regions[i]))
            for _, dur in regions[i]:
                lableDuration[i] += dur // 1000

    print(lableDuration[0])
    print(lableDuration[1])

    return lables

# featureFile = r'D:\librosa\MusicLevelAutoGen\train\beat_train_feature.raw'
# featureFile = r'D:\librosa\MusicLevelAutoGen\train\singing_train_feature.raw'
featureFile = rootDir + 'data/singing_train_feature.raw'

# @run
def LoadAllTrainningLable():
    # 加载所有的标记段落，保守数据到二进制文件
    metaData = LoadMarkedTrainningLable()
    metaData = metaData[0] + metaData[1]

    print('labled song', len(metaData))

    featureData = []
    lableData = []
    index = 0
    for song, _, regions in metaData:
        print('load', index, song)
        index += 1
        pathname = MakeMp3Pathname(song)
        features = myprocesser.LoadAndProcessAudio(pathname)
        numSample = len(features)
        
        pathname = MakeLevelPathname(song, difficulty=1)
        print(pathname)
        level = LevelInfo.LoadRhythmMasterLevel(pathname)
        lable = ConvertLevelToLables(level, numSample)

        for start, length in regions:
            start = int(start // 10)
            length = int(length // 10)
            featureData.append(features[start:start+length])
            lableData.append(lable[start:start+length])
        
    featureData = np.vstack(featureData)
    lableData = np.vstack(lableData)
    print('featureData', featureData.shape, 'lableData', lableData.shape)

    with open(featureFile, 'wb') as file:
        pickle.dump(featureData, file)
        pickle.dump(lableData, file)
        print('raw file saved', featureData.shape)


def PrepareTrainDataFromPack():
    
    with open(featureFile, 'rb') as file:
        xdata = pickle.load(file)
        ydata = pickle.load(file)
        print(xdata.shape, ydata.shape)
    
    return xdata, ydata

    

if __name__ == '__main__':
    
    for fun in runList:
        fun()



import numpy as np
import NotePreprocess
import TFLstm
import ModelTool
import os
import util
import tensorflow as tf
import NotePreprocess
import madmom
import NoteModel
import time

def CompareData(arrA, arrB):
    idA = id(arrA)
    idB = id(arrB)
    print('compare object id', idA, idB)
    if idA == idB:
        print('compare object id equal. may be error!')
        return False

    arrA = np.array(arrA)
    arrB = np.array(arrB)
    shapeA = np.shape(arrA)
    shapeB = np.shape(arrB)
    print('compare shape', shapeA, shapeB)
    if len(shapeA) != len(shapeB):
        print('compare shape failed')
        return False

    for valA, valB in zip(shapeA, shapeB):
        if valA != valB:
            print('compare shape val failed')
            return False

    reshapeA = np.reshape(arrA, [-1])
    reshapeB = np.reshape(arrB, [-1])
    print('compare length', len(arrA), len(arrB))
    if len(arrA) != len(arrB):
        print('compare length failed')
        return False
        
    subVal = np.abs(reshapeA - reshapeB)
    maxDis = np.max(subVal)
    minDis = np.min(subVal)
    aveDis = np.average(subVal)
    dis = np.sqrt(np.sum(np.square(subVal)))
    print('arrA to arrB dis %f, aveDis %f, maxDis %f, minDis %f' % (dis, aveDis, maxDis, minDis))

    for vA, vB in zip(reshapeA, reshapeB):
        if vA != vB:
            print('compare arr val failed')
            return False

    print('compare res true')
    return True

def CheckSplitFuncValid():
    fileList, trainList, validateList = TFLstm.LoadCsvSingingFileList()
    for song, _ in fileList:
        print('check song', song)
        songFilePath = TFLstm.MakeMp3Pathname(song)
        sampleRate = 44100
        audioData = NotePreprocess.LoadAudioFile(songFilePath, sampleRate)
        audioData = np.array(audioData)
        specDiff, melLogSpec = NotePreprocess.SpecDiffAndMelLogSpecEx(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
        specDiffSrc, melLogSpecSrc = NotePreprocess.SpecDiffAndMelLogSpec(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
        # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'specdiff_split'), specDiff)
        # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'specdiff_src'), specDiffSrc)
        if not CompareData(specDiffSrc, specDiff):
            print(song, 'check failed')
            return False

        if not CompareData(melLogSpecSrc, melLogSpec):
            print(song, 'check failed')
            return False

        print(song, 'check valid')

    return True

def RunDownbeatsTFModel(xData, modelFilePath, useLSTMBlockFusedCell):
    print('run model', modelFilePath)
    if useLSTMBlockFusedCell:
        batchSize = len(xData[0])
        maxTime = len(xData)
    else:
        batchSize = len(xData)
        maxTime = len(xData[0])

    inputDim = len(xData[0][0])

    seqLen = [maxTime] * batchSize

    usePeepholes = True

    graph = tf.Graph()
    variableScopeName = os.path.splitext(os.path.basename(modelFilePath))[0]
    with graph.as_default():
        tensorDic = ModelTool.BuildDownbeatsModelGraph(variableScopeName, 3, batchSize, maxTime, 25, inputDim, usePeepholes, [50, 3], [3], tf.nn.softmax, useLSTMBlockFusedCell)
        with tf.Session() as sess:
            varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variableScopeName)
            saver = tf.train.Saver(var_list=varList)
            saver.restore(sess, modelFilePath)
            
            X = tensorDic['X']
            sequenceLength = tensorDic['sequence_length']
            output = tensorDic['output']

            startTime = time.time()
            res = sess.run([output], feed_dict={X:xData, sequenceLength:seqLen})
            print('model real cost', time.time() - startTime)
            print('sub res shape', np.shape(res))

    res = np.reshape(res, (maxTime, -1))
    return res

def RunAllDownbeatsTFModel(audioFilePath):
    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(audioFilePath, sampleRate)
    audioData = np.array(audioData)
    specDiff, melLogSpec = NotePreprocess.SpecDiffAndMelLogSpec(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
    
    preCalcData = specDiff
    start, analysisLength = NotePreprocess.AnalysisInfo(len(audioData) / sampleRate)
    fps = 100
    startIdx = int(start * fps)
    endIdx = startIdx + int(analysisLength * fps)
    clipPreCalcData = preCalcData[startIdx:endIdx]

    specDiff = clipPreCalcData

    proc = NotePreprocess.CustomRNNDownBeatProcessor()
    srcRes = proc(specDiff)
    print('shape', np.shape(srcRes))
    import DownbeatTracking
    DownbeatTracking.SaveInstantValue(srcRes[:, 0], audioFilePath, '_rnn_src_0')
    DownbeatTracking.SaveInstantValue(srcRes[:, 1], audioFilePath, '_rnn_src_1')
    
    batchSize = 1
    maxTime = len(specDiff)
    inputDim = len(specDiff[0])

    useLSTMBlockFusedCell = True
    if useLSTMBlockFusedCell:
        xData = np.reshape(specDiff, (maxTime, batchSize, inputDim))
    else:
        xData = np.reshape(specDiff, (batchSize, maxTime, inputDim))

    startTime = time.time()
    res = []
    for i in range(8):
        print('model', i)
        varScopeName = 'downbeats_' + str(i)
        rootDir = util.getRootDir()
        outputFileDir = os.path.join(rootDir, 'madmom_to_tf')
        outputFileDir = os.path.join(outputFileDir, varScopeName)
        outputFilePath = os.path.join(outputFileDir, varScopeName + '.ckpt')
        res.append(RunDownbeatsTFModel(xData, outputFilePath, useLSTMBlockFusedCell))
        
    print('all model cost', time.time() - startTime)

    predict = madmom.ml.nn.average_predictions(res)
    print('DownbeatsTFModel predict shape', np.shape(predict))

    from functools import partial
    act = partial(np.delete, obj=0, axis=1)
    downBeatOutput = act(predict)
    DownbeatTracking.SaveInstantValue(downBeatOutput[:, 0], audioFilePath, '_rnn_dst_0')
    DownbeatTracking.SaveInstantValue(downBeatOutput[:, 1], audioFilePath, '_rnn_dst_1')

    CompareData(downBeatOutput, srcRes)

    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(audioFilePath, sampleRate)
    bpm, et = NotePreprocess.CalcBpmET(audioData, sampleRate, len(audioData) / sampleRate, downBeatOutput, downBeatOutput)
    print('bpm', bpm, 'et', et)
    srcBpm, srcEt = NotePreprocess.CalcBpmET(audioData, sampleRate, len(audioData) / sampleRate, srcRes, srcRes)
    print('src bpm', srcBpm, 'src et', srcEt)
    return True

if __name__ == '__main__':
    RunAllDownbeatsTFModel(TFLstm.MakeMp3Pathname('ouxiangwanwansui'))
    print('TestCase end')
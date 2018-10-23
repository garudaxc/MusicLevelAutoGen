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
import DownbeatTracking

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
    dis = np.sum(np.square(subVal))
    print('arrA to arrB dis2 %f, aveDis %f, maxDis %f, minDis %f' % (dis, aveDis, maxDis, minDis))

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
    startFuncTime = time.time()
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
    numUnits = 25

    graph = tf.Graph()
    variableScopeName = os.path.splitext(os.path.basename(modelFilePath))[0]
    with graph.as_default():
        tensorDic = ModelTool.BuildDownbeatsModelGraph(variableScopeName, 3, batchSize, maxTime, numUnits, inputDim, usePeepholes, [numUnits * 2, 3], [3], tf.nn.softmax, useLSTMBlockFusedCell)
        with tf.Session() as sess:
            # sess.run([tf.global_variables_initializer()])
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
    
    if useLSTMBlockFusedCell:
        res = np.reshape(res, (maxTime, batchSize, -1))
        res = res.transpose((1, 0, 2))
        res = res[0]
    else:
        res = np.reshape(res, (batchSize, maxTime, -1))
        res = res[0]

    print('model func cost', time.time() - startFuncTime)
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
    DownbeatTracking.SaveInstantValue(srcRes[:, 0], audioFilePath, '_rnn_src_0')
    DownbeatTracking.SaveInstantValue(srcRes[:, 1], audioFilePath, '_rnn_src_1')
    
    batchSize = 1
    maxTime = len(specDiff)
    inputDim = len(specDiff[0])

    xData = [specDiff] * batchSize
    xData = np.reshape(xData, (batchSize, maxTime, inputDim))

    useLSTMBlockFusedCell = True
    if useLSTMBlockFusedCell:
        xData = xData.transpose((1, 0, 2))
    else:
        pass

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

def RunOnsetModel():
    from madmom.ml.nn import NeuralNetwork
    from madmom.models import ONSETS_CNN
    nn = NeuralNetwork.load(ONSETS_CNN[0])
    layers = nn.layers

    testData = np.zeros((100, 80, 3))

    songFilePath = TFLstm.MakeMp3Pathname('ouxiangwanwansui')
    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(songFilePath, sampleRate)
    audioData = np.array(audioData)
    specDiff, melLogSpec = NotePreprocess.SpecDiffAndMelLogSpecEx(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
    specDiffSrc, melLogSpecSrc = NotePreprocess.SpecDiffAndMelLogSpec(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
    testData = melLogSpec

    madmomStartTime = time.time()
    madmomLayersRes = [testData]
    for layer in nn.layers:
        madmomLayersRes.append(layer(madmomLayersRes[-1]))
    madmomLayersRes = madmomLayersRes[1:]
    madmomModelRes = np.reshape(madmomLayersRes[-1], [-1])
    print('madmom cost', time.time() - madmomStartTime)

    # ModelTool.ConvertMadmomOnsetModelToTensorflow(nn)

    modelStart = time.time()
    graph = tf.Graph()
    variableScopeName = 'onset'
    modelFilePath = ModelTool.GenerateOutputModelPath(variableScopeName)
    with graph.as_default():
        tensorDic = ModelTool.BuildOnsetModelGraph(variableScopeName)
        with tf.Session() as sess:
            # sess.run([tf.global_variables_initializer()])
            varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variableScopeName)
            saver = tf.train.Saver(var_list=varList)
            saver.restore(sess, modelFilePath)
            
            X = tensorDic['X']
            output = tensorDic['output']

            startTime = time.time()
            allres = sess.run([output, tensorDic['layer_0'], tensorDic['layer_1'], tensorDic['layer_2'], tensorDic['layer_3'], tensorDic['layer_4'], tensorDic['layer_5'], tensorDic['layer_6'], tensorDic['layer_7']], feed_dict={X:testData})
            tfModelRes = allres[0]
            tfLayersRes = allres[1:]
            print('model real cost', time.time() - startTime)
            print('sub res shape', np.shape(tfModelRes))

    print('model all cost', time.time() - modelStart)

    for idx, tfRes, madRes in zip(range(len(tfLayersRes)), tfLayersRes, madmomLayersRes):
        print('compare layer idx', idx)
        print('layer res shape', np.shape(tfRes), np.shape(madRes))
        CompareData(np.reshape(tfRes, [-1]), np.reshape(madRes, [-1]))

    CompareData(tfModelRes, madmomModelRes)
    DownbeatTracking.SaveInstantValue(madmomModelRes, songFilePath, '_onset_src')
    DownbeatTracking.SaveInstantValue(tfModelRes, songFilePath, '_onset_dst')
    return True

def RunSess(sess, runArr, signalTensorArr, dataArr):
    time1 = time.time()
    dic = {}
    for signalTensor, data in zip(signalTensorArr, dataArr):
        dic[signalTensor] = data

    res = sess.run(runArr, feed_dict=dic)
    print('cost', time.time() - time1)
    return res

def RunAudioPreprocess():
    song = 'ouxiangwanwansui'
    audioFilePath = TFLstm.MakeMp3Pathname('ouxiangwanwansui')
    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(audioFilePath, sampleRate)
    fps = 100
    hopSize = int(sampleRate / fps)
    frameCount = ModelTool.FrameCount(audioData, hopSize)
    scaleValue = ModelTool.ScaleValue(audioData)
    tfAudioData = audioData / scaleValue

    print('frames info', np.shape(audioData), sampleRate / fps, len(audioData) / (sampleRate / fps))
    
    frameSizeArr = [1024, 2048, 4096]
    numBandArr = [3, 6, 12]
    
    timeMadmomStart = time.time()
    madmomSTFTRes = NotePreprocess.STFT(audioData, frameSizeArr, fps)
    madmomLogMelRes = NotePreprocess.MelLogarithmicSpectrogram(madmomSTFTRes)
    madmomSpecDiffRes = NotePreprocess.SpectrogramDifference(madmomSTFTRes, numBandArr)
    timeMadmomEnd = time.time()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            signalTensorArr, runTensorArr, diffFrameArr = ModelTool.BuildPreProcessGraph('preprocess', sampleRate, frameSizeArr, numBandArr, hopSize)
            
            sess.run([tf.global_variables_initializer()])
            
            tfAudioData = tfAudioData
            tfAudioDataArr = ModelTool.PaddingAudioData(tfAudioData, frameSizeArr)
            res = RunSess(sess, runTensorArr, signalTensorArr, tfAudioDataArr)
            res = RunSess(sess, runTensorArr, signalTensorArr, tfAudioDataArr)
            timePostProcessStart = time.time()
            tfLogMel = []
            tfSpecDiff = []
            for idx in range(0, len(res), 2):
                tfLogMel.append(res[idx])
                tfSpecDiff.append(res[idx + 1])

    tfLogMel = ModelTool.PostProcessTFLogMel(tfLogMel, frameCount)
    tfSpecDiff = ModelTool.PostProcessTFSpecDiff(tfSpecDiff, frameCount, diffFrameArr)
    print('shape log mel', np.shape(madmomLogMelRes), np.shape(tfLogMel))
    print('shape spec diff', np.shape(madmomSpecDiffRes), np.shape(tfSpecDiff))
    timePostProcessEnd = time.time()

    CompareData(madmomLogMelRes, tfLogMel)
    CompareData(madmomSpecDiffRes, tfSpecDiff)
    # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'log_mel_src'), np.reshape(madmomLogMelRes, (len(madmomLogMelRes), -1)))
    # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'log_mel_dst'), np.reshape(tfLogMel, (len(tfLogMel), -1)))
    # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'spec_diff_src'), madmomSpecDiffRes)
    # TFLstm.SaveFeatures(TFLstm.MakeSongDataPathName(song, 'spec_diff_dst'), tfSpecDiff)

    print('cost madmom', timeMadmomEnd - timeMadmomStart)
    print('cost post process', timePostProcessEnd - timePostProcessStart)

if __name__ == '__main__':
    RunAudioPreprocess()
    print('TestCase end')
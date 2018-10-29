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
import NoteLevelGenerator
import LevelInfo

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
    
    maxA = np.max(reshapeA)
    maxB = np.max(reshapeB)
    minA = np.min(reshapeA)
    minB = np.min(reshapeB)
    subVal = np.abs(reshapeA - reshapeB)
    maxDis = np.max(subVal)
    minDis = np.min(subVal)
    aveDis = np.average(subVal)
    dis = np.sum(np.square(subVal))
    print('arrA to arrB dis2 %f, aveDis %f, maxDis %f, minDis %f' % (dis, aveDis, maxDis, minDis))
    print('range', minA, maxA, minB, maxB)

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

    # mergeSize = 8
    # inputDim = inputDim * mergeSize
    # numUnits = numUnits * mergeSize
    # xData = np.random.rand(6000 * 4, 1, inputDim)

    graph = tf.Graph()
    variableScopeName = os.path.splitext(os.path.basename(modelFilePath))[0]
    with graph.as_default():
        tensorDic = ModelTool.BuildDownbeatsModelGraph(variableScopeName, 3, batchSize, numUnits, inputDim, usePeepholes, [numUnits * 2, 3], [3], tf.nn.softmax, useLSTMBlockFusedCell)
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
    audioFilePath = TFLstm.MakeMp3Pathname(song)
    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(audioFilePath, sampleRate)
    fps = 100
    hopSize = int(sampleRate / fps)
    frameCount = ModelTool.FrameCount(audioData, hopSize)
    scaleValue = ModelTool.ScaleValue(audioData)

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
            signalTensor, runTensorArr, diffFrameArr, logMelSpecTensor, specDiffTensor, frameCountTensor, scaleValueTensor = ModelTool.BuildPreProcessGraph('preprocess', sampleRate, frameSizeArr, numBandArr, hopSize)
            
            sess.run([tf.global_variables_initializer()])

            inputTensorArr = [signalTensor, frameCountTensor, scaleValueTensor]
            inputDataArr = [audioData, frameCount, scaleValue]
            res = RunSess(sess, runTensorArr, inputTensorArr, inputDataArr)
            res = RunSess(sess, runTensorArr, inputTensorArr, inputDataArr)
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

def TestAudioSpec():
    import TFLstm
    import NotePreprocess
    import time
    audioFilePath = TFLstm.MakeMp3Pathname('ouxiangwanwansui')
    sampleRate = 44100
    audioData = NotePreprocess.LoadAudioFile(audioFilePath, sampleRate)
    frameSize = 4096
    fps = 100
    # audioData = audioData[0:int(sampleRate * 10)]
    scaleValue = ModelTool.ScaleValue(audioData)
    tfAudioData = audioData / scaleValue
    hopSize = int(sampleRate / fps)
    print('frames info', np.shape(audioData), sampleRate / fps, len(audioData) / (sampleRate / fps))
    time1 = time.time()
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
        SpectrogramDifferenceProcessor)
    madmomSTFTRes = NotePreprocess.STFT(audioData, [frameSize], fps)
    filt = FilteredSpectrogramProcessor(filterbank=madmom.audio.filters.MelFilterbank, num_bands=80, fmin=27.5, fmax=16000, norm_filters=True, unique_filters=False)
    spec = LogarithmicSpectrogramProcessor(log=np.log, add=madmom.features.onsets.EPSILON)
    print('type of madmom.features.onsets.EPSILON', type(madmom.features.onsets.EPSILON))
    filtRes = filt(madmomSTFTRes[0])
    madmomRes = spec(filtRes)
    madmomRes = madmom.features.onsets._cnn_onset_processor_pad(np.dstack([madmomRes]))
    # madmomRes = NotePreprocess.MelLogarithmicSpectrogram(madmomSTFTRes)
    time2 = time.time()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            signalTensor = tf.placeholder(tf.float32, shape=[None])
            frameSize = int(frameSize)
            fftLength = None
            stft = ModelTool.TFSTFT(signalTensor, frameSize, hopSize)
            # madmom 没用最后一个 nyquist hertz
            stft = stft[:, :-1]

            magnitudeSpectrogram = tf.abs(stft)
            melFilterbankWeights = ModelTool.MelFilterbankWeight(sampleRate, frameSize // 2, 80, 27.5, 16000)
            melWeight = tf.constant(melFilterbankWeights)
            melSpectrogram = tf.matmul(magnitudeSpectrogram, melWeight)
            logMelSpec = melSpectrogram + madmom.features.onsets.EPSILON
            logMelSpec = tf.log(logMelSpec)

            sess.run([tf.global_variables_initializer()])

            tfAudioDataArr = ModelTool.PaddingAudioData(tfAudioData, [frameSize])
            res = RunSess(sess, [logMelSpec, magnitudeSpectrogram, melSpectrogram], [signalTensor], tfAudioDataArr)
            tfRes = res[0]
            magnitudeSpectrogramRes = res[1]
            melSpectrogramRes = res[2]

    madmomRes = np.reshape(madmomRes, np.shape(madmomRes)[0:2])
    frameCount = ModelTool.FrameCount(audioData, hopSize)
    tfRes = tfRes[0:frameCount, :]
    tfRes = np.dstack([tfRes])
    tfRes = madmom.features.onsets._cnn_onset_processor_pad(tfRes)
    tfRes = np.reshape(tfRes, np.shape(tfRes)[0:2])

    CompareData(madmomRes, tfRes)

    tempSTFT1 = np.array(np.abs(madmomSTFTRes[0]))
    tempSTFT2 = magnitudeSpectrogramRes[0:frameCount, :]
    tempMelSpec1 = filtRes
    tempMelSpec2 = melSpectrogramRes[0:frameCount, :]
    CompareData(tempSTFT1, tempSTFT2)
    CompareData(tempMelSpec1, tempMelSpec2)

    print('shape', np.shape(madmomRes), np.shape(tfRes))
    print('cost madmom', time2 - time1)

def RunNoteLevelGenerator():
    generator = NoteLevelGenerator.NoteLevelGenerator()
    if not generator.initialize():
        return False

    songArr = ['ouxiangwanwansui', 'jinzhongzhao', 'gundong', 'baaifangkai']
    for song in songArr:
        audioFilePath = TFLstm.MakeMp3Pathname(song)
        levelFilePath = os.path.join(os.path.dirname(audioFilePath), song+'.xml')
        runStart = time.time()
        generator.run(audioFilePath, levelFilePath, isTranscodeByQAAC=True, outputDebugInfo=True, saveDebugFile=True)
        print('cost _______ ', song, time.time() - runStart)

    generator.releaseResource()

    return True

def BPMAndETTestCallback(dic, runCallbackParam=None):
    levelBPM = runCallbackParam[0]
    levelET = runCallbackParam[1]
    resArr = runCallbackParam[2]

    generator = dic['generator']
    specDiff = dic['specDiff']
    generatorBpm = dic['bpm']
    generatorET = dic['et']
    audioData = dic['audioData']
    sampleRate = dic['sampleRate']
    isTranscodeByQAAC = dic['isTranscodeByQAAC']
    bpmModelRes = dic['bpmModelRes']
    audioFilePath = dic['audioFilePath']

    proc = NotePreprocess.CustomRNNDownBeatProcessor()
    srcRes = proc(specDiff)
    duration = len(audioData) / sampleRate
    analysisRange = generator.getBPMAnalysisRange(duration)
    srcBPM, srcET = NotePreprocess.CalcBpmET(audioData, sampleRate, duration, srcRes, srcRes, analysisRange)
    if isTranscodeByQAAC:
        srcET = srcET + NotePreprocess.DecodeOffset(audioFilePath)
    srcET = int(srcET * 1000)
    audioFileName = os.path.basename(audioFilePath)
    print('song', audioFileName)
    print('level    ', [levelBPM, levelET])
    print('generator', [generatorBpm, generatorET])
    print('old one  ', [srcBPM, srcET])

    resArr.append([levelBPM, levelET, generatorBpm, generatorET, srcBPM, srcET, audioFileName])

    # DownbeatTracking.SaveInstantValue(bpmModelRes[:, 0], audioFilePath, '_bpm_split_0_%d_overlap_%d' % (generator.bpmModelBatchSize, generator.bpmModelOverlap))
    # DownbeatTracking.SaveInstantValue(bpmModelRes[:, 1], audioFilePath, '_bpm_split_1_%d_overlap_%d' % (generator.bpmModelBatchSize, generator.bpmModelOverlap))
    # DownbeatTracking.SaveInstantValue(srcRes[:, 0], audioFilePath, '_bpm_src_0')
    # DownbeatTracking.SaveInstantValue(srcRes[:, 1], audioFilePath, '_bpm_src_1')

def RunBPMAndETTest():
    resourceDir = 'F:/p4resroot/H3D_X51_res/QQX5_Mainland/trunc/exe/resources'
    levelDir = os.path.join(resourceDir, 'level', 'game_level')
    audioDir = os.path.join(resourceDir, 'media', 'audio', 'Music')
    levelFileList = os.listdir(levelDir)
    levelInfoArr = LevelInfo.load_levels(levelDir, getSongFileName=True)
    testCaseArr = []
    for info in levelInfoArr:
        print('pre process', info['id'])
        audioFilePath = os.path.join(audioDir, info['songFileName'])
        if not os.path.exists(audioFilePath):
            print('audio file not exist', info['songFileName'])
            continue
        
        etVal = float(info['et'])
        testCaseArr.append((audioFilePath, float(info['bpm']), int(etVal)))

    generator = NoteLevelGenerator.NoteLevelGenerator()
    if not generator.initialize(runCallbackFunc=BPMAndETTestCallback):
        return False

    count = len(testCaseArr)
    resArr = []
    for idx, (audioFilePath, bpm, et) in enumerate(testCaseArr):
        print('%d/%d %s' % (idx + 1, count, audioFilePath))
        generator.run(audioFilePath, '', isTranscodeByQAAC=True, outputDebugInfo=True, runCallbackParam=(bpm, et, resArr))

    generator.releaseResource()

    resArr = np.array(resArr)
    rootDir = util.getRootDir()
    bpmTestResPath = os.path.join(rootDir, 'bpm_test_result.csv')
    TFLstm.SaveFeatures(bpmTestResPath, resArr)

    levelBPMArr = resArr[:, 0].astype(float)
    levelETArr = resArr[:, 1].astype(int)
    generatorBpmArr = resArr[:, 2].astype(float)
    generatorET = resArr[:, 3].astype(int)
    srcBPM = resArr[:, 4].astype(float)
    srcET = resArr[:, 5].astype(int)
    CompareData(levelBPMArr, generatorBpmArr)
    CompareData(levelBPMArr, srcBPM)
    CompareData(generatorBpmArr, srcBPM)
    CompareData(levelETArr, generatorET)
    CompareData(levelETArr, srcET)
    CompareData(generatorET, srcET)
    return True


if __name__ == '__main__':
    # RunBPMAndETTest()
    RunNoteLevelGenerator()
    # RunOnsetModel()
    # RunAudioPreprocess()
    # RunAllDownbeatsTFModel(TFLstm.MakeMp3Pathname('ouxiangwanwansui'))
    print('TestCase end')
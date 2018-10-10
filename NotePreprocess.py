import madmom
from madmom.features.downbeats import *
import librosa
import numpy as np
import time
import scipy
import os.path

from madmom.processors import ParallelProcessor, Processor, SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
from functools import partial
from madmom.models import ONSETS_CNN
from madmom.ml.nn import NeuralNetwork

import DownbeatTracking
import multiprocessing as mp
import math

def MSL(beats):
    # 最小二乘直线拟合
    numBeats = len(beats)

    AT = np.matrix([np.ones(numBeats), range(numBeats)])
    b = np.matrix(beats)
    x = (AT * AT.T).I * AT * b.T

    a = x[1, 0]
    b = x[0, 0]

    beat_times2 = np.arange(numBeats)
    beat_times2 = beat_times2 * a

    return a, b

def beat_intervals(samples):
    # 计算间隔
    numSamples = len(samples)
    t = np.zeros(numSamples-1)
    for i in range(numSamples-1):
        t[i] = samples[i+1] - samples[i]
        
    mean = sum(t) / len(t)
    return t


def min_measure(mean, dist):
    index = 0
    measure = np.inf
    for i in range(len(mean)):
        m = mean[i] * mean[i] + dist[i]
        if m < measure:
            measure = m
            index = i
    return measure, index


def diff(beats, a):
    winSize = 32
    numBeat = len(beats) - winSize
    mean = np.zeros(numBeat)
    dist = np.zeros(numBeat)
    for i in range(numBeat):
        v = beats[i:i+winSize]
        mean[i] = sum(v) / float(winSize) - a
        v = v - a
        dist[i] = sum(v * v)

    cnt = int(numBeat / 3)
    
    m, i1 = min_measure(mean[:cnt], dist[:cnt])
    m, i2 = min_measure(mean[-cnt:], dist[-cnt:])
    
    i2 = numBeat - cnt + i2
    return i1, i2
        
        
def calc_beat_interval(beats, i1, i2):
    size = 32
    bpm1 = (beats[i1 + size] - beats[i1]) / float(size)
    bpm2 = (beats[i2 + size] - beats[i2]) / float(size)
    bpm = (bpm1 + bpm2) / 2.0
    numbeat = (beats[i2] - beats[i1]) / bpm
    
    a2 = (beats[i2] - beats[i1]) / round(numbeat)    
    b1 = beats[i1] - a2 * i1
    b2 = beats[i2] - a2 * i2
    
    return a2, b1


def CalcBarInterval(beat_times):
    numBeats = len(beat_times)
    itvals = beat_intervals(beat_times)

    #最小二乘计算固定间隔拍子·
    a, b = MSL(beat_times)  

    # new_beat_times = np.arange(numBeats) * a + b
    return a, b


def NormalizeInterval(beat, threhold = 0.02, abThrehold=0.333):
    # 截掉前后不太准的beat
    interval = beat[1:,0] - beat[:-1, 0]

    aver = np.average(interval)
    diff = np.abs(interval - aver)

    abnormal = np.nonzero(diff > threhold)[0]

    if abnormal.size > beat.shape[0] * abThrehold:
        # 拍子均匀程度太低，自动生成失败
        return -1, abnormal.size / beat.shape[0]

    # #删除头尾不均匀的拍子
    # #从前往后删
    # count0 = 0
    # for i in range(len(abnormal)):
    #     if i == abnormal[i]:
    #         count0 = count0 + 1

    # #从后往前删
    # count1 = 0
    # for i in range(-1, -1-len(abnormal), -1):
    #     if len(interval) + i == abnormal[i]:
    #         count1 = count1 + 1

    return 0, len(beat)

# 新版本移除了这个函数，暂时替换一下，考虑用新版本的函数
def madmom_features_downbeats_filter_downbeats(beats):
    """

    Parameters
    ----------
    beats : numpy array, shape (num_beats, 2)
        Array with beats and their position inside the bar as the second
        column.

    Returns
    -------
    downbeats : numpy array
        Array with downbeat times.

    """
    # return only downbeats (timestamps)
    return beats[beats[:, 1] == 1][:, 0]
    
def AnalysisInfo(duration):
    analysisLength = 60
    start = int(duration * 0.3)
    if start + analysisLength > duration:
        start = duration - analysisLength
        start = max(start, 0)
        analysisLength = min(analysisLength, duration)

    return start, analysisLength

def CalcBpmET(y, sr, duration, preCalcData = None, downBeatData = None):
    # calc downbeat entertime
    start, analysisLength = AnalysisInfo(duration)
    minimumMusicLength = 110
    maximumMusicLength = 360
    numThread = 8
    threhold = 0.02
    abThrehold = 0.333
        
    clipTime = np.array([start, start+analysisLength])      
    clip = librosa.time_to_samples(clipTime, sr=sr)
    yy = y[clip[0]:clip[1]]

    fps = 100
    if preCalcData is None:
        print('use RNNDownBeatProcessor')
        processor = RNNDownBeatProcessor(num_threads=numThread)
        beatProba = processor(yy)
    elif downBeatData is None:
        print('use CustomRNNDownBeatProcessor')
        processor = CustomRNNDownBeatProcessor(num_threads=numThread)
        startIdx = int(start * fps)
        endIdx = startIdx + int(analysisLength * fps)
        clipPreCalcData = preCalcData[startIdx:endIdx]
        beatProba = processor(clipPreCalcData)

    else:
        beatProba = downBeatData
    print('beatProb shape', len(beatProba))
    downbeatTracking = DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda=1000, fps=fps, num_threads=numThread)
    beatIndex = downbeatTracking(beatProba)
    
    firstBeat, lastBeat = NormalizeInterval(beatIndex, threhold=threhold, abThrehold=abThrehold)
    if firstBeat == -1:        
        return 0, 0

    newBeat = beatIndex[firstBeat:lastBeat]
    downbeat = madmom_features_downbeats_filter_downbeats(newBeat)
    downbeat = downbeat + librosa.samples_to_time(clip, sr = sr)[0]

    barInter, etAuto = CalcBarInterval(downbeat)
    
    # enter time is less than a bar interval
    etAuto = etAuto % barInter

    bpm = 240.0 / barInter

    return bpm, etAuto

def SplitData(srcArr, splitCount, overlap):
    arrLength = len(srcArr)
    if splitCount > arrLength or arrLength == 0 or overlap < 0:
        return None

    arr = srcArr
    subArrDataLength = int(math.ceil(arrLength / splitCount))
    allLength = subArrDataLength * splitCount
    if allLength > splitCount:
        appendShape = np.concatenate(([allLength - splitCount], np.shape(arr[0])))
        appendArr = np.zeros(appendShape)
        arr = np.concatenate((arr, appendArr))

    if overlap > 0:
        overlapShape = np.concatenate(([overlap], np.shape(arr[0])))
        overlapArr = np.zeros(overlapShape)
        arr = np.concatenate((overlapArr, arr, overlapArr))

    res = []
    for i in range(splitCount):
        start = overlap + i * subArrDataLength
        end = start + subArrDataLength
        subArr = arr[(start - overlap) : (end + overlap)]
        res.append(subArr)

    return np.array(res)

class DataInputProcessor(Processor):
    def __init__(self, data, **kwargs):
        self.data = data

    def process(self, data, **kwargs):
        return self.data

def LoadAudioFile(audioFilePath, sampleRate):
    signalProcessor = SignalProcessor(num_channels=1, sample_rate=sampleRate)
    return signalProcessor(audioFilePath)

def STFT(audioData, frameSizeArr, fps):
    multi = ParallelProcessor([], num_threads=len(frameSizeArr))
    for frameSize in frameSizeArr:
        frames = FramedSignalProcessor(frame_size=frameSize, fps=fps)
        stft = ShortTimeFourierTransformProcessor()
        multi.append(SequentialProcessor((frames, stft)))
        
    return multi(audioData)

def SpectrogramDifferenceProcArr(numBand):
    filt = FilteredSpectrogramProcessor(num_bands=numBand, fmin=30, fmax=17000, norm_filters=True)
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
    return [filt, spec, diff]

def MelLogSpecProcArr():
    filt = FilteredSpectrogramProcessor(filterbank=madmom.audio.filters.MelFilterbank, num_bands=80, fmin=27.5, fmax=16000, norm_filters=True, unique_filters=False)
    spec = LogarithmicSpectrogramProcessor(log=np.log, add=madmom.features.onsets.EPSILON)
    return [filt, spec]

def SpectrogramDifference(stftDataArr, numBandArr):
    multi = ParallelProcessor([], num_threads=len(stftDataArr))
    for stftData, numBand in zip(stftDataArr, numBandArr):
        filt = FilteredSpectrogramProcessor(num_bands=numBand, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        multi.append(SequentialProcessor((DataInputProcessor(stftData), filt, spec, diff)))

    proc = SequentialProcessor((multi, np.hstack))
    return proc(0)

class CustomRNNDownBeatProcessor(SequentialProcessor):
    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM, **kwargs)
        # use only the beat & downbeat (i.e. remove non-beat) activations
        act = partial(np.delete, obj=0, axis=1)
        # instantiate a SequentialProcessor
        super(CustomRNNDownBeatProcessor, self).__init__((nn, act))

class CustomCNNOnsetProcessor(SequentialProcessor):
    def __init__(self, stftDataArr, **kwargs):
        # pylint: disable=unused-argument
        # define pre-processing chain
        # process the multi-resolution spec in parallel
        multi = ParallelProcessor([])
        for stftData in stftDataArr:
            filt = FilteredSpectrogramProcessor(
                filterbank=madmom.audio.filters.MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
                norm_filters=True, unique_filters=False)
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=madmom.features.onsets.EPSILON)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((DataInputProcessor(stftData), filt, spec)))
        # stack the features (in depth) and pad at beginning and end
        stack = np.dstack
        pad = madmom.features.onsets._cnn_onset_processor_pad
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor((multi, stack, pad))

        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetwork.load(ONSETS_CNN[0])

        # instantiate a SequentialProcessor
        super(CustomCNNOnsetProcessor, self).__init__((pre_processor, nn))

class NoteModelProcessor(Processor):
    def __init__(self, runFunc, songFile, modelFile, TrainData, modelParam, xData, **kwargs):
        self.runFunc = runFunc
        self.songFile = songFile
        self.modelFile = modelFile
        self.TrainData = TrainData
        self.modelParam = modelParam
        self.xData = xData

    def process(self, data, **kwargs):
        startTime = time.time()
        res = self.runFunc(self.songFile, self.modelFile, self.TrainData, self.modelParam, self.xData)
        print('model proc cost', time.time() - startTime)
        return res

def MelLogarithmicSpectrogram(stftDataArr):
    multi = ParallelProcessor([], num_threads=len(stftDataArr))
    for stftData in stftDataArr:
        filt = FilteredSpectrogramProcessor(filterbank=madmom.audio.filters.MelFilterbank, num_bands=80, fmin=27.5, fmax=16000, norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=madmom.features.onsets.EPSILON)
        multi.append(SequentialProcessor((DataInputProcessor(stftData), filt, spec)))

    stack = np.dstack
    pad = madmom.features.onsets._cnn_onset_processor_pad

    proc = SequentialProcessor((multi, stack, pad))
    return proc(0)

def SpecDiffAndMelLogSpec(audioData, sampleRate, frameSizeArr, numBandArr, fps):
    arr = []
    for frameSize, numBand in zip(frameSizeArr, numBandArr):
        signalProcessor = SignalProcessor(num_channels=1, sample_rate=sampleRate)
        frames = FramedSignalProcessor(frame_size=frameSize, fps=fps)
        stft = ShortTimeFourierTransformProcessor()
        subProc = ParallelProcessor([SequentialProcessor(SpectrogramDifferenceProcArr(numBand)), SequentialProcessor(MelLogSpecProcArr())])
        arr.append([signalProcessor, frames, stft, subProc])

    multi = ParallelProcessor(arr, num_threads=len(arr))
    res = multi(audioData)
    
    resA = [res[0][0], res[1][0], res[2][0]]
    # onset 的顺序和specdiff的不一样
    resB = [res[1][1], res[0][1], res[2][1]]
    specDiff = np.hstack(resA)
    melLogSpec = madmom.features.onsets._cnn_onset_processor_pad(np.dstack(resB))
    return specDiff, melLogSpec

def SpecDiffAndMelLogSpecEx(audioData, sampleRate, frameSizeArr, numBandArr, fps, splitCount = 4):
    maxFrameSize = np.max(frameSizeArr)
    hopSize = int(sampleRate / fps)
    minOverlapFrameCount = math.ceil(maxFrameSize / hopSize)
    overlap = int(minOverlapFrameCount * hopSize)

    splitAudioDataArr = []
    frameCount = math.ceil(len(audioData) / float(hopSize))
    frameCountPerSlice = math.ceil(frameCount / float(splitCount))
    for idx in range(splitCount):
        start = idx * frameCountPerSlice * hopSize
        end = start + frameCountPerSlice * hopSize + hopSize * minOverlapFrameCount
        if idx > 0:
            start = start - hopSize * minOverlapFrameCount
        if idx == splitCount - 1:
            end = len(audioData)

        splitAudioDataArr.append(audioData[start:end])

    # splitAudioDataArr = SplitData(audioData, splitCount, overlap)
    splitFrameSizeArr = np.concatenate([frameSizeArr] * splitCount)
    splitNumBandArr = np.concatenate([numBandArr] * splitCount)

    arr = []
    for idx, frameSize, numBand in zip(range(len(splitFrameSizeArr)), splitFrameSizeArr, splitNumBandArr):
        inputProc = DataInputProcessor(splitAudioDataArr[idx // len(frameSizeArr)])
        signalProcessor = SignalProcessor(num_channels=1, sample_rate=sampleRate)
        frames = FramedSignalProcessor(frame_size=frameSize, fps=fps)
        stft = ShortTimeFourierTransformProcessor()
        subProc = ParallelProcessor([SequentialProcessor(SpectrogramDifferenceProcArr(numBand)), SequentialProcessor(MelLogSpecProcArr())])
        arr.append([inputProc, signalProcessor, frames, stft, subProc])

    processorNum = len(arr)
    multi = ParallelProcessor(arr, num_threads=processorNum)
    res = multi(0)

    def RemoveOverlap(arr, left, right):
        return arr[left:len(arr)-right]

    resA = []
    resB = []
    for i in range(len(frameSizeArr)):
        tempA = []
        tempB = []
        for j in range(splitCount):
            left = minOverlapFrameCount
            right = minOverlapFrameCount
            if j == 0:
                left = 0
            if j == splitCount - 1:
                right = 0

            idx = i + j * len(frameSizeArr)
            removeA = RemoveOverlap(res[idx][0], left, right)
            removeB = RemoveOverlap(res[idx][1], left, right)
            # print('res', np.shape(removeA), np.shape(removeB))
            tempA.append(removeA)
            tempB.append(removeB)

        t = np.concatenate(tempA)
        resA.append(np.concatenate(tempA))
        resB.append(np.concatenate(tempB))
    
    # onset 的顺序和specdiff的不一样
    resB = [resB[1], resB[0], resB[2]]
    specDiff = np.hstack(resA)
    melLogSpec = madmom.features.onsets._cnn_onset_processor_pad(np.dstack(resB))
    return specDiff, melLogSpec

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

    for vA, vB in zip(reshapeA, reshapeB):
        if vA != vB:
            print('compare arr val failed')
            return False

    print('compare res true')
    return True


class CustomRNNDownBeatProcessorEx(SequentialProcessor):
    def __init__(self, idx, specDiff, **kwargs):
        # pylint: disable=unused-argument
        nn = NeuralNetwork.load(DOWNBEATS_BLSTM[idx])
        super(CustomRNNDownBeatProcessorEx, self).__init__((DataInputProcessor(specDiff), nn))

class CustomCNNOnsetProcessorEx(SequentialProcessor):
    def __init__(self, melLogSpec, **kwargs):
        # pylint: disable=unused-argument
        # stack the features (in depth) and pad at beginning and end
        # stack = np.dstack
        # pad = madmom.features.onsets._cnn_onset_processor_pad
        # pre-processes everything sequentially

        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetwork.load(ONSETS_CNN[0])

        # instantiate a SequentialProcessor
        super(CustomCNNOnsetProcessorEx, self).__init__((DataInputProcessor(melLogSpec), nn))
    def process(self, data, **kwargs):
        startTime = time.time()
        res = super().process(data, **kwargs)
        print('onset dectection proc cost', time.time() - startTime)
        return res

def ProcessFunc(func):
    return func(0)

class AllTaskProcessor(Processor):
    def __init__(self, shortParam, longParam, bpmParam, onsetParam, onsetSplitCount, **kwargs):
        self.shortParam = shortParam
        self.longParam = longParam
        self.bpmParam = bpmParam
        self.onsetParam = onsetParam
        self.onsetSplitCount = onsetSplitCount     

    def process(self, data, **kwargs):
        startTime = time.time()
        shortParam = self.shortParam
        longParam = self.longParam
        bpmParam = self.bpmParam
        onsetParam = self.onsetParam
        bpmModelCount = 8
        onsetSplitCount = self.onsetSplitCount
        processCount = 2 + onsetSplitCount + bpmModelCount
        pool = mp.Pool(processCount)

        onsetOverlap = 0
        onsetProcessorCutFrame = 0
        if onsetSplitCount <= 1:
            resOnsetArr = [pool.apply_async(CustomCNNOnsetProcessorEx(onsetParam[0]), [0])]
        else:
            onsetOverlap = 128
            onsetProcessorCutFrame = 14
            onsetInputArr = SplitData(onsetParam[0], onsetSplitCount, onsetOverlap)
            resOnsetArr = []
            for subInput in onsetInputArr:
                resOnset = pool.apply_async(CustomCNNOnsetProcessorEx(subInput), [0])
                resOnsetArr.append(resOnset)

        resShort = pool.apply_async(NoteModelProcessor(shortParam[0], shortParam[1], shortParam[2], shortParam[3], shortParam[4], shortParam[5]), [0])
        resLong = pool.apply_async(NoteModelProcessor(longParam[0], longParam[1], longParam[2], longParam[3], longParam[4], longParam[5]), [0])
        
        bpmProcArr = []
        preCalcData = bpmParam[0]
        start, analysisLength = AnalysisInfo(bpmParam[3])
        fps = 100
        startIdx = int(start * fps)
        endIdx = startIdx + int(analysisLength * fps)
        clipPreCalcData = preCalcData[startIdx:endIdx]
        for i in range(bpmModelCount):
            bpmProcArr.append(CustomRNNDownBeatProcessorEx(i, clipPreCalcData))

        bpmStartTime = time.time()
        subRes = pool.map(ProcessFunc, bpmProcArr)
        pool.close()

        predict = madmom.ml.nn.average_predictions(subRes)
        act = partial(np.delete, obj=0, axis=1)
        downBeatOutput = act(predict)
        bpm, et = CalcBpmET(self.bpmParam[1], self.bpmParam[2], self.bpmParam[3], downBeatOutput, downBeatOutput)
        duration = int(self.bpmParam[3] * 1000)
        print('origin et', et)
        et = et + DownbeatTracking.DecodeOffset(self.bpmParam[4])
        print('decode offset et', et)
        et = int(et * 1000)
        bpmEndTime = time.time()
        print('bpm task cost', bpmEndTime - bpmStartTime)

        pool.join()
        shortPredicts = resShort.get()
        longPredicts = resLong.get()

        onsetActivation = np.array([], dtype='float32')
        for idx, resOnset in zip(range(onsetSplitCount), resOnsetArr):
            subActivation = resOnset.get()
            cutFrame = onsetProcessorCutFrame
            if idx == onsetSplitCount - 1:
                cutFrame = 0
            subActivation = subActivation[onsetOverlap : len(subActivation) - onsetOverlap + cutFrame]
            onsetActivation = np.concatenate((onsetActivation, subActivation))

        endTime = time.time()
        print('all task cost', endTime - startTime)
        return shortPredicts, longPredicts, onsetActivation, duration, bpm, et
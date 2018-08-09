# -*- coding: utf-8 -*-

import madmom
from madmom.features.downbeats import *
import librosa
import numpy as np
import time
import sklearn.cluster
import scipy
import os.path
import QQDancerLog
import LevelMaker
import bisect
import random
import xml

# 错误 code
MUSIC_FILE_ERROR = 1    #打开音乐文件错误
MUSIC_TOO_LONG = 2      #音乐文件太长
MUSIC_TOO_SHORT = 3     #音乐文件太短
PROBABLY_NOT_MUSIC = 4  #音乐文件的内容可能不是歌曲
SEGMENT_ERROR = 5       #时长太短，或bpm太低，无法正常分段


logger = None
genDebugFile = False

# 允许的最小bpm，如果如果算得小于该值，则需要自动翻倍
MinimumBPM = 60

# 分段时可允许的偏移范围，范围越大，越容易分到音乐的变化处，但会导致段落的长度变化较大
Segment_Neighbourhood = 0.2

# 计算音乐bpm entertime和分段

def MSL(beats):
    # 最小二乘直线拟合
    numBeats = len(beats)

    AT = np.matrix([np.ones(numBeats), range(numBeats)])
    b = np.matrix(beats)
    x = (AT * AT.T).I * AT * b.T
    #logger.info(x)

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
    #logger.info(m, ' ', i1)    
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
    #logger.info('mean ' + str(mean))

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
    logger.info('abnormal count', abnormal.size, abnormal)

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
    # logger.info('del %d items form begining %d items form end' % (count0, count1))

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

def CalcDownbeat(y, sr, duration, **args):
    # calc downbeat entertime
    analysisLength = args['-duration']
    minimumMusicLength = 165
    maximumMusicLength = 360
    numThread = args['-thread']
    threhold = args['-threhold']
    abThrehold = args['-abThrehold']

    startTime = time.time()    

    if duration < minimumMusicLength:
        logger.error(MUSIC_TOO_SHORT, 'music is too short! duration:', duration)
        return 0, 0

    if duration > maximumMusicLength:
        logger.error(MUSIC_TOO_LONG, 'music is too long! duration:', duration)
        return 0, 0

    start = int(duration * 0.3)
    clipTime = np.array([start, start+analysisLength])      
    # logger.info('duration', duration, 'start', start)
    clip = librosa.time_to_samples(clipTime, sr=sr)
    # logger.info('total', y.shape, 'clip', clip)
    yy = y[clip[0]:clip[1]]

    processer = RNNDownBeatProcessor(num_threads=numThread)
    downbeatTracking = DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda=1000, fps=100)
    beatProba = processer(yy)

    beatIndex = downbeatTracking(beatProba)
    
    firstBeat, lastBeat = NormalizeInterval(beatIndex, threhold=threhold, abThrehold=abThrehold)
    if firstBeat == -1:        
        logger.error(PROBABLY_NOT_MUSIC, 'generate error,numbeat %d, abnormal rate %f' % (len(beatIndex), lastBeat))
        return 0, 0

    newBeat = beatIndex[firstBeat:lastBeat]
    downbeat = madmom_features_downbeats_filter_downbeats(newBeat)
    downbeat = downbeat + librosa.samples_to_time(clip, sr = sr)[0]

    barInter, etAuto = CalcBarInterval(downbeat)
    
    # enter time is less than a bar interval
    etAuto = etAuto % barInter

    bpm = 240.0 / barInter

    return bpm, etAuto
    

def LaplacianSegmentation(y, sr, beats, k = 4):
    # Next, we'll compute and plot a log-power CQT
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                ref=np.max)    
 
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                        sym=True)

    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # Msync = librosa.util.sync(mfcc, beats)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # And compute the balanced combination (Equations 6, 7, 9)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path

    # Now let's compute the normalized Laplacian (Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    X = evecs[:, :k] / Cnorm[:, k-1:k]

    # Let's use these k components to cluster beats into segments
    # (Algorithm 1)
    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = seg_ids[bound_beats]
    
    my_bound_beats = librosa.util.fix_frames(bound_beats, x_min=0, x_max = len(beats) - 1)
    # 确保分段数量之间的关系
    bound_segs = bound_segs[:my_bound_beats.shape[0]-1]
    
    my_bound_frames = beats[my_bound_beats]
    # Convert beat indices to frames
    bound_frames = beats[bound_beats]

    # Make sure we cover to the end of the track
    bound_frames = librosa.util.fix_frames(bound_frames,
                                        x_min=None,
                                        x_max=C.shape[1]-1)

    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    return my_bound_frames, bound_segs

# neighbourhood=0.167
def PickSegmentation(segments, duration, offset=0, neighbourhood=0.1):
    # 在三分之一和三分之二附近挑选两个分段点
    points = segments / duration
    offset = offset / duration
    def f(a):
        p = np.argmin(np.abs(points - a))
        p = points[p]
        if abs(p-a) > neighbourhood:
            p = a
        return p * duration

    p0 = (1.0 - offset) * 0.3333 + offset
    p1 = (1.0 - offset) * 0.6666 + offset
    result = np.array([f(p0), f(p1)])
    return result

def PickSegmentation2(pos, seg_points, neighbourhood):
    '''
    查找pos附近最近的点，范围不能超过neightbourhood
    '''
    p = np.argmin(np.abs(seg_points - pos))
    p = seg_points[p]
    if abs(p-pos) > neighbourhood * 0.5:
        p = None
    return p


def Segmentation(y, sr, duration, bpm, et):
    '''
    计算分段点
    分段最短长度16小节    
    '''
    beatInter = 60.0 / bpm
    barInterval = beatInter * 4
    numBeats = int((duration-et) / beatInter)
    beatTimes = np.arange(numBeats) * beatInter + et
    beatFrames = librosa.time_to_frames(beatTimes, sr=sr)

    logger.info('duration', duration, 'bpm', bpm)

    # 分段点要偏移四小节
    offset = (et + barInterval * 4) / duration
    # 允许的，最早的分段点
    mininumBarsPerSeg = (10 * barInterval) / duration
    firstSeg = offset + mininumBarsPerSeg
    logger.info('firstSeg', firstSeg)

    p0 = (1.0 - offset) * 0.3333 + offset
    p1 = (1.0 - offset) * 0.6666 + offset

    if firstSeg > p0:
        logger.error(SEGMENT_ERROR, 'first seg', firstSeg, 'seg0', p0, 'bpm', bpm)
        return None
            
    neighbourhood = min((p0 - firstSeg), Segment_Neighbourhood)
    logger.info('segment neighbourhood', neighbourhood)

    seg = [None, None]
    for k in range(3, 6):
        if seg[0] != None and seg[1] != None:
            break

        logger.info('LaplacianSegmentation for k=', k)

        seg_frames, _ = LaplacianSegmentation(y, sr, beatFrames, k = k)
        times = librosa.frames_to_time(seg_frames, sr=sr)

        if genDebugFile:
            name = r'd:\ab\QQX5_Mainland\exe\resources\media\audio\Music\seg%d.csv' % k
            SaveInstantValue(times, name, '_segment')

        seg_point = times / duration
        
        if seg[0] == None:
            seg[0] = PickSegmentation2(p0, seg_point, neighbourhood)

        if seg[0] != None and (seg[0] + mininumBarsPerSeg + neighbourhood * 0.5) > p1:
            p1 = seg[0] + mininumBarsPerSeg + neighbourhood * 0.5
        
        if seg[1] == None:
            seg[1] = PickSegmentation2(p1, seg_point, neighbourhood)
        
        logger.info('seg', seg)

    if seg[0] == None:
        seg[0] = p0

    if seg[1] == None:
        seg[1] = p1

    seg = np.array(seg) * duration    

    # align to downbeat
    segTimes = np.round((seg-et)/barInterval) * barInterval + et
    return segTimes

def SaveInstantValue(beats, filename, postfix = ''):
    import os
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True

def GenDebugFile(filename, duration, bpm, et, seg = None):
    
    def SaveInstantValue(beats, filename, postfix = ''):
        import os
        #保存时间点数据
        outname = os.path.splitext(filename)[0]
        outname = outname + postfix + '.csv'
        with open(outname, 'w') as file:
            for obj in beats:
                file.write(str(obj) + '\n')

        return True
        
    beatInter = 60.0 / bpm
    barInterval = beatInter * 4

    numDownbeats = int((duration-et) / barInterval)
    downBeatTimes = np.arange(numDownbeats) * barInterval + et
    SaveInstantValue(downBeatTimes, filename, '_downbeat')
    
    if seg is not None:
        SaveInstantValue(seg, filename, '_seg')


def AnalysisMusicFeature(filename, **args):
    '''
    计算音乐文件的特征数据
    返回值 tuple(a, b)
    a： True / False 计算成功还是失败，失败的原因有：
        音乐长度不合规范
        特征分析过程中发现可能不是音乐
    b:  是一个字典，里面记录具体特征数据，包括
        duration
        bpm
        EnterTime
        Seg0
        Seg1
    '''
    
    logger.info('start load', filename)
    t = time.time()
    try:
        y, sr = librosa.load(filename, mono=True, sr=44100)
    except:
        logger.error(MUSIC_FILE_ERROR, 'load music file error ', filename)
        return (False, None)

    logger.info('loaded', time.time() - t)
    
    duration = librosa.get_duration(y=y, sr=sr)

    bpm, et = CalcDownbeat(y, sr, duration, **args)
    if bpm == 0:
        return (False, None)        
    if bpm < MinimumBPM:
        logger.info('bpm too low')
        bpm = bpm * 2
            
    logger.info('analysis segmentation', time.time() - t)    
    segTimes = Segmentation(y, sr, duration, bpm, et)
    if segTimes[0] == 0:
        return (False, None)

    logger.info('AnalysisMusicFeature done', time.time() - t)
    result = {}
    result['duration'] = duration
    result['bpm'] = bpm
    result['EnterTime'] = et
    result['seg0'] = segTimes[0]
    result['seg1'] = segTimes[1]

    if genDebugFile:
        GenDebugFile(filename, duration, bpm, et, segTimes)

    GenerateNote(filename, duration, bpm, et, segTimes[0], segTimes[1], r'E:\work\dl\audio\proj\model\model_longnote.ckpt')

    return (True, result)

def PickOnset(onsetActivation, bpm, et, fps, segBegin, segEnd, count):
    threshold = 0.7
    checkFrameIdxArr = GenerateCheckFrameIdxArr(bpm, et, fps, segBegin, segEnd)
    onsetFrameIdxs = []
    offset = min(max(1, SecondToFrameIdx(60 / bpm / 8, fps)), 3)
    for arr in checkFrameIdxArr:
        onsetFrameIdx = []
        for frameIdx in arr:
            # if len(onsetTime) >= count:
            #     break

            begin = max(frameIdx - offset, 0)
            end = min(frameIdx + offset + 1, len(onsetActivation))
            for subIdx in range(begin, end):
                if onsetActivation[subIdx] >= threshold:
                    onsetFrameIdx.append(frameIdx)
                    break

        onsetFrameIdxs.append(onsetFrameIdx)

    
    def PickFromArr(onsetFrameIdx, curCount, targetCount):
        if curCount >= targetCount:
            return []

        if curCount + len(onsetFrameIdx) <= targetCount:
            return onsetFrameIdx

        remainCount= targetCount - curCount
        offset = len(onsetFrameIdx) / remainCount
        arr = []
        pickIdx = 0
        while pickIdx < len(onsetFrameIdx):
            idx = int(pickIdx)
            if len(arr) == 0 or arr[-1] != onsetFrameIdx[idx]:
                arr.append(onsetFrameIdx[idx])
            
            pickIdx += offset

        return arr

    pickIdx = []
    for onsetFrameIdx in onsetFrameIdxs:
        pickIdx = np.concatenate((pickIdx, PickFromArr(onsetFrameIdx, len(pickIdx), count)))

    pickIdx = np.sort(pickIdx)
    return pickIdx

def GenerateCheckFrameIdxArr(bpm, et, fps, segBegin, segEnd):
    beatInterval = 60 / bpm

    checkFrameIdxArrA = []
    checkFrameIdxArrB = []
    checkFrameIdxArrC = []

    beatOffset = int((segBegin - et) / beatInterval)
    # firstBeatTime = et + beatOffset * beatInterval
    firstBeatTime = et + beatOffset * beatInterval
    while firstBeatTime < segBegin:
        firstBeatTime += beatInterval

    curTime = firstBeatTime
    while curTime < segEnd:
        checkFrameIdxArrA.append(SecondToFrameIdx(curTime, fps))
        curTime += beatInterval

    curTime = firstBeatTime + beatInterval / 2
    while curTime < segEnd:
        checkFrameIdxArrB.append(SecondToFrameIdx(curTime, fps))
        curTime += beatInterval

    curTime = firstBeatTime + beatInterval / 4
    while curTime < segEnd:
        checkFrameIdxArrC.append(SecondToFrameIdx(curTime, fps))
        curTime += (beatInterval / 2)

    return [checkFrameIdxArrA, checkFrameIdxArrB, checkFrameIdxArrC]


def SecondToFrameIdx(time, fps):
    return int(time * fps)

def SegmentTimeToFrameIdx(playSegCount, begin, end, bpm, fps, frameCount):
    showTimeDuration = 60 / bpm * 4 * 4
    playOffset = (end - showTimeDuration - begin) / playSegCount
    timeArr = []
    lastEnd = begin
    for i in range(playSegCount):
        timeArr.append((lastEnd, begin + (i + 1) * playOffset))
        lastEnd = timeArr[-1][1]
    timeArr.append((lastEnd, end))
    
    frameArr = []
    for timeBegin, timeEnd in timeArr:
        frameBegin = max(SecondToFrameIdx(timeBegin, fps), 0)
        frameEnd = min(SecondToFrameIdx(timeEnd, fps), frameCount - 1)
        frameArr.append((frameBegin, frameEnd))

    return (frameArr[0:playSegCount], frameArr[playSegCount])


def SegmentTimesToFrameIdx(duration, bpm, et, seg0, seg1, playSegCount, fps, frameCount):
    segArr = []
    segArr.append(SegmentTimeToFrameIdx(playSegCount, et, seg0, bpm, fps, frameCount))
    segArr.append(SegmentTimeToFrameIdx(playSegCount, seg0, seg1, bpm, fps, frameCount))
    segArr.append(SegmentTimeToFrameIdx(playSegCount, seg1, duration, bpm, fps, frameCount))
    return segArr

def TransToNote(short, long):
    notes = []
    longBinary = long > 0
    edge = (longBinary[1:] != longBinary[:-1]).nonzero()[0] + 1
    i = 0
    while i < len(short):
        if long[i] > 0:
            end = bisect.bisect_right(edge, i)
            end = edge[end]
            notes.append((i, 1, end-i))
            i = end+1

        elif short[i] > 0:
            notes.append((i, 0, 0))
            i += 1
        else:
            i += 1

    return notes

def FrameToBarPos(frameIdx, fps, bpm, et, beatPerBar, beatLen):
    validPosPerBeat = 4
    validPosInterval = 60 / bpm / validPosPerBeat
    noteTime = frameIdx / fps
    tempPos = max(round((noteTime - et) / validPosInterval), 0)
    pos = tempPos % (validPosPerBeat * beatPerBar)
    bar = (tempPos - pos) // (validPosPerBeat * beatPerBar)
    pos = pos * (beatLen / validPosPerBeat)
    return bar, pos

def BarPosToFrame(bar, pos, fps, bpm, et, beatPerBar, beatLen):
    allPos = BarPosToAllPos(bar, pos, beatPerBar, beatLen)
    posInterval = 60 / bpm / beatLen
    noteTime = posInterval * allPos + et
    noteIdx = SecondToFrameIdx(noteTime, fps)
    return noteIdx

def FrameIdxToBeatPos(notes, fps, bpm, et, beatPerBar, beatLen):
    posNotes = []
    for i in range(len(notes)):
        if notes[i] > 0:
            posNotes.append(FrameToBarPos(i, fps, bpm, et, beatPerBar, beatLen))

    return posNotes

def SaveNotes(filePath, song, notes, seg0, seg1, bpm, et, beatPerBar, beatLen):
    def AppendChild(doc, node, name, text = None):
        childNode = doc.createElement(name)
        node.appendChild(childNode)
        if text is not None:
            childNode.appendChild(doc.createTextNode(str(text)))
        return childNode

    doc = xml.dom.minidom.Document()
    TangoMiddle = doc.createElement('TangoMiddle')
    doc.appendChild(TangoMiddle)
    AppendChild(doc, TangoMiddle, 'song', song)
    AppendChild(doc, TangoMiddle, 'bpm', bpm)
    AppendChild(doc, TangoMiddle, 'BeatPerBar', beatPerBar)
    AppendChild(doc, TangoMiddle, 'BeatLen', beatLen)
    AppendChild(doc, TangoMiddle, 'EnterTime', int(et * 1000))
    AppendChild(doc, TangoMiddle, 'Artist', 'xxx')
    AppendChild(doc, TangoMiddle, 'seg0', int(BarPosToAllPos(seg0[0], seg0[1], beatPerBar, beatLen)))
    AppendChild(doc, TangoMiddle, 'seg1', int(BarPosToAllPos(seg1[0], seg1[1], beatPerBar, beatLen)))
    NoteSequence = AppendChild(doc, TangoMiddle, 'NoteSequence')
    for idx in range(len(notes)):
        bar, pos = notes[idx]
        allpos = BarPosToAllPos(bar, pos, beatPerBar, beatLen)
        AppendChild(doc, NoteSequence, 'Note', int(allpos))

    with open(filePath, 'w') as file:
        doc.writexml(file, addindent='\t', newl='\n', encoding='UTF-8')

    return True

def BarPosToAllPos(bar, pos, beatPerBar, beatLen):
    return bar * beatPerBar * beatLen + pos

def AllPosToBarPos(allPos, beatPerBar, beatLen):
    pos = allPos % (beatPerBar * beatLen)
    bar = (allPos - pos) // (beatPerBar * beatLen)
    return bar, pos

def NoteType(bar, pos, beatLen):
    if pos % beatLen == 0:
        return 0

    if (pos - beatLen / 2) % beatLen == 0:
        return 1

    return 2

def PickNoteWithRule(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen):
    halfBeatLen = beatLen / 2
    quarterBeatLen = beatLen / 4
    # 按策划提供的规则筛选音符
    tempNoteIdxs = []
    continueHalfInterval = 0
    lastHalfIntervalAllPos = 0
    for noteIdx in noteIdxs:
        bar, pos = FrameToBarPos(noteIdx, fps, bpm, et, beatPerBar, beatLen)
        noteType = NoteType(bar, pos, beatLen)
        allPos = BarPosToAllPos(bar, pos, beatPerBar, beatLen)
        if noteType == 2:
            # 四分之一拍只能跟着半拍或整拍
            if len(tempNoteIdxs) <= 0:
                continue

            lastBar, lastPos = FrameToBarPos(tempNoteIdxs[-1], fps, bpm, et, beatPerBar, beatLen)
            if allPos - BarPosToAllPos(lastBar, lastPos, beatPerBar, beatLen) == quarterBeatLen:
                tempNoteIdxs.append(noteIdx)
        else:
            # 连续的半拍间隔最多6个
            if continueHalfInterval >= 6:
                continueHalfInterval = 0
                continue
            
            if allPos - lastHalfIntervalAllPos != halfBeatLen:
                continueHalfInterval = 0
            
            continueHalfInterval += 1
            tempNoteIdxs.append(noteIdx)
            lastHalfIntervalAllPos = allPos

    noteIdxs = tempNoteIdxs
    tempNoteIdxs = []
    lastQuarterIntervalAllPos = 0
    for noteIdx in noteIdxs:
        bar, pos = FrameToBarPos(noteIdx, fps, bpm, et, beatPerBar, beatLen)
        noteType = NoteType(bar, pos, beatLen)
        allPos = BarPosToAllPos(bar, pos, beatPerBar, beatLen)
        if noteType == 2:
            if len(tempNoteIdxs) > 0:
                if lastQuarterIntervalAllPos > 0:
                    if allPos - lastQuarterIntervalAllPos >= beatLen + halfBeatLen:
                        lastQuarterIntervalAllPos = allPos
                        tempNoteIdxs.append(noteIdx)
                else:
                    lastQuarterIntervalAllPos = allPos
                    tempNoteIdxs.append(noteIdx)
        else:
            tempNoteIdxs.append(noteIdx)

    noteIdxs = tempNoteIdxs
    return noteIdxs

def PickNoteRandom(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen):
    if len(noteIdxs) <= count:
        return noteIdxs

    removeCount = len(noteIdxs) - count
    pickIdxs = []
    notPickIdxs = []
    while len(pickIdxs) < count:
        for noteIdx in noteIdxs:
            bar, pos = FrameToBarPos(noteIdx, fps, bpm, et, beatPerBar, beatLen)
            if len(pickIdxs) < count and random.random() > 0.5:
                pickIdxs.append(noteIdx)
            else:
                notPickIdxs.append(noteIdx)
            
        noteIdxs = notPickIdxs
        notPickIdxs = []

    pickIdxs = np.sort(pickIdxs)
    return pickIdxs

def NextBeatBarPos(bar, pos, beatPerBar, beatLen):
    nextPos = (pos // beatLen) * beatLen + beatLen
    if nextPos >= beatPerBar * beatLen:
        nextPos = 0
        bar += 1
    return bar, nextPos

def AppendNoteIfNeed(noteIdxs, fps, bpm, et, beatPerBar, beatLen):
    if len(noteIdxs) <= 0:
        return noteIdxs

    interval = beatLen * 3
    tempNoteIdxs = []
    tempNoteIdxs.append(noteIdxs[0])
    for idx in range(1, len(noteIdxs)):
        bar, pos = FrameToBarPos(noteIdxs[idx], fps, bpm, et, beatPerBar, beatLen)
        allpos = BarPosToAllPos(bar, pos, beatPerBar, beatLen)
        while True:
            lastBar, lastPos = FrameToBarPos(tempNoteIdxs[-1], fps, bpm, et, beatPerBar, beatLen)
            lastAllpos = BarPosToAllPos(lastBar, lastPos, beatPerBar, beatLen)
            if allpos - lastAllpos > interval:
                nextBar, nextPos = NextBeatBarPos(lastBar, lastPos, beatPerBar, beatLen)
                appendNoteIdx =  BarPosToFrame(nextBar, nextPos, fps, bpm, et, beatPerBar, beatLen)
                tempNoteIdxs.append(appendNoteIdx)
            else:
                break

        tempNoteIdxs.append(noteIdxs[idx])
    return tempNoteIdxs

def PickNote(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen):
    noteIdxs = PickNoteWithRule(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen)
    if len(noteIdxs) > count:
        noteIdxs = PickNoteRandom(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen)
        noteIdxs = PickNoteWithRule(noteIdxs, count, fps, bpm, et, beatPerBar, beatLen)

    return noteIdxs

def GenerateNote(songFilePath, duration, bpm, et, seg0, seg1, modelFilePath):
    barDuration = 60 / bpm * 4
    beatInterval = 60 / bpm
    barNoteCountArr = [16] * 9
    playSegCount = 3
    fps = 100
    onsetProcessor = madmom.features.onsets.CNNOnsetProcessor()
    onsetActivation = onsetProcessor(songFilePath)

    segArr = SegmentTimesToFrameIdx(duration, bpm, et, seg0, seg1, playSegCount, fps, len(onsetActivation))
    idx = 0
    onsetFrameIdxs = []
    segTimeArr = []
    showTimeFrameIdx = []
    for playSegArr, showTimeSeg in segArr:
        for frameBegin, frameEnd in playSegArr:
            timeBegin = frameBegin / fps
            timeEnd = frameEnd / fps
            segCount = int((timeEnd - timeBegin) / barDuration * barNoteCountArr[idx])
            segOnsetFrameIdx = PickOnset(onsetActivation, bpm, et, fps, timeBegin, timeEnd, segCount)
            idx += 1
            onsetFrameIdxs.append(segOnsetFrameIdx)
            if len(segTimeArr) == 0 or segTimeArr[-1] != timeBegin:
                segTimeArr.append(timeBegin)
            segTimeArr.append(timeEnd)

        showTimeBegin = showTimeSeg[0] / fps
        beatOffset = int((showTimeBegin - et) / beatInterval)
        beatTimeA = et + beatOffset * beatInterval
        beatTimeB = beatTimeA + beatInterval
        if abs(beatTimeA - showTimeBegin) <= abs(beatTimeB - showTimeBegin):
            showTimeFirstIdx = SecondToFrameIdx(beatTimeA, fps)
        else:
            showTimeFirstIdx = SecondToFrameIdx(beatTimeB, fps)
        
        onsetFrameIdx = onsetFrameIdxs[-1]
        while len(onsetFrameIdx) > 0 and showTimeFirstIdx <= onsetFrameIdx[-1]:
            onsetFrameIdx = onsetFrameIdx[:-1]

        onsetFrameIdx = np.append(onsetFrameIdx, showTimeFirstIdx)
        onsetFrameIdxs[-1] = onsetFrameIdx
        showTimeFrameIdx.append(showTimeFirstIdx)

    beatPerBar = 4
    beatLen = 8
    noteCountScaleArr = [1, 1.2, 1.25, 1.3, 1.5, 1.3, 1.3, 1, 1]
    pickFrameIdx = []
    tempFrameIdx = []
    baseCount = len(onsetFrameIdxs[4]) * 1.0
    for idx in range(len(onsetFrameIdxs)):
        frameIdxArr = onsetFrameIdxs[idx]
        res = PickNote(frameIdxArr, int(baseCount * (noteCountScaleArr[idx] / noteCountScaleArr[4])), fps, bpm, et, beatPerBar, beatLen)
        tempFrameIdx = np.concatenate((tempFrameIdx, res))
        if idx % playSegCount == playSegCount - 1:
            segIdx = idx // playSegCount
            if showTimeFrameIdx[segIdx] > tempFrameIdx[-1]:
                tempFrameIdx = np.append(tempFrameIdx, showTimeFrameIdx[segIdx])

            tempFrameIdx = AppendNoteIfNeed(tempFrameIdx, fps, bpm, et, beatPerBar, beatLen)
            pickFrameIdx = np.concatenate((pickFrameIdx, tempFrameIdx))
            tempFrameIdx = []

    short = np.zeros_like(onsetActivation)
    pickFrameIdx = np.array(pickFrameIdx).astype(int)
    short[pickFrameIdx] = 1
    short[showTimeFrameIdx] = 1

    notes = TransToNote(short, np.zeros_like(short))

    posNotes = FrameIdxToBeatPos(short, fps, bpm, et, beatPerBar, beatLen)
    seg0 = FrameToBarPos(segArr[0][1][1], fps, bpm, et, beatPerBar, beatLen)
    seg1 = FrameToBarPos(segArr[1][1][1], fps, bpm, et, beatPerBar, beatLen)
    notePath = filename.split('.')[0] + '_note.xml'
    songName = os.path.basename(filename).split('.')[0] + '.ogg'
    SaveNotes(notePath, songName, posNotes, seg0, seg1, bpm, et, beatPerBar, beatLen)

    debugInfo = True
    if debugInfo:
        SaveInstantValue(onsetActivation, songFilePath, '_onset_activation')
        SaveInstantValue(pickFrameIdx / fps, songFilePath, '_tango_pick')
        SaveInstantValue(segTimeArr, songFilePath, '_tango_seg_time')
        import LevelInfo
        levelNotes = []
        posOffset = (beatInterval / 8 / 2) * 1000
        for start, noteType, end in notes:
            levelNoteType = LevelInfo.shortNote
            if noteType != 0:
                levelNoteType = LevelInfo.longNote
            levelNotes.append((start / fps * 1000 + posOffset, levelNoteType, end / fps * 1000 + posOffset, 0))
        name = os.path.basename(songFilePath).split('.')[0]
        levelEditorRoot = 'E:/work/dl/audio/proj/LevelEditorForPlayer_8.0/LevelEditor_ForPlayer_8.0/'    
        levelFile = '%sclient/Assets/LevelDesign/%s.xml' % (levelEditorRoot, name)
        LevelInfo.GenerateIdolLevelForTangoDebug(levelFile, levelNotes, bpm, int(et * 1000), int(duration * 1000))

    return notes

def TestSegm():
    y, sr = librosa.load(filename, mono=True, sr=44100)
    logger.info('loaded')
    t = time.time()

    bpm, et = CalcDownbeat(y, sr)    
    duration = librosa.get_duration(y=y, sr=sr)
    beatInter = 60.0 / bpm
    barInterval = beatInter * 4
    numBeats = int((duration-et) / beatInter)
    beatTimes = np.arange(numBeats) * beatInter + et

    numDownbeats = int((duration-et) / barInterval)
    downBeatTimes = np.arange(numDownbeats) * barInterval + et
    SaveInstantValue(downBeatTimes, filename, '_downbeat')
    # return

    beatFrames = librosa.time_to_frames(beatTimes, sr=sr)
    frames, _ = calc_segment(y, sr, beatFrames)
    times = librosa.frames_to_time(frames, sr=sr)

    segTimes = PickSegmentation(times, duration)
    # align to downbeat
    segTimes = np.round((segTimes-et)/barInterval) * barInterval + et

    logger.info('seg times', segTimes)

    SaveInstantValue(times, filename, '_seg')
    SaveInstantValue(segTimes, filename, '_seg2')

    logger.info('t0', time.time() - t)


def GenerateLevelFile(filename, musicInfo, logger):
    '''
    生成音乐关卡文件
    parameter:
    filename : 输出关卡文件的完整的路径+文件名
    musicInfo： 音乐文件信息， 是一个字典，里面记录具体特征数据，包括
        diffculty
        duration
        bpm
        EnterTime
        Seg0
        Seg1
        时间相关的数据单位都为秒

    logger: 用于输出日志，可以输出一些debug信息,如果发生会导致生成关卡文件失败的错误
    则必须调用logger.error接口输出错误码和错误信息，以供服务进程分析错误原因，并返回
    合适的信息
    '''
    pass
    print('call GenerateLevelFile')

def Run(filename = None):
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0201.ogg'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0858.ogg'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1196.ogg'
    # filename = r'd:\librosa\炫舞自动关卡生成\郭德纲 - 做推车.mp3'
    # filename = r'f:\music\有声小说 - 书读至乐，物观化境.mp3'
    # filename = r'f:\music\K One - 爱情蜜语.mp3'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1229.ogg'
    # filename = r'f:\music\英语听力 - 大卫·科波菲尔04.mp3'
    # filename = r'f:\music\侯宝林,郭启儒 - 抬杠.mp3'
    # filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'
    # filename = r'f:\music\song_1698.ogg'

    # if (len(sys.argv)) < 2:
    #     print('missing music file(ogg) name as input!')
    #     exit(1)

    if len(sys.argv) > 1:
        filename = sys.argv[-1]

    print('argv', sys.argv)

    outdir = os.path.dirname(filename)
    levelFile = os.path.splitext(filename)[0] + '.xml'

    globals()['logger'] = QQDancerLog.Logger(outdir)

    args = {}
    args['-thread'] = 1
    args['-duration'] = 60     #用来做分析的音乐长度
    args['-threhold'] = 0.02    #判断节拍不准的阈值
    args['-abThrehold'] = 0.333 #判断为不是音乐的比例阈值
    args['-diffculty'] = 6      #难度


    for i in range(len(sys.argv)):
        if sys.argv[i] in args:
            name = sys.argv[i]
            valType = type(args[name])
            args[name] = valType(sys.argv[i+1])

    logger.info('args', args)
    
    success, info = AnalysisMusicFeature(filename, **args)

    if success:
        info['diffculty'] = args['-diffculty']
        LevelMaker.GenerateLevelFile(levelFile, info, logger)
        #GenerateLevelFile(levelFile, info, logger)
    
    logger.info('result', info)


def do_work(filelist, dummy):
    print('filelist', filelist)
    for file in filelist:
        if not os.path.exists(file):
            print(file, 'not exists!')
        else:
            Run(file)



def BatchTest():
    import multiprocessing as mp
    numWorker = 6

    idlist = [1, 4, 6, 16, 22, 130, 168, 378, 520, 1392, 25, 34, 60, 83, 98, 118, 134, 156, 177, 204, 212, 236, 254, 310, 323, 61, 487, 823, 1005, 1044, 1462, 1549, 217, 1688, 1842, 956, 975, 152, 514, 150]
    idlist2 = [1, 4, 6, 16, 34, 60, 61, 83, 98, 130, 134, 150, 152, 156, 177, 178, 204, 212, 236, 254, 310, 323, 378, 514, 520, 956, 1005, 1044, 1392, 1462, 1489, 1549, 1688, 1842]

    a = [i for i in idlist if i not in idlist2]

    idlist = [184, 336, 461, 515, 784, 937, 1080, 1233, 1303, 1422]

    path = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_%04d.ogg'

    if len(idlist) > 0:
        filelist = [path % i for i in idlist]
    
    lists = [filelist[i::numWorker] for i in range(numWorker)]
    startTime = time.time()
    processes = []

    for i in range(numWorker):
        if len(lists[i]) > 0:
            print('thread ', i)
            t = mp.Process(target=do_work, args=(lists[i], 1))
            processes.append(t)
            t.start()
    
    for t in processes:
        t.join()    

    print()
    print('done in %.1f minute' % ((time.time() - startTime) / 60.0))

if __name__ == '__main__':
    
    ext = '.m4a'
    songs = []
    songs.append('Emmanuel - Corazon de Melao')
    songs.append('Kaoma - Chacha la Vie')
    songs.append('Livan Nunez - Represent, Cuba')
    songs.append('Marc Anthony - Dímelo')
    songs.append('Marc Anthony - I Need to Know')
    songs.append('Michael Bublé - Sway')
    songs.append('Michael Learns To Rock - Blue Night')
    songs.append('Nana Mouskouri - Rayito de Luna')
    songs.append('Santa Esmeralda - I Heart It Through The Grapevine／Latin Vers')
    songs.append('Andy Fortuna Productions - Amor')
    for song in songs:
        filename = r'E:\work\dl\audio\proj\rm\%s\%s%s' % (song, song, ext)
        Run(filename)
    # BatchTest()


    

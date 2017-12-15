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

# 错误 code
MUSIC_FILE_ERROR = 1    #打开音乐文件错误
MUSIC_TOO_LONG = 2      #音乐文件太长
MUSIC_TOO_SHORT = 3     #音乐文件太短
PROBABLY_NOT_MUSIC = 4  #音乐文件的内容可能不是歌曲


logger = None


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


def normalizeInterval(beat, threhold = 0.02, abThrehold=0.333):
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


def CalcDownbeat(y, sr, **args):
    # calc downbeat entertime
    analysisLength = args['-duration']
    minimumMusicLength = 165
    maximumMusicLength = 360
    numThread = args['-thread']
    threhold = args['-threhold']
    abThrehold = args['-abThrehold']

    startTime = time.time()
    
    duration = librosa.get_duration(y=y, sr=sr)

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
    
    firstBeat, lastBeat = normalizeInterval(beatIndex, threhold=threhold, abThrehold=abThrehold)
    if firstBeat == -1:        
        logger.error(PROBABLY_NOT_MUSIC, 'generate error,numbeat %d, abnormal rate %f' % (len(beatIndex), lastBeat))
        return 0, 0

    newBeat = beatIndex[firstBeat:lastBeat]
    downbeat = madmom.features.downbeats.filter_downbeats(newBeat)
    downbeat = downbeat + start

    barInter, etAuto = CalcBarInterval(downbeat)

    # first 10 seconds samples to find first beat
    yy = y[:441000]
    act = processer(yy)
    beginBeat = downbeatTracking(act)                    
    firstBeatTime = beginBeat[0, 0]

    if abs(etAuto - firstBeatTime) > barInter:
        etAuto += ((firstBeatTime - etAuto) // barInter + 1) * barInter

    assert abs(etAuto - firstBeatTime) < barInter
    
    bpm = 240.0 / barInter

    return bpm, etAuto
    

def CalcSegmentation(y, sr, beats, k = 4):
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


def PickSegmentation(segments, duration, neighbourhood=0.167):
    # 在三分之一和三分之二附近挑选两个分段点
    points = segments / duration
    def f(a):
        p = np.argmin(np.abs(points - a))
        p = points[p]
        if abs(p-a) > neighbourhood:
            p = a
        return p * duration

    result = np.array([f(0.3333), f(0.6666)])
    return result


def SaveInstantValue(beats, filename, postfix = ''):
    import os
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True


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

    bpm, et = CalcDownbeat(y, sr, **args)
    if bpm == 0:
        return (False, None)
        
    duration = librosa.get_duration(y=y, sr=sr)
    beatInter = 60.0 / bpm
    barInterval = beatInter * 4
    numBeats = int((duration-et) / beatInter)
    beatTimes = np.arange(numBeats) * beatInter + et
    beatFrames = librosa.time_to_frames(beatTimes, sr=sr)

    logger.info('analysis segmentation', time.time() - t)
    frames, _ = CalcSegmentation(y, sr, beatFrames)
    times = librosa.frames_to_time(frames, sr=sr)
    segTimes = PickSegmentation(times, duration)
    # align to downbeat
    segTimes = np.round((segTimes-et)/barInterval) * barInterval + et

    logger.info('AnalysisMusicFeature done', time.time() - t)
    result = {}
    result['duration'] = duration
    result['bpm'] = bpm
    result['EnterTime'] = et
    result['seg0'] = segTimes[0]
    result['seg1'] = segTimes[1]
    return (True, result)


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

def Test():
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0201.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0858.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1196.ogg'
    filename = r'd:\librosa\炫舞自动关卡生成\郭德纲 - 做推车.mp3'
    filename = r'f:\music\有声小说 - 书读至乐，物观化境.mp3'
    filename = r'f:\music\K One - 爱情蜜语.mp3'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1229.ogg'
    filename = r'f:\music\英语听力 - 大卫·科波菲尔04.mp3'
    filename = r'f:\music\侯宝林,郭启儒 - 抬杠.mp3'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'

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


if __name__ == '__main__':
    
    # logger.info(vars())

    Test()
    

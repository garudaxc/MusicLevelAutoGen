# -*- coding: utf-8 -*-

import madmom
from madmom.features.downbeats import *
import librosa
import numpy as np
import time
import sklearn.cluster
import scipy


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


def normalizeInterval(beat, threhold = 0.02):
    # 截掉前后不太准的beat
    interval = beat[1:,0] - beat[:-1, 0]

    aver = np.average(interval)
    diff = np.abs(interval - aver)

    abnormal = np.nonzero(diff > threhold)[0]
    print('abnormal count', abnormal.size, abnormal)

    if abnormal.size > beat.shape[0] / 3:
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
    # print('del %d items form begining %d items form end' % (count0, count1))

    return 0, len(beat)


def CalcDownbeat(y=None, sr=None):
    # calc downbeat entertime
    analysisLength = 60
    minimumMusicLength = 165
    maximumMusicLength = 360
    numThread = 1


    print('begin')
    startTime = time.time()
    
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < minimumMusicLength:
        print('music is too short! duration:', duration)
        return

    if duration > maximumMusicLength:
        print('music is too long! duration:', duration)
        return

    start = int(duration * 0.3)
    clipTime = np.array([start, start+analysisLength])      
    print('duration', duration, 'start', start)
    clip = librosa.time_to_samples(clipTime, sr=sr)
    print('total', y.shape, 'clip', clip)
    yy = y[clip[0]:clip[1]]

    print('time in %.1f' % ((time.time() - startTime)))
    processer = RNNDownBeatProcessor(num_threads=numThread)
    downbeatTracking = DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda=1000, fps=100)
    beatProba = processer(yy)
    print('time in %.1f' % ((time.time() - startTime)))

    beatIndex = downbeatTracking(beatProba)
    
    firstBeat, lastBeat = normalizeInterval(beatIndex)
    if firstBeat == -1:        
        print('generate error,numbeat %d, abnormal rate %f' % (len(beatIndex), lastBeat))
        return

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
    print('bpm', bpm)

    return bpm, etAuto
    





def calc_segment(y, sr, beats, k = 5):
    # Next, we'll compute and plot a log-power CQT
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                ref=np.max)    
 
    # To reduce dimensionality, we'll beat-synchronous the CQT
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    # Csync = librosa.util.sync(C, beats, aggregate=np.median)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # For plotting purposes, we'll need the timing of the beats
    # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    # beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
    #                                                             x_min=0,
    #                                                             x_max=C.shape[1]),
    #                                     sr=sr)

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

    # sphinx_gallery_thumbnail_number = 5

    # from os import listdir
    # import os.path
    # def save_file(beats, mp3filename, postfix = ''):
    #     outname = os.path.splitext(mp3filename)[0]
    #     outname = outname + postfix + '.csv'
    #     librosa.output.times_csv(outname, beats)
    #     logger.info('output beat time file ' + outname)

    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    # freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
    #                                 fmin=librosa.note_to_hz('C1'),
    #                                 bins_per_octave=BINS_PER_OCTAVE)

    # librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
    #                         bins_per_octave=BINS_PER_OCTAVE,
    #                         x_axis='time')
    # ax = plt.gca()

    # for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
    #     ax.add_patch(patches.Rectangle((interval[0], freqs[0]),
    #                                 interval[1] - interval[0],
    #                                 freqs[-1],
    #                                 facecolor=colors(label),
    #                                 alpha=0.50))

    # plt.tight_layout()
    return my_bound_frames, bound_segs



def CalcSegmentProbability(segments, duration, neighbourhood=0.167):
    # 计算每个候选分段点的概率，三分之一和三分之二处的概率最高
    points = segments / duration

    def f(x):
        if x > 0.5:
            x = 1 - x
        a = 1 / neighbourhood
        y0 = a * (x-(0.3333-neighbourhood))
        y1 = -a * (x-(0.3333+neighbourhood))
        y = min(max(0.0, y0), max(0.0, y1))
        return y

    y = [f(a) for a in points]
    return np.array(y)


import os
def SaveInstantValue(beats, filename, postfix = ''):
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True


def TestSegm():
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    y, sr = librosa.load(filename, mono=True, sr=44100)
    print('loaded')
    t = time.time()

    bpm, et = CalcDownbeat(y, sr)    
    duration = librosa.get_duration(y=y, sr=sr)
    downbeatInter = 60.0 / bpm
    numDownBeats = int((duration-et) / downbeatInter)
    downbeatTimes = np.arange(numDownBeats) * downbeatInter + et
    # SaveInstantValue(downbeatTimes, filename, '_down')
    # return

    downbeatFrames = librosa.time_to_frames(downbeatTimes, sr=sr)

    frames, segs = calc_segment(y, sr, downbeatFrames)

    times = librosa.frames_to_time(frames, sr=sr)
    SaveInstantValue(times, filename, '_seg')

    print('t0', time.time() - t)




if __name__ == '__main__':
    
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    filename = r'd:\librosa\炫舞自动关卡生成\郭德纲 - 做推车.mp3'
    filename = r'f:\music\英语听力 - 大卫·科波菲尔04.mp3'
    filename = r'f:\music\有声小说 - 书读至乐，物观化境.mp3'
    filename = r'f:\music\侯宝林,郭启儒 - 抬杠.mp3'
    filename = r'f:\music\K One - 爱情蜜语.mp3'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_0178.ogg'

    
    # TestSegm()

    CalcSegmentProbability(None, None)


    # CalcDownbeat(filename)

    # import madmom.ml.nn.layers
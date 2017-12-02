# -*- coding: utf-8 -*-

import madmom
from madmom.features.downbeats import *
import librosa
import numpy as np
import time


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

    #计算头，尾两处相对准确的拍子，进一步计算更准确的拍子间隔
    i1, i2 = diff(itvals, a)
    a, b = calc_beat_interval(beat_times, i1, i2)
    # 将b补偿到最近的正数位置
    compensate = int(min(0, b//a))
    b -= compensate * a
    
    return a, b



def normalizeInterval(beat, threhold = 0.03):
    # 截掉前后不太准的beat
    interval = beat[1:,0] - beat[:-1, 0]

    aver = np.average(interval)
    diff = np.abs(interval - aver)

    abnormal = np.nonzero(diff > threhold)[0]
    print('abnormal count', abnormal.size, abnormal)

    if abnormal.size > beat.shape[0] / 3:
        # 拍子均匀程度太低，自动生成失败
        return -1, abnormal.size / beat.shape[0]

    #删除头尾不均匀的拍子
    #从前往后删
    count0 = 0
    for i in range(len(abnormal)):
        if i == abnormal[i]:
            count0 = count0 + 1

    #从后往前删
    count1 = 0
    for i in range(-1, -1-len(abnormal), -1):
        if len(interval) + i == abnormal[i]:
            count1 = count1 + 1
    print('del %d items form begining %d items form end' % (count0, count1))

    return count0, len(beat) - count1


def CalcDownbeat(filename):
    # calc downbeat

    print('begin')
    startTime = time.time()
    y, sr = librosa.load(filename, mono=True, offset=20, duration=100, sr=44100) 
    print('time in %.1f' % ((time.time() - startTime)))

    beatProba = RNNDownBeatProcessor(num_threads=4)(y)
    print('time in %.1f' % ((time.time() - startTime)))

    beatIndex = DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda=1000, fps=100)(beatProba)
    
    firstBeat, lastBeat = normalizeInterval(beatIndex)
    
    print('time in %.1f' % ((time.time() - startTime)))

    if firstBeat == -1:        
        print('%s generate error, abnormal rate %f' % (id, lastBeat))
        print('num beat', len(beatIndex), beatIndex[:20])
        return

    firstBeatTime = beatIndex[0, 0]
    newBeat = beatIndex[firstBeat:lastBeat]
    downbeat = madmom.features.downbeats.filter_downbeats(newBeat)

    newBeat = newBeat[:,0]
    barInter, et = CalcBarInterval(downbeat)

    #et = downbeat[0]
    print('first down beat', et)
    while (et - barInter) > firstBeatTime:
        et = (et - barInter)
    
    bpm = 240.0 / barInter
    print('bpm', bpm)
    



if __name__ == '__main__':
    
    filename = r'd:\librosa\炫舞自动关卡生成\郭德纲 - 做推车.mp3'
    filename = r'f:\music\侯宝林,郭启儒 - 抬杠.mp3'
    filename = r'f:\music\有声小说 - 书读至乐，物观化境.mp3'
    filename = r'f:\music\K One - 爱情蜜语.mp3'
    filename = r'f:\music\英语听力 - 大卫·科波菲尔04.mp3'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    CalcDownbeat(filename)

    # import madmom.ml.nn.layers
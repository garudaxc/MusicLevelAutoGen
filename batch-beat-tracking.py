from os import listdir
import os.path
import librosa
import matplotlib.pyplot as plt
import numpy as np


tempos = []

#path = '/Users/xuchao/Music/网易云音乐/'
path = 'd:/librosa/炫舞自动关卡生成/测试歌曲2/'
filenames = ['薛之谦 - 我好像在哪见过你.mp3',
             '薛之谦 - 演员.mp3',
             '赵雷 - 成都.mp3',
             'Alan Walker - Fade.mp3',
             'Kelly Clarkson - Stronger.mp3',
             '李健 - 风吹麦浪.mp3',
             '李健 - 绽放.mp3',
             '王麟 - 伤不起.mp3',
             'G.E.M.邓紫棋 - 后会无期.mp3',
             'G.E.M.邓紫棋 - 龙卷风.mp3']


def load_data(path):
    y, sr = librosa.load(path, sr = None)
    print('load ' + path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempos.append([path, tempo])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times

def MSL(beats):
    numBeats = len(beats)
    print(len(beats))

    AT = np.matrix([np.ones(numBeats), range(numBeats)])
    b = np.matrix(beats)
    x = (AT * AT.T).I * AT * b.T
    #print(x)

    a = x[1, 0]
    b = x[0, 0]

    beat_times2 = np.array(range(numBeats))
    beat_times2 = beat_times2 * a

    return a, b

def beat_intervals(beats):
    numbeats = len(beats)
    t = np.zeros(numbeats-1)
    for i in range(numbeats-1):
        t[i] = beats[i+1] - beats[i]
    plt.plot(t)
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
    plt.plot(mean, 'g')
    plt.plot(dist, 'r')

    cnt = int(numBeat / 3)
    print('count ', cnt)
    m, i1 = min_measure(mean[:cnt], dist[:cnt])
    print(m, ' ', i1)
    #plt.plot(i, m, 'o')
    
    m, i2 = min_measure(mean[-cnt:], dist[-cnt:])
    print(m, ' ', i2)
    #plt.plot(numBeat - cnt + i, m, '^')
    i2 = numBeat - cnt + i2

    return i1, i2
        
        
def calc_beat_interval(beats, i1, i2):
    size = 32
    bpm1 = (beats[i1 + size] - beats[i1]) / float(size)
    bpm2 = (beats[i2 + size] - beats[i2]) / float(size)
    bpm = (bpm1 + bpm2) / 2.0
    print('bpm', bpm1, bpm2, bpm)
    numbeat = (beats[i2] - beats[i1]) / bpm
    print('beat count', numbeat)
    a2 = (beats[i2] - beats[i1]) / round(numbeat)
    print('a2', a2)
    b1 = beats[i1] - a2 * i1
    b2 = beats[i2] - a2 * i2
    print('b', b1, b2)
    return a2, b1
    
def save_file(beats, mp3filename, postfix = ''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    librosa.output.times_csv(outname, beats)
    print('output beat time file ' + outname)

def do_file(pathname):
    beat_times = load_data(pathname)
    #save_file(beat_times, pathname, '')
    numBeats = len(beat_times)
    itvals = beat_intervals(beat_times)
    #print('mean ' + str(mean))

    a, b = MSL(beat_times)
    beat_times_msl = np.array(range(numBeats)) * a + b
    #save_file(beat_times_msl, pathname, '_msl')
    print('a b', a, b)
    #plt.plot(np.ones(numBeats) * a, 'r')

    i1, i2 = diff(itvals, a)
    a, b = calc_beat_interval(beat_times, i1, i2)
    print('a b', a, b)

    new_beat_times = np.array(range(numBeats)) * a + b

    save_file(new_beat_times, pathname, '')

##filenames = filenames[-2:-1]
##for f in filenames:
##    pathname = path + f
##    do_file(pathname)


files = [path + f for f in listdir(path) if os.path.splitext(f)[1] == '.mp3']

def dummy(f):
    do_file(f)
    print(f)

list(map(dummy, files))

#plt.show()


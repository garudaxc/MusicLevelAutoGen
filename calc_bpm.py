from os import listdir
import os.path
import librosa
import numpy as np
import sys
import io
import logger

hop_lenth = 512

def calc_power(y, sr):
    # 计算音乐强度
    S = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(S)
    db = (db - np.min(db)) / (np.max(db) - np.min(db))
    l = np.sum(db * db, 0)**0.5
    return l

def max_power_beat(frame_power, beat_frames):    
    frame_strength = frame_power[beat_frames]
    frame_strength = np.resize(frame_strength, (beat_frames.size//4, 4)).T
    frame_strength = np.sum(frame_strength, 1) / (beat_frames.size // 4)
    logger.info('frame strength', frame_strength, 'average', np.average(frame_power)) 
    
    max_index = np.argmax(frame_strength)
    return max_index

def init_beat(y, sr, onset_env):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_lenth, tightness=50)

    max_index = max_power_beat(onset_env, beat_frames)
    logger.info('max index', max_index) 

    return beat_frames, max_index

def MSL(beats):
    numBeats = len(beats)
    logger.info(len(beats))

    AT = np.matrix([np.ones(numBeats), range(numBeats)])
    b = np.matrix(beats)
    x = (AT * AT.T).I * AT * b.T
    #logger.info(x)

    a = x[1, 0]
    b = x[0, 0]

    beat_times2 = np.arange(numBeats)
    beat_times2 = beat_times2 * a

    return a, b

def beat_intervals(beats):
    numbeats = len(beats)
    t = np.zeros(numbeats-1)
    for i in range(numbeats-1):
        t[i] = beats[i+1] - beats[i]
        
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
    # numBeats += compensate
    # logger.info('a b ', a, b)

    # new_beat_times = np.arange(numBeats) * a + b
    return a, b



def CalcBarInterval2(beat_times):
    numBeats = len(beat_times)
    itvals = beat_intervals(beat_times)
    #logger.info('mean ' + str(mean))

    #最小二乘计算固定间隔拍子·
    a, b = MSL(beat_times)  
    compensate = int(min(0, b//a))
    b -= compensate * a

    # new_beat_times = np.arange(numBeats) * a + b
    return a, b



def doWork():
    logger.init('calc_bpm.log', to_console=False)
    
    if len(sys.argv) < 2:
        logger.error('no music file input!')
        print('usage : calc_bpm.exe music file (mp3 or m4a)')
        sys.exit(1)
    
    pathname = sys.argv[1]
    ext = os.path.splitext(pathname)[1]
    if ext != '.mp3' and ext != '.m4a':
        logger.error('input file is not mp3 or m4a')
        print('music file must be mp3 or m4a')
        sys.exit(1)

    y, sr = librosa.load(pathname, sr = None)
    logger.info('load ' + pathname)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    num_frames = onset_env.shape[0]
    logger.info('frame count ', num_frames)
    music_length = librosa.frames_to_time(num_frames, sr=sr)

    beat_frames, max_index = init_beat(y, sr, onset_env)    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

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
    numBeats += compensate
    logger.info('a b ', a, b)

    new_beat_times = np.arange(numBeats) * a + b

    bpm = 60.0 / a
    logger.info('bpm ', bpm)
    print('bpm', bpm)

    # 挑出重拍，只支持4/4拍
    bar_times = new_beat_times[max_index:new_beat_times.size:4]

    print('et', bar_times[0])

    sys.exit(0)


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

    # calc_bpm()

    doWork()
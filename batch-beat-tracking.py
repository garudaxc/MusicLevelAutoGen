from os import listdir
import os.path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import csv
import plot_segmentation as seg


#path = '/Users/xuchao/Music/网易云音乐/'
# path = 'd:/librosa/炫舞自动关卡生成/测试歌曲2/'
# filenames = ['薛之谦 - 我好像在哪见过你.mp3',
#              '薛之谦 - 演员.mp3',
#              '赵雷 - 成都.mp3',
#              'Alan Walker - Fade.mp3',
#              'Kelly Clarkson - Stronger.mp3',
#              '李健 - 风吹麦浪.mp3',
#              '李健 - 绽放.mp3',
#              '王麟 - 伤不起.mp3',
#              'G.E.M.邓紫棋 - 后会无期.mp3',
#              'G.E.M.邓紫棋 - 龙卷风.mp3']

file = '/Users/xuchao/Music/网易云音乐/BEYOND - 真的爱你.mp3'
file = '/Users/xuchao/Music/网易云音乐/王麟 - 伤不起.mp3'
file = '/Users/xuchao/Music/网易云音乐/Good Time.mp3'
file = '/Users/xuchao/Music/网易云音乐/Maroon 5-Maps.mp3'
file = '/Users/xuchao/Music/网易云音乐/Maroon 5 - Sugar.mp3'
file = '/Users/xuchao/Music/网易云音乐/极乐净土.mp3'

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
    print('frame strength', frame_strength, 'average', np.average(frame_power)) 
    
    max_index = np.argmax(frame_strength)
    return max_index

def init_beat(y, sr, onset_env):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_lenth, tightness=50)

    max_index = max_power_beat(onset_env, beat_frames)
    print('max index', max_index) 

    # plt.plot(onset_env)
    # plt.vlines(beat_frames, 0, np.max(onset_env))
    # plt.show()

    return beat_frames, max_index

def MSL(beats):
    numBeats = len(beats)
    print(len(beats))

    AT = np.matrix([np.ones(numBeats), range(numBeats)])
    b = np.matrix(beats)
    x = (AT * AT.T).I * AT * b.T
    #print(x)

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
    # plt.plot(mean, 'g')
    # plt.plot(dist, 'r')

    cnt = int(numBeat / 3)
    
    m, i1 = min_measure(mean[:cnt], dist[:cnt])
    print(m, ' ', i1)
    #plt.plot(i, m, 'o')
    
    m, i2 = min_measure(mean[-cnt:], dist[-cnt:])
    
    #plt.plot(numBeat - cnt + i, m, '^')
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
    
def save_file(beats, mp3filename, postfix = ''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    librosa.output.times_csv(outname, beats)
    print('output beat time file ' + outname)

def do_file(pathname):    
    y, sr = librosa.load(pathname, sr = None)
    print('load ' + pathname)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    print('num onset_env', onset_env.size)

    beat_frames, max_index = init_beat(y, sr, onset_env)    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    numBeats = len(beat_times)
    itvals = beat_intervals(beat_times)
    #print('mean ' + str(mean))

    #最小二乘计算固定间隔拍子·
    a, b = MSL(beat_times)    
    print('a b', a, b)

    #计算头，尾两处相对准确的拍子，进一步计算更准确的拍子间隔
    i1, i2 = diff(itvals, a)
    a, b = calc_beat_interval(beat_times, i1, i2)
    print('a b', a, b)

    new_beat_times = np.arange(numBeats) * a + b
    bpm = 60.0 / a
    print('bpm ', bpm)

    # 挑出重拍，只支持4/4拍
    bar_times = new_beat_times[max_index:new_beat_times.size:4]
    bar_frames = librosa.time_to_frames(bar_times, sr=sr, hop_length=512)

    bound_frames, bound_segs = seg.calc_segment(y, sr)
    seg_power, power_data = seg.calc_power(y, sr, bound_frames, bound_segs)

    num_frames = onset_env.shape[0]
    bar_value = np.zeros(bar_frames.shape[0])
    for i in range(bar_frames.shape[0]):
        index = np.flatnonzero(bound_frames < bar_frames[i])
        if len(index) == 0:
            bar_value[i] = 0
        else:
            index = index[-1]
            if index == len(seg_power):
                bar_value[i] = 0
            bar_value[i] = seg_power[index]
    bar_value = bar_value[:-1] - bar_value[1:]
    bar_value = np.append(bar_value, 0)
    bar_value = np.fmax(bar_value, np.zeros(bar_value.shape))
    bar_value **= 0.5
    bar_value = bar_value * 0.5 + 0.5
    
    plt.plot(bar_frames, bar_value)

    gen_pro = seg.gen_seg_probability(num_frames, bar_frames) 
    bar_value = bar_value * gen_pro
    plt.plot(bar_frames, bar_value)

    big_seg = np.zeros(2)
    bar_index = np.argmax(bar_value)
    bar_value[bar_index] = 0
    big_seg[0] = bar_frames[bar_index]
    bar_index = np.argmax(bar_value)
    big_seg[1] = bar_frames[bar_index]
    big_seg = np.sort(big_seg)
    big_seg_time = librosa.frames_to_time(big_seg, sr=sr, hop_length=512)

    save_file(bar_times, pathname, '_bar')

    save_file(big_seg_time, pathname, '_seg')

    save_time_value_tofile(power_data, pathname, '_pow')

    plt.show()
    
def save_time_value_tofile(data, mp3filename, postfix=''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        list(map(spamwriter.writerow, data))


##filenames = filenames[-2:-1]
##for f in filenames:
##    pathname = path + f····
##    do_file(pathname)


def dummy(f):
    do_file(f)

# files = [path + f for f in listdir(path) if os.path.splitext(f)[1] == '.mp3']
#list(map(dummy, files))
#plt.show()

if __name__ == '__main__':
    dummy(file)
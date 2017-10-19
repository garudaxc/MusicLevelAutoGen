import madmom
import numpy as np
from os import listdir
import os.path
import librosa
import LevelInfo
import logger
import threading
import time

def save_file(beats, mp3filename, postfix = ''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    librosa.output.times_csv(outname, beats)
    logger.info('output beat time file ' + outname)

def list_file(path):
    files = [path + f for f in listdir(path) if os.path.splitext(f)[1] == '.mp3' or os.path.splitext(f)[1] == '.m4a']
    return files



def do_work(*args, **kwargs):
    print(kwargs)
    for i in range(10):
        print(i)
        time.sleep(0.5)


def test():
    
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Lily Allen - Hard Out Here.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Sam Tsui - Make It Up.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\T-ara Falling U.mp3' #不准
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\王心凌 - 灰姑娘的眼泪 (电音版).mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\夏天Alex - 不再联系.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\CNBLUE- LOVE.mp3' #不准
    filename = r'D:\librosa\炫舞自动关卡生成\music\100017.mp3'

    levelInfo = LevelInfo.load_levelinfo_file('D:/librosa/炫舞自动关卡生成/level-infos.xml')

    id = os.path.basename(filename).split('.')[0]
    level = levelInfo[id]
    
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=200)

    act = processer(filename)
    beat = downbeatTracking(act)
    downbeat = madmom.features.downbeats.filter_downbeats(beat)

    beat = beat[:,0]
    intervals = beat[1:] - beat[:-1]
    print(intervals)
    aver = np.average(intervals)
    print('average', aver)

    bpm = 60.0 / aver
    print('bpm', bpm)
    print('bpm in level', level[0])

    variance = (intervals - aver) ** 2
    variance = np.sum(variance)
    print('variance', variance)


    #print(downbeat)
    # index = int(5 - beat[0, 1]) % 4
    # print('index', index)
    # downbeat = beat[:,0][index::4]
    # print(downbeat)

    # save_file(downbeat, filename, '_downbeat')



if __name__ == '__main__':
    t0 = threading.Thread(target=do_work, kwargs={'aaa':'bbb'})
    t1 = threading.Thread(target=do_work)

    t0.start()
    t1.start()
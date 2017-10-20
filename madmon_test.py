import madmom
import numpy as np
from os import listdir
import os.path
import librosa
import LevelInfo
import logger
import multiprocessing as mp
import time

def save_file(beats, mp3filename, postfix = ''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    librosa.output.times_csv(outname, beats)
    logger.info('output beat time file ' + outname)

def list_file(path):
    files = [os.path.join(path, f) for f in listdir(path) if os.path.splitext(f)[1] == '.mp3' or os.path.splitext(f)[1] == '.m4a']
    return files


def do_work(filelist, levelInfo, processer, downbeatTracking, result):
    print('process ', os.getpid())
    for file in filelist:
        id = os.path.basename(file).split('.')[0]
        levelId = levelInfo[id]

        print('id', id)
        act = processer(file)
        beat = downbeatTracking(act)
        #downbeat = madmom.features.downbeats.filter_downbeats(beat)

        beat = beat[:,0]
        intervals = beat[1:] - beat[:-1]
        aver = np.average(intervals)

        #print('average', aver)

        bpm = 60.0 / aver
        print(id, 'bpm', bpm)
        # print('bpm in level', level[0])

        variance = (intervals - aver) ** 2
        variance = np.sum(variance)
        #print('variance', variance)

        r = {'id':id, 'bpm':bpm, 'bil':levelId[0], 'variance':variance}
        result.put(r)


def test():
    
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Lily Allen - Hard Out Here.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Sam Tsui - Make It Up.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\T-ara Falling U.mp3' #不准
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\王心凌 - 灰姑娘的眼泪 (电音版).mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\夏天Alex - 不再联系.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\CNBLUE- LOVE.mp3' #不准
    filename = r'd:\librosa\炫舞自动关卡生成\庄心妍 - 繁星点点.mp3'
    filename = r'D:\librosa\炫舞自动关卡生成\music\100003.mp3'

    levelInfo = LevelInfo.load_levelinfo_file('D:/librosa/炫舞自动关卡生成/level-infos.xml')

    id = os.path.basename(filename).split('.')[0]
    level = levelInfo[id]
    enterTime = float(level[1]) / 1000.0
    
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=100)

    act = processer(filename)
    beat = downbeatTracking(act)
    downbeat = madmom.features.downbeats.filter_downbeats(beat)

    #print(downbeat)
    # index = int(5 - beat[0, 1]) % 4
    # print('index', index)
    # downbeat = beat[:,0][index::4]
    # print(downbeat)

    beat = beat[:,0]
    #save_file(beat, filename, '_beat')

    intervals = beat[1:] - beat[:-1]
    #print(intervals)
    aver = np.average(intervals)
    print('average', aver)

    bpm = 60.0 / aver
    print('bpm', bpm)
    print('bpm in level', level[0])

    variance = (intervals - aver) ** 2
    variance = np.sum(variance)
    print('variance', variance)
    
    downbeatInter = aver * 4
    downbeatDiff = abs(downbeat[0] - enterTime) % downbeatInter
    if downbeatDiff > (0.5 * downbeatInter):
        downbeatDiff = downbeatInter - downbeatDiff
    print('downbeatInter', downbeatInter, 'downbeat diff', downbeatDiff, 'enter time', enterTime)

    save_file(downbeat, filename, '_downbeat')

def doMultiProcess(numWorker = 4):
    filelist = list_file(r'D:/librosa/炫舞自动关卡生成/music')
    filelist = filelist[:2]
    
    lists = [filelist[i::numWorker] for i in range(numWorker)]
    queue = mp.Queue()
    
    levelInfo = LevelInfo.load_levelinfo_file('D:/librosa/炫舞自动关卡生成/level-infos.xml')

    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=200)

    processes = []

    for i in range(numWorker):
        t = mp.Process(target=do_work, args=(lists[i], levelInfo, processer, downbeatTracking, queue))
        processes.append(t)
        t.start()

    for i in range(len(filelist)):
        r = queue.get(True)
        print(r)

    for t in processes:
        t.join()    

    print('done')


def calcEnterTime():
    file = r'D:/librosa/炫舞自动关卡生成/music/100002.mp3'



if __name__ == '__main__':
    test()
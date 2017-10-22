import madmom
import numpy as np
from os import listdir
import os.path
import librosa
import LevelInfo
import logger
import multiprocessing as mp
import time
import logger

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
        level = levelInfo[id]
        enterTime = float(level[1]) / 1000.0

        print('id', id)
        act = processer(file)
        beat = downbeatTracking(act)
        downbeat = madmom.features.downbeats.filter_downbeats(beat)

        beat = beat[:,0]    

        beatInter = (beat[-1] - beat[0]) / (len(beat) - 1)
        bpm = 60.0 / beatInter
        # print('bpm in level', level[0])

        downbeatInter = beatInter * 4
        downbeatDiff = abs(downbeat[0] - enterTime) % downbeatInter
        if downbeatDiff > (0.5 * downbeatInter):
            downbeatDiff = downbeatInter - downbeatDiff        

        r = {'id':id, 'bpm':bpm, 'bil':level[0], 'downbeatDiff':downbeatDiff}
        r2 = [id, bpm, level[0], downbeat[0], enterTime, downbeatDiff]
        result.put(r2)


def test():
    
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Lily Allen - Hard Out Here.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Sam Tsui - Make It Up.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\T-ara Falling U.mp3' #不准
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\王心凌 - 灰姑娘的眼泪 (电音版).mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\夏天Alex - 不再联系.mp3'
    filename = r'd:\librosa\炫舞自动关卡生成\测试歌曲\拍子不准\CNBLUE- LOVE.mp3' #不准
    filename = r'd:\librosa\炫舞自动关卡生成\庄心妍 - 繁星点点.mp3'
    filename = r'D:\librosa\炫舞自动关卡生成\music\100003.mp3'

    if os.name == 'posix':
        filename = '/Users/xuchao/Music/网易云音乐/G.E.M.邓紫棋 - 后会无期.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/李健 - 绽放.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/Good Time.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/金志文 - 哭给你听.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/薛之谦 - 我好像在哪见过你.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/萧敬腾 - 王妃.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/小沈阳 - 我的好兄弟.mp3'
        filename = '/Users/xuchao/Music/网易云音乐/极乐净土.mp3'         #enterTime = 2.349
        filename = '/Users/xuchao/Music/网易云音乐/信乐团 - 海阔天空.mp3'

    levelInfo = LevelInfo.load_levelinfo_file('D:/librosa/炫舞自动关卡生成/level-infos.xml')
    if levelInfo != None:
        id = os.path.basename(filename).split('.')[0]
        level = levelInfo[id]
        enterTime = float(level[1]) / 1000.0
    
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=100)

    act = processer(filename)
    beat = downbeatTracking(act)
    downbeat = madmom.features.downbeats.filter_downbeats(beat)

    beat = beat[:,0]
    # save_file(beat, filename, '_beat')

    beatInter = (beat[-1] - beat[0]) / (len(beat) - 1)
    bpm = 60.0 / beatInter
    print('bpm', bpm)

    if levelInfo != None:
        downbeatInter = beatInter * 4
        downbeatDiff = abs(downbeat[0] - enterTime) % downbeatInter
        if downbeatDiff > (0.5 * downbeatInter):
            downbeatDiff = downbeatInter - downbeatDiff
        print('downbeat0', downbeat[0])
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
        s = str.format('{0}, {1}, {2}, {3}, {4}, {5}', *r)        
        print(s)
        logger.info(s)

    for t in processes:
        t.join()    

    print('done')



if __name__ == '__main__':
    
    #test()
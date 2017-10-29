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
import calc_bpm
import matplotlib.pyplot as plt


FPS = 100

def save_file(beats, mp3filename, postfix = ''):
    outname = os.path.splitext(mp3filename)[0]
    outname = outname + postfix + '.csv'
    librosa.output.times_csv(outname, beats)

def list_file(path):
    files = [os.path.join(path, f) for f in listdir(path) if os.path.splitext(f)[1] in ('.mp3', '.m4a', '.ogg')]
    return files


def do_work(filelist, levelInfo, processer, downbeatTracking, result):
    print('process ', os.getpid())
    for file in filelist:
        id = os.path.basename(file).split('.')[0]
        id = id.split('_')[-1]
        level = levelInfo[id]
        etManual = float(level[1]) / 1000.0

        print('id', id)
        act = processer(file)
        beat = downbeatTracking(act)
        firstBeat, lastBeat = normalizeInterval(beat)

        if firstBeat == -1:
            print('%s generate error, abnormal rate %f' % (id, lastBeat))
            r = [id, -1, lastBeat]
            result.put(r)
            continue

        bpm, etAuto = calcDownBeat(beat, firstBeat, lastBeat)
        beatInter = 60.0 / bpm
        # beat = beat[:,0]
        # # save_file(beat, filename, '_beat')

        if levelInfo != None:
            barManual = 240.0 / float(level[0])
            barAuto = beatInter * 4

            etDiff = abs(etAuto - etManual) % barAuto
            if etDiff > (0.5 * barAuto):
                etDiff = barAuto - etDiff    
            r2 = [id, bpm, float(level[0]), barAuto, barManual, etAuto, etManual, etDiff]
            result.put(r2)

        lastBeat = beat[-1, 0]
        SaveDownbeat(bpm, etAuto, lastBeat, file)


def normalizeInterval(beat, threhold = 0.03):
    interval = beat[1:,0] - beat[:-1, 0]

    aver = np.average(interval)
    diff = np.abs(interval - aver)

    abnormal = np.nonzero(diff > threhold)[0]
    print('abnormal count', abnormal.size, abnormal)

    if abnormal.size > beat.shape[0] / 3:
        return 0, len(beat)
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

def calcDownBeat(beat, firstBeat, lastBeat):
    firstBeatTime = beat[0, 0]
    newBeat = beat[firstBeat:lastBeat]
    downbeat = madmom.features.downbeats.filter_downbeats(newBeat)

    newBeat = newBeat[:,0]
    barInter, et = calc_bpm.CalcBarInterval(downbeat)

    #et = downbeat[0]
    print('first down beat', et)
    while (et - barInter) > firstBeatTime:
        et = (et - barInter)
    
    bpm = 240.0 / barInter

    return bpm, et

def SaveDownbeat(bpm, et, lastBeat, filename):
    downbeatInter = 240.0 / bpm
    numBar = ((lastBeat - et) // downbeatInter) + 1
    downbeat = np.arange(numBar) * downbeatInter + et
    save_file(downbeat, filename, '_downbeat')



def test():
    
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1351.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1446.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1417.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1370.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1374.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1352.ogg'
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1347.ogg'

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


    idlist = [1254, 1400, 1446, 1447, 1449, 1462, 1463, 1465, 1475, 1478, 1488, 1491] #拍子减半
    # 1400 快了一倍？
    # 1254 1447 1488 1491 觉得没问题
    # 1463 bpm对,et 差两拍
    # 1446 查乐谱是对的
    idlist = ['Nightwish - Bye Bye Beautiful', '胡夏 - 爱夏', 'See You Again', '王力宏 - 就是现在', '林俊杰 - 你有没有过', '拜你所赐', 'Bad Blood', '林昕阳 - 外婆的澎湖湾']

    path = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_%d.ogg'
    if os.name == 'posix':
        path = '/Users/xuchao/Music/网易云音乐/%s.mp3'

    if len(idlist) > 0:
        filelist = [path % i for i in idlist]
        filename = filelist[0]
        

    levelInfo = LevelInfo.load_levelinfo_file('D:/ab/QQX5_Mainland/exe/resources/level/level-infos.xml')
    if levelInfo != None:
        id = os.path.basename(filename).split('.')[0]
        id = id.split('_')[-1]
        level = levelInfo[id]
        etManual = float(level[1]) / 1000.0
    
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda=1000, fps=FPS)    

    print(filename)
    act = processer(filename)
    beat = downbeatTracking(act)
    downbeat_orig = madmom.features.downbeats.filter_downbeats(beat)
    
    firstBeat, lastBeat = normalizeInterval(beat)

    if firstBeat == -1:
        print('%s generate error, abnormal rate %f' % (id, lastBeat))
        return

    bpm, etAuto = calcDownBeat(beat, firstBeat, lastBeat)
    beatInter = 60.0 / bpm
    print('bpm', bpm)
    # beat = beat[:,0]
    # # save_file(beat, filename, '_beat')

    if levelInfo != None:
        barManual = 240.0 / float(level[0])
        barAuto = beatInter * 4

        etDiff = abs(etAuto - etManual) % barAuto
        if etDiff > (0.5 * barAuto):
            etDiff = barAuto - etDiff    
        r2 = [id, bpm, level[0], barAuto, barManual, etAuto, etManual, etDiff]
        print(r2)

    lastBeat = beat[-1, 0]
    SaveDownbeat(bpm, etAuto, lastBeat, filename)
    #save_file(downbeat_orig, filename, '_downbeatorig')

def doMultiProcess(numWorker = 4):
    resLog = logger.Logger('result.log', to_console=True)
    errLog = logger.Logger('error.log', to_console=True)

    filelist = list_file(r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music')
    filelist = filelist[200:230]

    idlist = [1266, 1407, 1404, 1262]
    idlist = [1254, 1400, 1446, 1447, 1449, 1462, 1463, 1465, 1475, 1478, 1488, 1491] #拍子减半
    idlist = [1262, 1279, 1374, 1391] #差两拍
    idlist = [1347, 1426] #差拍
    idlist = [1245] #不准
    idlist = []

    path = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_%d.ogg'
    if os.name == 'posix':
        path = '/Users/xuchao/Music/网易云音乐/%s.mp3'

    if len(idlist) > 0:
        filelist = [path % i for i in idlist]

    
    lists = [filelist[i::numWorker] for i in range(numWorker)]
    queue = mp.Queue()
    
    levelInfo = LevelInfo.load_levelinfo_file('D:/ab/QQX5_Mainland/exe/resources/level/level-infos.xml')

    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda = 1000, fps=FPS)

    processes = []

    for i in range(numWorker):
        if len(lists[i]) > 0:
            t = mp.Process(target=do_work, args=(lists[i], levelInfo, processer, downbeatTracking, queue))
            processes.append(t)
            t.start()

    for i in range(len(filelist)):
        r = queue.get(True)
        if r[1] == -1:
            s = str.format('{0}, too many abnormal rate {1}', r[0], r[2])
            errLog.info(s)
        else :
            s = str.format('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}', *r) 
            # 小节时长差别应该小于1毫秒
            if abs(r[3] - r[4]) > 0.001:
                errLog.info(s)
            else:
                resLog.info(s)

    for t in processes:
        t.join()    

    print('done')

def study():
    
    filename = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_1409.ogg'
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=FPS)    

    act = processer(filename)

    beatPropo = act[:,0]
    downbeatPropo = act[:,1]

    plt.plot(beatPropo)
    plt.show()




if __name__ == '__main__':    
    #doMultiProcess(4)
    test()
    #study()



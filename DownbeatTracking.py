import madmom
import numpy as np
from os import listdir
import os.path
import LevelInfo
import logger
import multiprocessing as mp
import time
import logger
import calc_bpm
import matplotlib.pyplot as plt
import librosa


FPS = 100

def SaveInstantValue(beats, filename, postfix = ''):
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True


def list_file(path):
    files = [os.path.join(path, f) for f in listdir(path) if os.path.splitext(f)[1] in ('.mp3', '.m4a', '.ogg')]
    return files


def do_work(filelist, levelInfo, processer, downbeatTracking, result):
    print('process ', os.getpid())
    for file in filelist:
        id = os.path.basename(file).split('.')[0]
        id = id.split('_')[-1]

        try:            
            level = levelInfo[id]
        except:
            r = [id, -1, 0]
            result.put(r)
            continue

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

        bpm, etAuto = CalcBPM(beat, firstBeat, lastBeat)
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


def do_work2(filelist, levelInfo, processer, downbeatTracking, result):
    print('process ', os.getpid())
    for file in filelist:
        id = os.path.basename(file).split('.')[0]
        id = id.split('_')[-1]

        try:            
            level = levelInfo[id]
        except:
            r = [id, -1, 0]
            result.put(r)
            continue

        level = levelInfo[id]
        etManual = float(level[1]) / 1000.0

        print('id', id)
            
        y, sr = librosa.load(file, mono=True, sr=44100)
        duration = librosa.get_duration(y=y, sr=sr)
        start = int(duration * 0.3)
        clipTime = np.array([start, start+90])
        print('duration', duration, 'start', start)
        clip = librosa.time_to_samples(clipTime, sr=sr)
        print('total', y.shape, 'clip', clip)
        yy = y[clip[0]:clip[1]]


        act = processer(yy)
        beat = downbeatTracking(act)
        firstBeat, lastBeat = CalcAbnormal(beat)        

        if firstBeat == -1:
            print('%s generate error, abnormal rate %f' % (id, lastBeat))
            r = [id, -1, lastBeat, 0, 0, 0, 0, 0]
            result.put(r)
            continue

        # first 10 seconds samples
        yy = y[:441000]
        act = processer(yy)
        beginBeat = downbeatTracking(act)                    
        firstBeatTime = beginBeat[0, 0]

        # newBeat = beat[firstBeat:lastBeat]
        newBeat = beat
        downbeat = madmom.features.downbeats.filter_downbeats(newBeat)
        
        downbeat = downbeat + start
        barInter, etAuto = calc_bpm.CalcBarInterval2(downbeat)

        #et = downbeat[0]
        print('first down beat', etAuto)
        if abs(etAuto - firstBeatTime) > barInter:
            etAuto += ((firstBeatTime - etAuto) // barInter + 1) * barInter

        assert abs(etAuto - firstBeatTime) < barInter
        
        bpm = 240.0 / barInter

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



def CalcAbnormal(beat, threhold = 0.02):
    interval = beat[1:,0] - beat[:-1, 0]

    aver = np.average(interval)
    diff = np.abs(interval - aver)

    abnormal = np.nonzero(diff > threhold)[0]
    print('abnormal count', abnormal.size, abnormal)

    if abnormal.size > beat.shape[0] / 3:
        # 拍子均匀程度太低，自动生成失败
        return -1, abnormal.size / beat.shape[0]

    return 0, len(beat)



def CalcBPM(beat, firstBeat, lastBeat):
    firstBeatTime = beat[0, 0]
    newBeat = beat[firstBeat:lastBeat]
    downbeat = madmom.features.downbeats.filter_downbeats(newBeat)

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
    SaveInstantValue(downbeat, filename, '_downbeat')


def CalcMusicInfoFromFile(filename):
    y, sr = librosa.load(filename, mono=True, sr=44100)
    logger.info('loaded')
    duration = librosa.get_duration(y=y, sr=sr)

    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda = 1000, fps=FPS)
    
    act = processer(y)
    beat = downbeatTracking(act)
    firstBeat, lastBeat = normalizeInterval(beat)

    if firstBeat == -1:
        print('generate error, abnormal rate %f' % (lastBeat))
        return

    bpm, etAuto = CalcBPM(beat, firstBeat, lastBeat)
    beatInter = 60.0 / bpm

    lastBeat = beat[-1, 0]
    print('bpm', bpm, 'et', etAuto)

    dir = os.path.dirname(filename) + os.path.sep
    infoFileName = dir + 'info.txt'
    with open(infoFileName, 'w') as file:
        file.write('duration=%f\nbpm=%f\net=%f' % (duration, bpm, etAuto))

    # SaveDownbeat(bpm, etAuto, lastBeat, filename)


def CalcMusicInfo():
    filename = r'd:\leveledtior\client\Assets\resources\audio\bgm\jilejingtu.m4a'
    filename = r'd:\librosa\RhythmMaster\BBoomBBoom\BBoomBBoom.mp3'

    filelist = [
        # r'd:\librosa\RhythmMaster\dainiqulvxing\dainiqulvxing.',
        # r'd:\librosa\RhythmMaster\aLIEz\aLIEz.'
        # r'd:\librosa\RhythmMaster\foxishaonv\foxishaonv.',
        # r'd:\librosa\RhythmMaster\xiagelukoujian\xiagelukoujian.',
        r'd:\librosa\RhythmMaster\CheapThrills\CheapThrills.'
    ]

    for f in filelist:
        filename = f + 'm4a'
        if not os.path.exists(filename):
            filename = f + 'mp3'

        if not os.path.exists(filename):
            print('can not find', filename)
            continue

        CalcMusicInfoFromFile(filename)




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

    bpm, etAuto = CalcBPM(beat, firstBeat, lastBeat)
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
    err0Log = logger.Logger('error0.log', to_console=True)
    err1Log = logger.Logger('error1.log', to_console=True)

    filelist = list_file(r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music')
    filelist = filelist[1000:1100]

    idlist = [1262, 1279, 1374, 1391] #差两拍
    idlist = [1254, 1400, 1446, 1447, 1449, 1462, 1463, 1465, 1475, 1478, 1488, 1491] #拍子减半
    idlist = [1245] #bpm有点不准
    idlist = [1374, 1370]

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

    startTime = time.time()
    numError = 0
    processes = []

    for i in range(numWorker):
        if len(lists[i]) > 0:
            t = mp.Process(target=do_work2, args=(lists[i], levelInfo, processer, downbeatTracking, queue))
            processes.append(t)
            t.start()

    header = 'id\tbpm\tlevelbpm\tbarAuto\tbarManual\tetAuto\tetManual\tetDiff'
    err0Log.info(header)
    err1Log.info(header)
    for i in range(len(filelist)):
        r = queue.get(True)
        if r[1] == -1:
            s = str.format('{0}, too many abnormal rate {1}', r[0], r[2])
            err0Log.info(s)
        else :
            s = str.format('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}', *r) 
            # 小节时长差别应该小于1毫秒

            if abs(r[3] - r[4]) > 0.001:
                err0Log.info(s)
                numError = numError + 1
            elif r[7] > 0.06:
                err1Log.info(s)
                numError = numError + 1
            else:
                logger.info(s)

    for t in processes:
        t.join()    

    print()
    print('done in %.1f minute' % ((time.time() - startTime) / 60.0))
    print('%d abnormal in %d samples' % (numError, len(filelist)))

def study():
    
    filename = r'd:\librosa\RhythmMaster\4minuteshm\4minuteshm.mp3'
    filename = r'd:\leveledtior\client\Assets\resources\audio\bgm\4minuteshm.m4a'
    
    processer = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeatTracking = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, transition_lambda = 1000, fps=FPS)
    act = processer(filename)

    print('samples', act.shape)

    beat = downbeatTracking(act)
    firstBeat, lastBeat = normalizeInterval(beat)

    if firstBeat == -1:
        print('%s generate error, abnormal rate %f' % (id, lastBeat))
        r = [id, -1, lastBeat]
        result.put(r)
        return

    bpm, etAuto = CalcBPM(beat, firstBeat, lastBeat)
    print('bpm', bpm, 'et', etAuto)

def OnsetTest():
    onsetPorc = madmom.features.onsets.CNNOnsetProcessor()
    
    filename = r'd:\librosa\RhythmMaster\4minuteshm\4minuteshm.mp3'
    samples = onsetPorc(filename)

    picker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=0.99, smooth=0.0, fps=100)
    onsettime = picker(samples)
    print(len(onsettime))
    # print(onsettime)

    SaveInstantValue(onsettime, filename, '_onset')    




if __name__ == '__main__':    
    # doMultiProcess(8)
    #test()
    # OnsetTest()
    CalcMusicInfo()



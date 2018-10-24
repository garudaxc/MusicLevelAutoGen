import numpy as np
import scipy.stats
import pickle
import scipy.signal
import numpy.random
import bisect
import LevelInfo
import os
import DownbeatTracking
import madmom
import time
import NotePreprocess

def TrainDataToLevelData(data, sampleInterval, threhold, timeOffset=0):
    notes = []
    longNote = False
    last = 0
    start = 0  
    for i in range(len(data)):                
        d = data[i]
        t = int(i * sampleInterval + timeOffset)

        if not isinstance(d, np.ndarray):
            if (d > threhold):
                notes.append([t, 0, 0])
            continue

        dim = len(d)

        if (d[0] > threhold):
            notes.append([t, 0, 0])

        if (dim > 1 and d[1] > threhold):
            notes.append([t, 1, 0])

        if (dim > 2 and d[2] > threhold):
            if not longNote:
                start = t
            longNote = True
            last += sampleInterval            
        else:
            if longNote:
                if last <= sampleInterval:
                    print('long note last too short')
                else:
                    last = int(last)
                    notes.append([start, 2, last])
            longNote = False
            start = 0
            last = 0
    
    notes = np.array(notes)
    return notes



def pick(data, kernelSize = 21):
    
    n = len(data) // kernelSize
    dim = data.shape[1]
    newNotes = np.zeros(data.shape)
    for i in range(n):
        a = data[i*kernelSize:(i+1)*kernelSize]
        b = newNotes[i*kernelSize:(i+1)*kernelSize]
        maxindex = np.argmax(a, axis=0)
        for j in range(dim):
            b[:,j][maxindex[j]] = a[:,j][maxindex[j]]
    
    return newNotes


def SaveResult(prid, msMinInterval, time, pathname):
    # pathname = '/Users/xuchao/Documents/python/MusicLevelAutoGen/result.txt'
    # if os.name == 'nt':
    #     pathname = r'D:\librosa\result.log'
    with open(pathname, 'w') as file:
        n = 0
        for i in prid:
            t = time + n * msMinInterval
            minite = t // 60000
            sec = (t % 60000) // 1000
            milli = t % 1000

            file.write('%d:%d:%d, %s\n' % (minite, sec, milli, str(i)))
            n += 1

    print('saved ', pathname)


def ConvertIntermediaNoteToLevelNote(notes):
    '''
    choose track
    track = -1时，在相邻两个track随机一个，否则保持track,滑动音符时切换track（实现组合音符）
    '''
    result = []
    for n in notes:
        time, t, value, side = n
    
        channel = np.random.randint(0, 2)
        track = channel + side * 2

        time *= 10

        if t == LevelInfo.combineNode:
            cnote = []
            for nn in value:
                time2, type2, value2, side2 = nn
                time2 *= 10
                value2 *= 10
                cnote.append((time2, type2, value2, track))
                if type2 == LevelInfo.slideNote:
                    channel = 1 - channel
                    track = channel + side*2

            result.append((time, t, cnote, -1))
        else:
            value *= 10
            result.append((time, t, value, track))

    return result
    

def PickInstanceSample(short, count=500):
    '''
    根据阈值选取音符，迭代并调整阈值
    '''
    threhold = 0.9
    index = (short > threhold).nonzero()[0]
    while len(index) * 0.8 > count:
        threhold += 0.01
        index = (short > threhold).nonzero()[0]
        # print(threhold, len(index))

    newSample = np.zeros_like(short)    
    newSample[index] = short[index]
    
    maxDis = 3
    index = newSample.nonzero()[0]
    dis = index[1:] - index[0:-1]
    closeIndex = (dis<maxDis).nonzero()[0]
    index = index[closeIndex]
    index = index[:, np.newaxis]

    index = np.hstack((index, index+1, index+2))
    value = newSample[np.reshape(index, -1)].reshape((-1, maxDis))
    maxValueIndex = np.argmax(value, axis=1)

    selectIndex = index[range(index.shape[0]), maxValueIndex]
    selectValue = newSample[selectIndex]
    newSample[np.reshape(index, -1)] = 0
    newSample[selectIndex] = selectValue

    return newSample



def BilateralFilter(samples, ratio = 0.8):
    pRange = 31
    gauKernel = scipy.signal.gaussian(pRange, 10)
    # print(gauKernel)
    gauKernel = gauKernel / np.sum(gauKernel)
    side = pRange // 2
    print('side', side)
    # print(gauKernel, np.sum(gauKernel))
    gauSample = np.zeros_like(samples)
    newSample = np.zeros_like(samples)
    count = len(samples)
    
    for i in range(side, len(samples)-side):
        s = samples[i-side:i+side+1]
        gauSample[i] = np.dot(gauKernel, s)

        d = np.abs(s-samples[i]) ** 0.5
        kernel = gauKernel * (1-d)
        weight = np.sum(kernel)

        newSample[i] = np.dot(kernel, s) / weight


    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(samples[800:2500])
    # ax[1].plot(gauSample[800:2500])    
    # ax[2].plot(newSample[800:2500])
    # plt.show()

    # 计算rato比率计算阈值
    his, bins = np.histogram(newSample, 100)
    for i in range(1, len(his)):
        his[i] = his[i] + his[i-1]

    r = len(newSample) * ratio
    index = bisect.bisect_left(his, r)
    threhold = bins[index]
    result = np.zeros_like(samples)
    acc = 0
    for i in range(len(newSample)):
        if newSample[i] > threhold:
            result[i] = newSample[i]
            acc += (newSample[i] - threhold) * 2
            continue
        
        acc -= (threhold - newSample[i])
        if acc > 0:
            result[i] = newSample[i]
            continue
        
        acc = 0


    # index = samples > threhold
    # print(index.nonzero()[0].shape)

    # result[index] = newSample[index]

    # plt.plot(result)
    # plt.show()

    # plt.plot(his)
    # plt.vlines(index, 0, len(newSample))
    # plt.show()

    return result

def EliminateShortSamples(long, threhold=300):
    '''
    去掉过短的长音符
    threhold 单位毫秒
    '''
    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    result = np.copy(long)
    threhold /= 10
    for i in range(len(edge)-1):
        if (edge[i+1] - edge[i]) < threhold:
            # print('found too short long note', result[edge[i+1]-1], result[edge[i+1]], result[edge[i+1]+1])
            result[edge[i]:edge[i+1]] = 0
    
    return result
    
def GetLongNoteInfo(long):
    length = len(long)
    idx = 0
    searchBegin = True
    begin = 0
    end = 0
    longArr = []
    while idx < length:
        if searchBegin:
            if long[idx] > 0:
                begin = idx
                searchBegin = False
        else:
            if long[idx] <= 0 or idx == length - 1:
                end = idx
                searchBegin = True
                longArr.append([begin, end])

        idx += 1

    return np.array(longArr)    

def AlignNotePosition(short, long, threhold=100):
    '''
    将挨得很近的长音符和短音符对齐到同一位置
    threhold 范围内的会对齐 单位毫秒
    '''
    print('shape', short.shape, long.shape)
    assert short.shape == long.shape

    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    newEdge = []
    offset = int((threhold / 10) // 2)
    print('offset', offset, 'edge count', len(edge))
    for e in edge:
        beg = max(e-offset, 0)
        end = min(e+offset+1, len(short))
        pos = short[beg:end].nonzero()[0] - offset
        if len(pos) > 0:
            m = np.argmin(np.abs(pos))
            e += pos[m]

        newEdge.append(e)

    # remove very close edge
    threhold = 3
    i = 0
    length = len(newEdge)    
    while i < length-1:
        if newEdge[i+1] - newEdge[i] < threhold:
            print('remove', newEdge[i], newEdge[i+1])
            del newEdge[i]
            del newEdge[i+1]
            length -= 2        
        else:
            i += 1
    
    long_new = np.zeros_like(long)
    for i in range(0, len(newEdge), 2):
        long_new[newEdge[i]:newEdge[i+1]] = long[newEdge[i]:newEdge[i+1]]

    return long

def AlignNotePositionEx(short, long, threhold=100, limitStartAndEnd = False):
    '''
    将挨得很近的长音符和短音符对齐到同一位置
    threhold 范围内的会对齐 单位毫秒
    '''
    print('shape', short.shape, long.shape)
    assert short.shape == long.shape

    offset = int((threhold / 10) // 2)

    longArr = GetLongNoteInfo(long)
    long_new = np.zeros_like(long)
    for longNote in longArr:
        isValid = True
        for idx in range(2):
            begin = max(longNote[idx] - offset, 0)
            end = min(longNote[idx] + offset, len(short))
            pos = short[begin:end].nonzero()[0] + begin
            if len(pos) > 0:
                m = np.argmin(np.abs(pos - longNote[idx]))
                longNote[idx] = pos[m]
            elif idx == 0:
                isValid = False
                break
            elif limitStartAndEnd:
                isValid = False
                break

        if isValid and longNote[0] < longNote[1]:
            long_new[longNote[0]:longNote[1]] = 1 

    return long_new


def EliminateTooCloseSample(short, long, threhold=200):
    '''
    去掉和长音符挨的太近的音符
    '''
     #should calc with bpm
    assert short.shape == long.shape
    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    offset = int((threhold / 10) // 2)

    for i in range(len(edge)-1):
        p1 = edge[i]
        p0 = max(p1-offset, 0)
        pos = short[p0:p1].nonzero()[0] + p0
        if len(pos) > 0:
            # print('too close sample at', pos, i)
            short[pos] = 0

        p0 = edge[i+1]+1
        p1 = min(p0+offset+1, len(short))
        pos = short[p0:p1].nonzero()[0] + p0
        if len(pos) > 0:
            # print('too close sample at', pos, i+1)
            short[pos] = 0

    return short


def RandSide():
    # 随机生成0 1，作为side
    l = np.random.randint(0, 2)
    r = 1 - l
    return l, r

def NextNoteIdx(short, long, start):
    arrLength = min(len(short), len(long))
    i = start + 1
    idxShort = arrLength
    while i < arrLength:
        if short[i] > 0:
            idxShort = i
            break
        i += 1
    
    i = start + 1
    idxLong = arrLength
    while i < arrLength:
        if long[i] > 0:
            idxLong = i
            break
        i += 1

    if idxShort < idxLong:
        return idxShort

    return -1
    

def MutateSamples(short, long, bpm):
    '''
    生成滑动音符，双按音符
    '''
    SideLeft, SideRight = RandSide()
    # 1 normal short, 2 slide, 3 double short, 4 double slide
    DoubleSlide = 4
    DoubleShort = 3
    Slide = 2
    doubleRate = 0.0
    param = [('doubleSlide', 0.0, 4), ('doubleRate', 0.0, 3), ('slideRate', 0.0, 2)]
    # param = [('doubleSlide', 0.0, 4), ('doubleRate', 0.0, 3), ('slideRate', 0.0, 2)]
    
    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    longNoteValue = []
    for i in range(0, len(edge), 2):
        value = np.average(long[edge[i]:edge[i+1]])
        longNoteValue.append((value, i))
    
    longNoteValue.sort(reverse=True)
    long2 = np.zeros_like(long)
    count = int(len(longNoteValue) * doubleRate)
    print('double long count', count)
    # 选取value最大的为双按音符
    for i in range(count):
        e = longNoteValue[i][1]
        long2[edge[e]:edge[e+1]] = long[edge[e]:edge[e+1]]

    sortIndex = np.argsort(short)    
    sortIndex = np.flip(sortIndex, axis=0)

    shortMute = np.zeros_like(short)
    totalCount = np.count_nonzero(short)
    print('total short count', totalCount)

    begin = 0
    for n, ratio, id in param:
        count = int(totalCount * ratio)
        index = sortIndex[begin:begin+count]
        print('id %d name %s count %d' % (id, n, len(index)))
        shortMute[index] = id
        begin += count

    # generate notes from samples
    notes = []
    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    i = 0
    while i < len(short):
        if long[i] > 0:
            #long note
            end = bisect.bisect_right(edge, i)
            end = edge[end]
            
            s = short[i:end+1].nonzero()[0]+i
            if len(s) > 0:
                # 暂时注掉组合音符
                # #combine note
                # cnotes = TransferCombineNote(i, end, s, SideLeft)
                # # print('combine note', cnotes)   
                # notes.append((i, LevelInfo.combineNode, cnotes, SideLeft))
            
                notes.append((i, LevelInfo.longNote, end-i, SideLeft))
                
                for p in s:
                    notes.append((p, LevelInfo.shortNote, 0, SideRight))

                # tempShort = []
                # lengthShort = len(s)
                # interval = ((60 / bpm) * 1000 * 0.95) // 10
                # beatInterval = ((60 / bpm) * 1000) // 10
                # for shortIdx in range(lengthShort):
                #     isValidShort = True
                #     p = s[shortIdx]
                #     if shortIdx > 0:
                #         if p - s[shortIdx - 1] < interval:
                #             isValidShort = False

                #     if shortIdx < lengthShort - 1:
                #         if s[shortIdx + 1] - p < interval:
                #             isValidShort = False

                #     if isValidShort or shortIdx == lengthShort - 1:
                #         if shortIdx == lengthShort - 1:
                #             tempShort.append(p)
                #         if len(tempShort) > 1:
                #             if tempShort[-1] - tempShort[0] > interval:
                #                 notes.append((tempShort[0], LevelInfo.longNote, tempShort[-1] - tempShort[0], SideRight))
                #             elif p - (tempShort[0] + beatInterval) > interval:
                #                 notes.append((tempShort[0], LevelInfo.longNote, beatInterval, SideRight))
                #         notes.append((p, LevelInfo.shortNote, 0, SideRight))
                #     else:
                #         tempShort.append(p)

            else:
                #long note
                notes.append((i, LevelInfo.longNote, end-i, SideLeft))
            
            # if long2[i] > 0:
            #     #double long                
            #     if len(s) > 0:
            #         #combine note
            #         cnotes = TransferCombineNote(i, end, s, SideRight)
            #         notes.append((i, LevelInfo.combineNode, cnotes, SideRight))
            #     else:
            #         #long note
            #         notes.append((i, LevelInfo.longNote, end-i, SideRight))
            # else:
            #     # short in other side
            #     for n in s:
            #         if (shortMute[n] == DoubleSlide) or (shortMute[n] == Slide):
            #             # print('slide beside combine')
            #             notes.append((i, LevelInfo.slideNote, 0, SideRight))
            #         elif shortMute[n] == DoubleShort:
            #             # print('short beside combine')
            #             notes.append((i, LevelInfo.shortNote, 0, SideRight))

            i = end+1
            # SideLeft, SideRight = RandSide()
            SideLeft = 1 - SideLeft
            SideRight = 1 - SideRight

        elif shortMute[i] > 0 or short[i] > 0:
            # nextIdx = NextNoteIdx(short, long, i)
            # interval = ((60 / bpm) * 1000 * 1.1) // 10
            # maxLength = ((60 / bpm) * 1000 * 2) // 10
            # if nextIdx > 0 and nextIdx - i > interval:
            #     longNoteLength = min(nextIdx - i, maxLength)
            #     notes.append((i, LevelInfo.longNote, longNoteLength, SideLeft))
            #     notes.append((nextIdx, LevelInfo.shortNote, 0, SideRight))
            #     i = nextIdx + 1
            #     SideLeft = 1 - SideLeft
            #     SideRight = 1 - SideRight
            #     continue

            if shortMute[i] == DoubleSlide:
                notes.append((i, LevelInfo.slideNote, 0, SideLeft))
                notes.append((i, LevelInfo.slideNote, 0, SideRight))
            elif shortMute[i] == DoubleShort:
                notes.append((i, LevelInfo.shortNote, 0, SideLeft))
                notes.append((i, LevelInfo.shortNote, 0, SideRight))
            elif shortMute[i] == Slide:
                notes.append((i, LevelInfo.slideNote, 0, SideLeft))
            else:
                notes.append((i, LevelInfo.shortNote, 0, SideLeft))

            i += 1
            # SideLeft, SideRight = RandSide()
            SideLeft = 1 - SideLeft
            SideRight = 1 - SideRight
        else:
            i += 1

    print('got notes', len(notes))

    return notes


def TransferCombineNote(begin, end, shortPos, side):
    notes = []
    i = 0
    if shortPos[i] == begin:
        notes.append((begin, LevelInfo.slideNote, 0, side))
        i += 1

    while i < len(shortPos):
        notes.append((begin, LevelInfo.longNote, shortPos[i]-begin, side))
        notes.append((shortPos[i], LevelInfo.slideNote, 0, side))
        begin = shortPos[i]
        i += 1

    if begin == end:
        notes.append((end, LevelInfo.slideNote, 0, side))
    else:
        notes.append((begin, LevelInfo.longNote, end-begin, side))
        
    return notes


# def SampleDistribute(samples):
#     '''
#     compute distribute in unit time
#     '''
#     if samples is None:
#         with open('d:/raw_sample.raw', 'rb') as file:
#             samples = pickle.load(file)

#     assert samples.ndim == 1
#     stride = 800
#     numUnit = len(samples) // stride
#     print('units count', numUnit)

#     distri = np.zeros((stride + 1))
#     gammaSample = np.zeros((numUnit))
#     for i in range(numUnit):
#         unit = samples[i*stride:(i+1)*stride]
#         count = (unit > 0.9).nonzero()[0].shape[0]
#         distri[count] += 1
#         gammaSample[i] = float(count)

#     rang = 60
#     totalCount = np.sum(distri)
#     distri = distri / float(numUnit)
#     distri = distri[0:rang]
#     plt.plot(distri)
#     # plt.show()

#     args = scipy.stats.gamma.fit(gammaSample)
#     print(args)
#     maxgamma = np.max(gammaSample)
#     y = scipy.stats.gamma.pdf(range(rang), args[0], loc=args[1], scale=args[2])
#     plt.plot(y)

    
#     fig, ax = plt.subplots(1, 1)
#     mu = totalCount / float(numUnit)
#     print('mu', mu)
#     x = np.arange(poisson.ppf(0.01, mu),
#                   poisson.ppf(0.99, mu))
#     ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
#     ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

#     plt.show()


def SaveInstantValue(beats, filename, postfix=''):
    import os
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True


def LoadInstanceValue(filename):
    with open(filename, 'r') as file:
        data = file.read()
    
    lines = data.split('\n')
    result = []

    for i in range(len(lines)):
        if lines[i] != '':
            result.append(int(lines[i]))

    result = np.array(result)
    return result

def TestInputParam(input):
    # 输入参数可以作为返回值

    print(input)

    input[3:4] = 0

    print(input)

    return input



def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
    return path

def MakeMp3Dir(song):
    path = GetSamplePath()
    pathname = '%s%s/' % (path, song)
    if not os.path.exists(pathname):
        assert False
    return pathname

def SaveDownbeat(bpm, et, duration, filename):
    downbeatInter = 240.0 / bpm
    numBar = ((duration - et) // downbeatInter) + 1
    downbeat = np.arange(numBar) * downbeatInter + et
    SaveInstantValue(downbeat, filename, '_downbeat')

def ProcessSampleToIdolLevel(rawFileLong, rawFileShort):

    np.random.seed(0)

    print('load raw file')
    with open(rawFileLong, 'rb') as file:
        predicts = pickle.load(file)

    pre = predicts[:, 1]
    long = BilateralFilter(pre, ratio=0.85)
    long = EliminateShortSamples(long)
    
    with open(rawFileShort, 'rb') as file:
        predicts = pickle.load(file)

    short = predicts[:, 1]
    short = PickInstanceSample(short, count=400)
    
    long = AlignNotePosition(short, long)

    short = EliminateTooCloseSample(short, long)

    notes = MutateSamples(short, long) 
    levelNotes = ConvertIntermediaNoteToLevelNote(notes)

    return levelNotes

def AlignLongNote(long, bpm, et):
    beatInterval = 60.0 / bpm
    beatInterval *= 1000
    posPerbeat = 8
    posInterval = beatInterval / posPerbeat

    long_new = np.zeros_like(long)
    longArr = GetLongNoteInfo(long)
    for longNote in longArr:
        begin = longNote[0]
        end = longNote[1]
        begin = round((begin * 10 - et) / beatInterval)
        begin = (begin * beatInterval + et + posInterval / 2) / 10
        begin = int(begin)

        end = round((end * 10 - et) / beatInterval)
        end = (end * beatInterval + et + posInterval / 2) / 10
        end = int(end)

        if begin < end and end < len(long):
            long_new[begin: end] = 1

    return long_new


def ProcessSampleToIdolLevel2(long, short, bpm, et):
    # 

    np.random.seed(0)

    short = short[:long.shape[0]]

    # long = BilateralFilter(long, ratio=0.9)
    
    beatInterval = int((60 / bpm) * 1000)
    # long = AlignLongNote(long, bpm, et)
    long = EliminateShortSamples(long, threhold=(beatInterval * 0.9))

    # short = EliminateTooCloseSample(short, long)
    
    # long = np.zeros_like(long)
    notes = MutateSamples(short, long, bpm) 
    levelNotes = ConvertIntermediaNoteToLevelNote(notes)

    return levelNotes

def BeatAndPosInfo(bpm):
    beatPerBar = 4
    posPerBeat = 8
    beatInterval = 60 / bpm
    posInterval = 60 / bpm / posPerBeat
    return beatInterval, posInterval, beatPerBar, posPerBeat

def LongNoteActivationProcess(predicts):
    threshold = 0.7
    mergeFrameInterval = 5
    longDuration = predicts[:, 2]
    temp = np.zeros_like(longDuration)
    frameCount = len(longDuration)
    for idx in range(frameCount):
        if longDuration[idx] >= threshold:
            temp[idx] = 1

    findBegin = False
    curContinueZero = []
    for idx in range(frameCount):
        if not findBegin:
            if temp[idx] == 1:
                findBegin = True
            continue

        if temp[idx] == 1:
            if len(curContinueZero) > 0 and len(curContinueZero) <= mergeFrameInterval:
                temp[curContinueZero] = 1
            curContinueZero = []
            continue

        if len(curContinueZero) > mergeFrameInterval:
            findBegin = False
            curContinueZero = []
            continue

        curContinueZero.append(idx)

    return temp

def SplitLongNoteWithDuration(longNote, fps, bpm, beatThreshold):
    beatInterval, posInterval, beatPerBar, posPerBeat = BeatAndPosInfo(bpm)
    thresholdFrameCount = int(beatThreshold * beatInterval * fps)
    longArr = GetLongNoteInfo(longNote)
    shorter = np.zeros_like(longNote)
    longer = np.zeros_like(longNote)
    for note in longArr:
        begin = note[0]
        end = note[1]
        noteFrameCount = end - begin
        if noteFrameCount > thresholdFrameCount:
            longer[begin:end] = 1
        else:
            shorter[begin:end] = 1

    return shorter, longer

def ShorterLongNote(shortNote, longNote, fps, bpm):
    if len(shortNote) != len(longNote):
        print('short len != long len')
        return [], []

    shortIdx = np.nonzero(shortNote)[0]
    if len(shortIdx) < 2:
        return shortNote, np.zeros_like(longNote)

    tempShortNote = np.zeros_like(shortNote)
    tempLongNote = np.zeros_like(longNote)
    beatInterval, posInterval, beatPerBar, posPerBeat = BeatAndPosInfo(bpm)
    thresholdFrameCount = int(1 * beatInterval * fps)
    minFrameCount = int(1 * beatInterval * fps * 0.9)
    for i in range(len(shortIdx) - 1):
        noteIdx = shortIdx[i]
        nextNoteIdx = shortIdx[i+1]
        if nextNoteIdx - noteIdx < minFrameCount:
            tempShortNote[noteIdx] = 1
            continue

        searchStart = noteIdx
        searchEnd = min(nextNoteIdx, searchStart + thresholdFrameCount)
        isLong = False
        idx = 0
        for j in range(searchStart, searchEnd):
            if longNote[j] > 0:
                isLong = True
                idx = j
                break
        if isLong:
            start = searchStart
            end = idx
            for j in range(idx, nextNoteIdx):
                if longNote[j] > 0:
                    end = j
                else:
                    break

            end = max(end, start + thresholdFrameCount)
            end = min(end, nextNoteIdx - 1)
            if start >= end:
                continue

            tempLongNote[start:end] = 1
        else:
            tempShortNote[noteIdx] = 1

    tempShortNote[shortIdx[-1]] = 1

    return tempShortNote, tempLongNote

def MergeLongNote(longNotesArr):
    if len(longNotesArr) < 1:
        return []

    baseArr = longNotesArr[0]
    if len(longNotesArr) >= 2:
        for idx in range(1, len(longNotesArr)):
            arr = longNotesArr[idx]
            info = GetLongNoteInfo(arr)
            for note in info:
                baseArr[note[0]:note[1]] = 1

    return baseArr

def AlignNoteWithBPMAndET(notes, frameInterval, bpm, et):
    '''
    notes 10 ms
    et ms
    '''
    beatInterval = 60.0 / bpm
    beatInterval *= 1000
    posPerbeat = 8
    posInterval = beatInterval / posPerbeat
    secondPerBeat = 60.0 / bpm
    noteTimes = []
    alignedPos = []
    maxNotePerBeat = 2
    posScale = posInterval * (posPerbeat / maxNotePerBeat)
    for i in range(len(notes)):
        if notes[i] < 1:
            continue

        timeInMS = i * frameInterval - et
        pos = round(timeInMS / posScale)
        addPos = pos
        # if pos % maxNotePerBeat == 0:
        #     addPos = pos
        # else:
        #     lenAlignedPos = len(alignedPos)
        #     if lenAlignedPos > 0 and (pos - alignedPos[-1]) < maxNotePerBeat:
        #         subPos = (timeInMS - noteTimes[-1]) / posScale
        #         if subPos > 1.5:
        #             subPos = 2
        #             addPos = alignedPos[-1] + subPos
        #         else:
        #             addPos = pos
        #     else:
        #         addPos = addPos

        if len(alignedPos) > 0:
            if addPos <= alignedPos[-1]:
                continue

        alignedPos.append(addPos)
        noteTimes.append(timeInMS)

    halfBeatInfoDic = {}
    for idx in range(len(alignedPos)):
        pos = alignedPos[idx]
        if pos % maxNotePerBeat != 0:
            hasPre = (idx > 0) and (pos - alignedPos[idx - 1] == 1)
            hasPost = (idx < len(alignedPos) -1) and (alignedPos[idx + 1] - pos == 1)
            count = 0
            if hasPre:
                count += 1
            if hasPost:
                count += 1
            halfBeatInfoDic[pos] = (pos, hasPre, hasPost, count)

    # DownbeatTracking.SaveInstantValue(alignedPos, MakeMp3Pathname('jinjuebianjingxian'), '_alignpos')
    newNotes = [0.0] * len(notes)
    for i in range(len(alignedPos)):
        pos = alignedPos[i]
        idx = int((pos * posScale + et + posInterval / 2) / frameInterval)
        if idx >= len(notes):
            continue

        # 只移除孤立的半拍
        if pos % maxNotePerBeat != 0:
            curHalfBeat = halfBeatInfoDic[pos]
            if curHalfBeat[3] == 0:
                halfBeatInfoDic.pop(pos)
                continue


        # if pos % maxNotePerBeat != 0:
        #     curHalfBeat = halfBeatInfoDic[pos]
        #     if (pos - maxNotePerBeat not in halfBeatInfoDic) and (pos + maxNotePerBeat not in halfBeatInfoDic):
        #         # if curHalfBeat[3] != 2:
        #         halfBeatInfoDic.pop(pos)
        #         continue
        #     elif curHalfBeat[3] == 0:
        #         threshold = 4
        #         tempCount = 0
        #         for tempIdx in range(-(threshold - 1), threshold):
        #             if pos + maxNotePerBeat * tempIdx not in halfBeatInfoDic:
        #                 tempCount = 0
        #                 continue
                    
        #             halfBeat = halfBeatInfoDic[pos + maxNotePerBeat * tempIdx]
        #             if halfBeat[3] == 0:
        #                 tempCount += 1
                    
        #             if tempCount == threshold:
        #                 break

        #         if tempCount < threshold:
        #             halfBeatInfoDic.pop(pos)
        #             continue

        #     elif not curHalfBeat[2]:
        #         if pos - maxNotePerBeat not in halfBeatInfoDic:
        #             halfBeatInfoDic.pop(pos)
        #             continue
        #         else:
        #             beat = halfBeatInfoDic[pos - maxNotePerBeat]
        #             if beat[3] != 2:
        #                 halfBeatInfoDic.pop(pos)
        #                 continue        
        #     else:
        #         if not curHalfBeat[1] and (pos - maxNotePerBeat not in halfBeatInfoDic):
        #             halfBeatInfoDic.pop(pos)
        #             continue

        #         if (pos - maxNotePerBeat in halfBeatInfoDic) and (pos - 2 * maxNotePerBeat in halfBeatInfoDic):
        #             beatA = halfBeatInfoDic[pos - maxNotePerBeat]
        #             beatB = halfBeatInfoDic[pos - 2 * maxNotePerBeat]
        #             if beatA[3] != 0 and beatB[3] != 0:
        #                 halfBeatInfoDic.pop(pos)
        #                 continue
        newNotes[idx] = 1

    return np.array(newNotes)

def GenerateLevelImp(songFilePath, duration, bpm, et, shortPredicts, longPredicts, levelFilePath, templateFilePath, onsetThreshold, shortThreshold, saveDebugFile = False, onsetActivation = None, enableDecodeOffset=True):    
    startTime = time.time()
    print('bpm', bpm, 'et', et, 'dur', duration)
    fps = 100
    frameInterval = int(1000 / fps)

    time1 = time.time()
    if onsetActivation is None:
        onsetProcessor = madmom.features.onsets.CNNOnsetProcessor()
        onsetActivation = onsetProcessor(songFilePath)

    frameCount = len(onsetActivation)
    print('pick cost', time.time() - time1)

    singingPredicts = shortPredicts
    singingActivation = singingPredicts[:, 1]
    dis_time = 60 / bpm / 8
    singingPicker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=shortThreshold, smooth=0.0, pre_max=dis_time, post_max=dis_time, fps=fps)
    singingTimes = singingPicker(singingActivation)
    print('sing pick count', shortThreshold, len(singingTimes))

    def IsPureMusic(singingTimes, bpm, duration):
        minInterval = 60 / bpm * 2
        minCount = int(duration / minInterval)
        print('singingTimes', len(singingTimes), 'minCount', minCount)
        return len(singingTimes) < minCount

    songIsPureMusic = IsPureMusic(singingTimes, bpm, duration / 1000)
    print('pure music', songIsPureMusic)
    if songIsPureMusic:
        print('adjust onset threshold from', onsetThreshold, 'to', onsetThreshold /2, 'for pure music')
        onsetThreshold = onsetThreshold / 2

    singingTimes = singingTimes * fps
    singingTimes = singingTimes.astype(int)
    singing = np.zeros_like(onsetActivation)
    singing[singingTimes] = 1

    if saveDebugFile:
        DownbeatTracking.SaveInstantValue(singingActivation, songFilePath, '_result_singing')
        if len(singingPredicts[0]) > 2:
            DownbeatTracking.SaveInstantValue(singingPredicts[:, 2], songFilePath, '_result_bg')
        if len(singingPredicts[0]) > 3:
            DownbeatTracking.SaveInstantValue(singingPredicts[:, 3], songFilePath, '_result_dur')
        DownbeatTracking.SaveInstantValue(singingPredicts[:, 0], songFilePath, '_result_no_label')
        DownbeatTracking.SaveInstantValue(singingTimes / fps, songFilePath, '_result_singing_pick')

    onset = DownbeatTracking.PickOnsetFromFile(songFilePath, bpm, duration, onsetThreshold, onsetActivation, saveDebugFile, songIsPureMusic)

    countBefore = [np.sum(singing), np.sum(onset)]
    singing = AlignNoteWithBPMAndET(singing, frameInterval, bpm, et)
    onset = AlignNoteWithBPMAndET(onset, frameInterval, bpm, et)
    countAfter = [np.sum(singing), np.sum(onset)]
    print('count before', countBefore, 'count after', countAfter)

    mergeShort = np.copy(onset)
    idxOffsetPre = int(60 / bpm * 1 * fps * 0.9)
    idxOffsetPost = int(60 / bpm * 1 * fps * 0.9)
    for singIdx in range(0, frameCount):
        if (singing[singIdx] < 1):
            continue
        
        idxStart = max(singIdx - idxOffsetPre, 0)
        idxEnd = min(singIdx + idxOffsetPost, frameCount)
        for idx in range(idxStart, idxEnd):
            mergeShort[idx] = 0

    checkShortCount = 0
    for val in mergeShort:
        if val > 0:
            checkShortCount += 1
    print('remove some short by singing. remain ', checkShortCount)
    mergeShort[singing > mergeShort] = 1
    countBefore = np.sum(mergeShort)
    mergeShort = AlignNoteWithBPMAndET(mergeShort, frameInterval, bpm, et)
    print('count before', countBefore, 'count after', np.sum(mergeShort))

    longNoteSrc = LongNoteActivationProcess(longPredicts)
    longNoteSrc.resize(frameCount)
    beatThreshold = 1.5
    shorter, longer = SplitLongNoteWithDuration(longNoteSrc, fps, bpm, beatThreshold)
    tempShort, tempLong = ShorterLongNote(mergeShort, shorter, fps, bpm)
    mergeShort = tempShort
    longNote = MergeLongNote([longer, tempLong])
    if saveDebugFile:
        DownbeatTracking.SaveInstantValue(longNoteSrc, songFilePath, '_long_processed')
        DownbeatTracking.SaveInstantValue(shorter, songFilePath, '_long_shorter')
        DownbeatTracking.SaveInstantValue(longer, songFilePath, '_long_longer')
        DownbeatTracking.SaveInstantValue(tempShort, songFilePath, '_long_temp_short')
        DownbeatTracking.SaveInstantValue(tempLong, songFilePath, '_long_temp_long')

    if enableDecodeOffset:
        mergeShort = NotePreprocess.AppendEmptyDataWithDecodeOffset(songFilePath, mergeShort, fps)
        longNote = NotePreprocess.AppendEmptyDataWithDecodeOffset(songFilePath, longNote, fps)
    levelNotes = ProcessSampleToIdolLevel2(longNote, mergeShort, bpm, et)
    LevelInfo.GenerateIdolLevel(levelFilePath, levelNotes, bpm, et, duration, templateFilePath)
    print('GenerateLevelImp cost', time.time() - startTime)

def Run():

    pathname = 'd:\librosa\RhythmMaster\dainiqulvxing\dainiqulvxing.mp3'


    if False:    
        print('load raw file')
        with open('d:/work/evaluate_data.raw', 'rb') as file:
            predicts = pickle.load(file)

        pre = predicts[:, 1]
        pre = PurifyInstanceSample(pre)

        picked = PickInstanceSample(pre)
        # print(picked)

        sam = np.zeros_like(pre)
        sam[picked] = 1
        
        notes = TrainDataToLevelData(sam, 10, 0.8, 0)
        notes = np.asarray(notes)
        notes[0]

        notes = notes[:,0]
        SaveInstantValue(notes, pathname, '_inst_new')
    

        # print('level file', levelFile)
        
        # notes = TrainDataToLevelData(long, 10, acceptThrehold, 0)
        # print('gen notes number', len(notes))
        # notes = notes[:,0]
        # SaveInstantValue(notes, pathname, '_region')        


def TestShortNote2():
    #使用阈值筛选短音符
    
    songname = 'mrq'
    songname = 'ribuluo'
    songname = 'CheapThrills'
    songname = 'dainiqulvxing'

    pathname = 'd:/librosa/RhythmMaster/%s/%s.mp3' % (songname, songname)

    np.random.seed(0)

    rawFile = 'd:/librosa/RhythmMaster/%s/evaluate_data_short_singing.raw' % (songname)
    rawFile = 'd:/librosa/RhythmMaster/%s/evaluate_data_short_beat.raw' % (songname)

    with open(rawFile, 'rb') as file:
        predicts = pickle.load(file)
        print('raw file loaded', predicts.shape)

    short = predicts[:, 1]
    short = PickInstanceSample(short)
        
    note = TrainDataToLevelData(short, 10, 0.1)
    note = note[:,0]
    print('note count', len(note))
    # SaveInstantValue(note, pathname, '_inst_singing')
    SaveInstantValue(note, pathname, '_inst_beat')

    

if __name__ == '__main__':

    TestShortNote2()
    # Run()

    # a = np.zeros(100)
    # a[range(0, 100, 5)] = 0.05

    # FindCumulativeIndexInArray(a, 1.2)
    # print(numpy.random.rand(1))
    
    # SampleDistribute2()

    # SampleDistribute(None)
    # Test()
    # Test2()

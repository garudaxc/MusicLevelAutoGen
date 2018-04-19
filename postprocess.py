import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import scipy.signal
import numpy.random
import bisect
import LevelInfo
import os


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
    

def PickInstanceSample(short, count=300):
    '''
    从大到小选取sample，并消去临近的sample
    '''

    sortarg = np.argsort(short)[-count:]
    newSample = np.zeros_like(short)
    
    newSample[sortarg] = short[sortarg]
    
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

def EliminateShortSamples(long, threhold=200):
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
    

def AlignNotePosition(short, long, threhold=100):
    '''
    将挨得很近的长音符和短音符对齐到同一位置
    threhold 范围内的会对齐 单位毫秒
    '''
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

def MutateSamples(short, long):
    '''
    生成滑动音符，双按音符
    '''
    SideLeft, SideRight = RandSide()
    # 1 normal short, 2 slide, 3 double short, 4 double slide
    DoubleSlide = 4
    DoubleShort = 3
    Slide = 2
    doubleRate = 0.15
    doubleRate = 0.0
    param = [('doubleSlide', 0.01, 4), ('doubleRate', 0.02, 3), ('slideRate', 0.05, 2)]
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
            
            s = short[i:end].nonzero()[0]+i
            if len(s) > 0:
                #combine note
                cnotes = TransferCombineNote(i, end, s, SideLeft)
                # print('combine note', cnotes)   
                notes.append((i, LevelInfo.combineNode, cnotes, SideLeft))
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
            SideLeft, SideRight = RandSide()

        elif shortMute[i] > 0 or short[i] > 0:
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
            SideLeft, SideRight = RandSide()
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


def SampleDistribute(samples):
    '''
    compute distribute in unit time
    '''
    if samples is None:
        with open('d:/raw_sample.raw', 'rb') as file:
            samples = pickle.load(file)

    assert samples.ndim == 1
    stride = 800
    numUnit = len(samples) // stride
    print('units count', numUnit)

    distri = np.zeros((stride + 1))
    gammaSample = np.zeros((numUnit))
    for i in range(numUnit):
        unit = samples[i*stride:(i+1)*stride]
        count = (unit > 0.9).nonzero()[0].shape[0]
        distri[count] += 1
        gammaSample[i] = float(count)

    rang = 60
    totalCount = np.sum(distri)
    distri = distri / float(numUnit)
    distri = distri[0:rang]
    plt.plot(distri)
    # plt.show()

    args = scipy.stats.gamma.fit(gammaSample)
    print(args)
    maxgamma = np.max(gammaSample)
    y = scipy.stats.gamma.pdf(range(rang), args[0], loc=args[1], scale=args[2])
    plt.plot(y)

    
    fig, ax = plt.subplots(1, 1)
    mu = totalCount / float(numUnit)
    print('mu', mu)
    x = np.arange(poisson.ppf(0.01, mu),
                  poisson.ppf(0.99, mu))
    ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

    plt.show()


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

def ProcessSampleToIdolLevel(songname):

    pathname = 'd:/librosa/RhythmMaster/%s/%s.mp3' % (songname, songname)
    levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (songname)

    np.random.seed(0)

    print('load raw file')
    path = MakeMp3Dir(song)
    rawfile = path + 'evaluate_data_long.raw'    
    with open(rawfile, 'rb') as file:
        predicts = pickle.load(file)

    pre = predicts[:, 1]
    long = BilateralFilter(pre, ratio=0.85)
    long = EliminateShortSamples(long)
    
    path = MakeMp3Dir(song)
    rawfile = path + 'evaluate_data_short.raw'
    with open(rawfile, 'rb') as file:
        predicts = pickle.load(file)

    short = predicts[:, 1]
    short = PickInstanceSample(short)
    
    long = AlignNotePosition(short, long)

    short = EliminateTooCloseSample(short, long)

    notes = MutateSamples(short, long) 
    levelNotes = ConvertIntermediaNoteToLevelNote(notes)

    duration, bpm, et = LevelInfo.LoadMusicInfo(pathname)

    LevelInfo.GenerateIdolLevel(levelFile, levelNotes, bpm, et, duration)



def Run():

    pathname = 'd:\librosa\RhythmMaster\dainiqulvxing\dainiqulvxing.mp3'

    a = os.path.split(pathname)[1]
    a = os.path.splitext(a)
    print(a)

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


def TestShortNote():
    # 从大到小筛选短音符
    
    songname = 'mrq'
    songname = 'ribuluo'

    pathname = 'd:/librosa/RhythmMaster/%s/%s.mp3' % (songname, songname)
    levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (songname)

    np.random.seed(0)

    rawFile = 'd:/librosa/RhythmMaster/%s/evaluate_data_short.raw' % (songname)
    with open(rawFile, 'rb') as file:
        predicts = pickle.load(file)
        print('raw file loaded', predicts.shape)

    short = predicts[:, 1]
    sortarg = np.argsort(short)[-240:]
    newSample = np.zeros_like(short)
    
    newSample[sortarg] = short[sortarg]

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
    
    note = TrainDataToLevelData(newSample, 10, 0.01)
    note = note[:,0]
    print('note count', len(note))
    SaveInstantValue(note, pathname, '_short')


def TestShortNote2():
    #使用阈值筛选短音符
    
    songname = 'mrq'
    songname = 'ribuluo'

    pathname = 'd:/librosa/RhythmMaster/%s/%s.mp3' % (songname, songname)
    levelFile = 'd:/LevelEditor_ForPlayer_8.0/client/Assets/LevelDesign/%s.xml' % (songname)

    np.random.seed(0)

    rawFile = 'd:/librosa/RhythmMaster/%s/evaluate_data_short.raw' % (songname)
    with open(rawFile, 'rb') as file:
        predicts = pickle.load(file)
        print('raw file loaded', predicts.shape)

    short = predicts[:, 1]
    
    note = TrainDataToLevelData(short, 10, 0.93)
    note = note[:,0]
    print('note count', len(note))
    SaveInstantValue(note, pathname, '_shortThre')


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

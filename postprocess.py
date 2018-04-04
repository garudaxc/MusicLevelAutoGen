import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import scipy.signal
import numpy.random
import bisect
import LevelInfo


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

def ConvertToLevelNote(notes, bpm, et):
    '''
    转换为关卡音符，增加音轨信息
    '''
    hand = 0

    barInterval = 240.0 / bpm
    barInterval *= 1000
    posInterval = barInterval / 64.0

    data = []
    for n in notes:
        time, type = n[0], n[1]

        time -= et
        bar = int(time // barInterval)
        pos = int((time % barInterval) // posInterval)

        track = np.random.randint(0, 2) + hand

        data.append((bar, pos, track))
        hand = 1 - hand

    return data

def ConvertIntermediaNoteToLevelNote(notes, bpm, et):
    barInterval = 240.0 / bpm
    barInterval *= 1000
    posInterval = barInterval / 64.0

    result = []
    for n in notes:
        time, type, value, side = n

        time *= 10
        time -= et
        bar = int(time // barInterval)
        pos = int((time % barInterval) // posInterval)
        track = np.random.randint(0, 2) + side * 2

        # long note


        # combine note


def PurifyInstanceSample(samples):
    '''
    '''    
    peakind = scipy.signal.find_peaks_cwt(samples, np.arange(1,50), min_length=3)
    #todo
    # seek to more prescise peak position

    newSample = np.zeros_like(samples)
    newSample[peakind] = samples[peakind]

    # delete very small value
    index = newSample < 0.2
    newSample[index.nonzero()] = 0

    # pick the largger value in nearby peaks
    maxDis = 3
    index = newSample.nonzero()[0]
    dis = index[1:] - index[0:-1]
    index = index[(dis<maxDis).nonzero()]
    index = index[:, np.newaxis]

    # print('joint count', len(index))
    index = np.hstack((index, index+1, index+2))
    value = newSample[np.reshape(index, -1)].reshape((-1, maxDis))
    maxValueIndex = np.argmax(value, axis=1)

    selectIndex = index[range(index.shape[0]), maxValueIndex]
    selectValue = newSample[selectIndex]
    newSample[np.reshape(index, -1)] = 0
    newSample[selectIndex] = selectValue

    return newSample


def FindCumulativeIndexInArray(a, x):
    '''
    a为一维数组，查找x在a积分中所在的位置，sigma(a'i') < x < sigma(a'i+1')
    '''
    index = (a>0).nonzero()[0]
    # print('index', index)
    values = a[index]
    # print(values)
    for i in range(1, len(values)):
        values[i] = values[i] + values[i-1]
    # print(values)

    i = bisect.bisect_left(values, x)
    # print('index %d in %d values' % (i, len(values)))
    if i == len(values):
        i = np.argmax(values)

    i = index[i]

    return i


def PickInstanceSample(samples, rate=40):
    '''
    使用指数分布生成短音符位置
    '''
    maxRange = int(scipy.stats.expon.ppf(0.999, scale=rate))
    x = np.arange(0, maxRange, 1)
    kernel = scipy.stats.expon.pdf(x, scale=rate)

    numSample = samples.shape[0]
    index = 0
    picked = []
    while index < numSample:
        length = maxRange
        if index + maxRange > numSample:
            length = numSample - index
        prob = kernel[:length] * samples[index:index+length]
        totalProb = np.sum(prob)
        if totalProb < 0.0001:
            index = index + length
            continue
        prob = prob / totalProb
        # print('total prob', totalProb, np.sum(prob))
        r = numpy.random.rand(1)[0]
        ii = FindCumulativeIndexInArray(prob, r)
        picked.append(index + ii)
        index = index + ii + 1

    picked = np.array(picked)
    print('picked', picked.shape[0])
    return picked
    

def BilateralFilter(samples, ratio = 0.7):
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
            acc += (newSample[i] - threhold) * 0.5
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



def AlignNotePosition(short, long, threhold=50):
    '''
    将挨得很近的长音符和短音符对齐到同一位置
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


def RandSide():
    # 随机生成0 1，作为side
    l = np.random.rand() // 0.5
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
    param = [('doubleSlide', 0.05, 4), ('doubleRate', 0.15, 3), ('slideRate', 0.15, 2)]
    
    longBinay = long > 0
    edge = (longBinay[1:] != longBinay[:-1]).nonzero()[0] + 1
    longNoteValue = []
    for i in range(0, len(edge), 2):
        value = np.average(long[edge[i]:edge[i+1]])
        longNoteValue.append((value, i))
    
    longNoteValue.sort(reverse=True)
    long2 = np.zeros_like(long)
    count = int(len(longNoteValue) * doubleRate)
    # 选取value最大的为双按音符
    for i in range(count):
        e = longNoteValue[i][1]
        long2[edge[e]:edge[e+1]] = long[edge[e]:edge[e+1]]

    sortIndex = np.argsort(short)    
    sortIndex = np.flip(sortIndex, axis=0)
    
    # print(short[sortIndex][0:100])

    shortMute = np.zeros_like(short)
    totalCount = np.count_nonzero(short)
    print(totalCount)

    begin = 0
    for n, ratio, id in param:
        count = int(totalCount * ratio)
        index = sortIndex[begin:begin+count]
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
            
            if long2[i] > 0:
                #double long                
                if len(s) > 0:
                    #combine note
                    cnotes = TransferCombineNote(i, end, s, SideRight)
                    notes.append((i, LevelInfo.combineNode, cnotes, SideRight))
                else:
                    #long note
                    notes.append((i, LevelInfo.longNote, end-i, SideRight))
            else:
                # short in other side
                for n in s:
                    if (shortMute[n] == DoubleSlide) or (shortMute[n] == Slide):
                        print('slide beside combine')
                        notes.append((i, LevelInfo.slideNote, 0, SideRight))
                    elif shortMute[n] == DoubleShort:
                        print('short beside combine')
                        notes.append((i, LevelInfo.shortNote, 0, SideRight))

            i = end
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

    # plt.show()



def SampleDistribute2():
    with open('d:/raw_sample.raw', 'rb') as file:
        samples = pickle.load(file)
    
    index = (samples > 0.1).nonzero()[0]
    inter = index[1:] - index[:-1]
    # inter = inter * 100

    his, b = np.histogram(inter, 100)
    his = his / samples.shape[0]
    maxx = b[-1]
    # his = his / 5000

    a = scipy.stats.expon.fit(inter)
    print(a)

    count = int(inter.shape[0] / 3)
    a0 = scipy.stats.expon.fit(inter[:count])
    a1 = scipy.stats.expon.fit(inter[count:2*count])
    a2 = scipy.stats.expon.fit(inter[2*count:])
    print('fit3', a)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(his[:50])

    x = np.linspace(0, maxx, 100)
    y = scipy.stats.expon.pdf(x, a[0], a[1])
    ax[1].plot(y[:50])    
    y = scipy.stats.expon.pdf(x, a0[0], a0[1])
    ax[1].plot(y[:50])    
    y = scipy.stats.expon.pdf(x, a1[0], a1[1])
    ax[1].plot(y[:50])    
    y = scipy.stats.expon.pdf(x, a2[0], a2[1])
    ax[1].plot(y[:50])    

    # n, bins, _ = plt.hist(inter, 200)
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




if __name__ == '__main__':


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
    

    if True:
        print('load raw file')
        with open('d:/work/evaluate_data_long.raw', 'rb') as file:
            predicts = pickle.load(file)

        pre = predicts[:, 1]
        long = BilateralFilter(pre, ratio=0.9)
       
        with open('d:/work/evaluate_data_short.raw', 'rb') as file:
            predicts = pickle.load(file)

        short = predicts[:, 1]
        short = PurifyInstanceSample(short)
        picked = PickInstanceSample(short)
        sam = np.zeros_like(short)
        sam[picked] = short[picked]
        
        long = AlignNotePosition(sam, long)

        MutateSamples(sam, long)

        
        # notes = TrainDataToLevelData(long, 10, acceptThrehold, 0)
        # print('gen notes number', len(notes))
        # notes = notes[:,0]
        # SaveInstantValue(notes, pathname, '_region')        


    # a = np.zeros(100)
    # a[range(0, 100, 5)] = 0.05

    # FindCumulativeIndexInArray(a, 1.2)
    # print(numpy.random.rand(1))
    
    # SampleDistribute2()

    # SampleDistribute(None)
    # Test()
    # Test2()

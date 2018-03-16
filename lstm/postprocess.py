import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle


def TrainDataToLevelData(data, sampleInterval, threhold, timeOffset):
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


from scipy import stats
from scipy.stats import norm
from scipy.stats import poisson


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
    maxx = b[-1]
    # his = his / 5000

    a = scipy.stats.expon.fit(inter)
    print(a)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(his[:50])

    x = np.linspace(0, maxx, 100)
    y = scipy.stats.expon.pdf(x, a[0], a[1])
    ax[1].plot(y[:50])    

    # n, bins, _ = plt.hist(inter, 200)
    plt.show()


def Test():
    x = np.arange(-10, 10, 0.01)    
    y = norm.pdf(x, scale = 2)
    samples = norm.rvs(size=1000, scale = 2)
    f = norm.fit(samples)
    
    print(f)

    # y2 = norm.cdf(x)

    plt.plot(x, y)
    # plt.plot(x, y2)
    plt.show()


def Test2():
    fig, ax = plt.subplots(1, 1)
    mu = 5
    x = np.arange(poisson.ppf(0.01, mu),
                  poisson.ppf(0.99, mu))
    ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

    plt.show()


if __name__ == '__main__':
    SampleDistribute2()

    # SampleDistribute(None)
    # Test()
    # Test2()

import numpy as np

def process(notes):
    pass



def TrainDataToLevelData(data, timeOffset, threhold = 0.8):
    notes = []
    longNote = False
    last = 0
    start = 0  
    for i in range(len(data)):                
        d = data[i]
        t = i * 10 + timeOffset

        if not isinstance(d, np.ndarray):
            if (d > threhold):
                notes.append((t, 0, 0))
            continue

        dim = len(d)

        if (d[0] > threhold):
            notes.append((t, 0, 0))

        if (dim > 1 and d[1] > threhold):
            notes.append((t, 1, 0))

        if (dim > 2 and d[2] > threhold):
            if not longNote:
                start = t
            longNote = True
            last += 10            
        else:
            if longNote:
                if last <= 10:
                    print('long note last too short')
                else:
                    notes.append((start, 2, last))
            longNote = False
            start = 0
            last = 0
    
    return notes



def pick(data):
    kernelSize = 21
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


def SaveResult(prid, target, time, pathname):
    # pathname = '/Users/xuchao/Documents/python/MusicLevelAutoGen/result.txt'
    # if os.name == 'nt':
    #     pathname = r'D:\librosa\result.log'
    with open(pathname, 'w') as file:
        n = 0
        for i in zip(prid, target):
            t = time + n * 10
            minite = t // 60000
            sec = (t % 60000) // 1000
            milli = t % 1000

            file.write('%d:%d:%d, %s , %s\n' % (minite, sec, milli, i[0], i[1]))
            n += 1

    print('saved ', pathname)

def ConvertToLevelNote(notes, bpm, et):
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

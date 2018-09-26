import os



def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
    return path

def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.m4a' % (path, song, song)
    if not os.path.exists(pathname):
        pathname = '%s%s/%s.mp3' % (path, song, song)

    return pathname

def MakeMp3Dir(song):
    path = GetSamplePath()
    pathname = '%s%s/' % (path, song)
    if not os.path.exists(pathname):
        assert False
    return pathname

def MakeLevelPathname(song, difficulty=2):
    path = GetSamplePath()
    diff = ['ez', 'nm', 'hd']
    pathname = '%s%s/%s_4k_%s.imd' % (path, song, song, diff[difficulty])
    return pathname   

def CalcBPMManually(et, t1, bpm0):
    bpm0 = float(bpm0)

    barTime = 240000.0 / bpm0
    bars = (t1 - et) / barTime
    bars2 = int(bars + 0.5)
    print('bars diff', abs(bars2-bars))

    barTime = (t1-et) / float(bars2)
    bpm = 240000.0 / barTime

    print('new bpm', bpm)


def getRootDir():
    if os.name == 'nt':
        return 'E:/work/dl/audio/proj/'
    return '../../proj/'

if __name__ == '__main__':
    # CalcBPMManually(766, 182606, 128.0)
    CalcBPMManually(762, 182605, 128.0)

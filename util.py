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
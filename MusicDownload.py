import urllib.request
import urllib.parse
from struct import *
import os

url = 'http://game.ds.qq.com/Com_SongRes/song/dontmatter/dontmatter.mp3'
url = 'http://game.ds.qq.com/Com_SongRes/song/dontmatter/dontmatter_4k_ez.imd'

'''
filename = urllib.parse.unquote(url).split('/')[-1]
print(filename)
filename = 'd:/' + filename

filename, info = urllib.request.urlretrieve(url, filename)
print(filename)
print(info)

'''


def ReadAndOffset(fmt, buffer, offset):
    r = unpack_from(fmt, buffer, offset)
    offset += calcsize(fmt)
    return r, offset



def ReadRythmMasterSongList():
    
    file = 'd:\librosa\RhythmMaster\mrock_song_client_android.bin'
    with open(file, 'rb') as file:        
        data = file.read()

    offset = 206
    songList = []
    while offset < len(data):
        r, readOffset = ReadAndOffset('64s', data, offset)
        s = r[0].strip(b'\x00')
        s = s.decode()
        songList.append(s)
        offset += 830

    print('total', len(songList))

    return songList


def DownloadRythmMasterLevels():

    songList = ReadRythmMasterSongList()
    
    for i in range(len(songList)):
        song = songList[i]
        dir = 'd:\librosa\RhythmMaster\%s' % (song)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('make dir')
        filename = '%s\%s.mp3' % (dir, song)
        if not os.path.exists(filename):
            url = 'http://game.ds.qq.com/Com_SongRes/song/%s/%s.mp3' % (song, song)
            filename, info = urllib.request.urlretrieve(url, filename)
            print('downloaded %d in %d %s' % (i , len(songList), filename))
        
        filename = '%s\%s_4k_nm.imd' % (dir, song)
        if not os.path.exists(filename):
            url = 'http://game.ds.qq.com/Com_SongRes/song/%s/%s_4k_nm.imd' % (song, song)
            try:
                filename, info = urllib.request.urlretrieve(url, filename)
            except:
                print('can not download', filename)
            else:                
                print('downloaded %d in %d %s' % (i , len(songList), filename))
                

        filename = '%s\%s_4k_hd.imd' % (dir, song)
        if not os.path.exists(filename):
            url = 'http://game.ds.qq.com/Com_SongRes/song/%s/%s_4k_hd.imd' % (song, song)
            try:
                filename, info = urllib.request.urlretrieve(url, filename)
            except:
                print('can not download', filename)
            else:
                print('downloaded %d in %d %s' % (i , len(songList), filename))



if __name__ == '__main__':
    # ReadRythmMasterSongList()
    DownloadRythmMasterLevels()
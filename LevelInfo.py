# codingd=UTF-8
import logger
from xml.etree import ElementTree  
import numpy as np
import os
import io
import sys
from struct import *

shortNote   = 0x00
slideNote   = 0x01
longNote    = 0x02
combineNode = 0x03

def parse_level_info(tree):
    #解析传统关卡中的bpm和et
    #root = tree.getroot() #level
    root = tree
    levelInfo = root.find('LevelInfo')
    node = levelInfo.find('BPM')
    bpm = float(node.text)
    # num bar node = levelInfo.find('BarAmount')
    node = levelInfo.find('EnterTime')
    et = float(node.text)

    musicInfo = root.find('MusicInfo')
    # title = musicInfo.find('Title').text
    # artist = musicInfo.find('Artist').text
    id = musicInfo.find('FilePath').text.split('_')[-1]
    id = id.split('.')[0]

    info = {'id':id, 'bpm':str(bpm), 'et':str(et)}
    print(info)
    return info


def load_levels(path):
    # 遍历目录下的xml，
    infos = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]

        if (ext != '.xml'):
            continue
        
        pathname = os.path.join(path, f)
        text=open(pathname).read()
        xmlparser = ElementTree.XMLParser(encoding='utf-8')
        #print(pathname)
        #tree = ElementTree.parse(pathname, parser=xmlparser)
        tree = ElementTree.fromstring(text)
        
        info = parse_level_info(tree)
        infos.append(info)

    return infos


def LoadMusicInfo(filename):
    '''
    读取歌曲的长度,bpm,entertime等信息
    '''
    # filename = r'd:\librosa\RhythmMaster\jilejingtu\jilejingtu.mp3'
    dir = os.path.dirname(filename) + os.path.sep
    filename = dir + 'info.txt'
    with open(filename, 'r') as file:
        value = [float(s.split('=')[1]) for s in file.readlines()]
        
        # duration, bpm, entertime
        value[0] = int(value[0] * 1000)
        value[2] = int(value[2] * 1000)
        # print(value)
        return tuple(value)

def ReadInfoInXml(filename):
    # 加载集合好的关卡信息
    if not os.path.exists(filename):
        return None
    
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.parse(filename, parser=xmlparser)
    #tree = ElementTree.fromstring(text)
    root = tree.getroot()
    infos = {}
    for l in root.iter('level'):
        attrib = l.attrib
        #print(attrib)
        id = attrib['id']
        infos[id] = [attrib['bpm'], attrib['et']]
    return infos

def CollectLevelInfo(path):   
    #加载所有传统对局的bpm和et，写到单独文件里
    infos = load_levels(path)

    filename = path + '../level-infos.xml'
    root = ElementTree.Element('LevelInfo')
    tree = ElementTree.ElementTree(root)
    for level in infos:
        node = ElementTree.Element('level', attribb=level)       
        root.append(node)

    tree.write(filename, encoding="utf-8",xml_declaration=True)

def TransfromNotes(bpm, enterTime, notes):
    barInterval = (60.0 / bpm) * 4
    posInterval = barInterval / 64

    result = [n[0] * barInterval + n[1] * posInterval + enterTime for n in notes]
    return result

def LoadIdolInfo(pathname):
    #加载心动模式关卡
    text=open(pathname, encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    root = tree
    levelInfo = root.find('LevelInfo')
    node = levelInfo.find('BPM')
    bpm = float(node.text)
    node = levelInfo.find('EnterTimeAdjust')
    et = float(node.text) / 1000.0
    
    notesNode = root.find('NoteInfo').find('Normal')
    notes = []
    for e in notesNode:
        n = e
        # 组合音符取第一个音符
        if e.tag == 'CombineNote':
            n = e[0]
        
        bar = int(n.attribb['Bar'])
        pos = int(n.attribb['Pos'])
        notes.append((bar, pos))
        # print(bar, pos)

    # unique
    notes = list(set(notes))
    notes = TransfromNotes(bpm, et, notes)
    return notes

    print(bpm, et)


def ReadAndOffset(fmt, buffer, offset):
    r = unpack_from(fmt, buffer, offset)
    offset += calcsize(fmt)
    return r, offset

def ReadRhythmMasterLevelTime(pathname):
    #读取节奏大师关卡时间
    with open(pathname, 'rb') as file:
        data = file.read()
        
    offset = 0
    r, offset = ReadAndOffset('2i', data, offset)
    totalTime = r[0]

    return totalTime


def LoadRhythmMasterLevel(pathname):
    #加载节奏大师关卡
    with open(pathname, 'rb') as file:
        data = file.read()
        
    offset = 0
    r, offset = ReadAndOffset('2i', data, offset)
    totalTime = r[0]
    numTrunk = r[1]

    r = unpack_from('4i', data, offset)
    interval = r[-1]

    # print('total time', totalTime, 'interval', interval)    

    # 跳过一段数据
    offset += calcsize('3i') * numTrunk
    r, offset = ReadAndOffset('h', data, offset)
    assert r[0] == 0x0303

    r, offset = ReadAndOffset('i', data, offset)
    numNote = r[0]
    # print('numNote', numNote)
    
    combineNoteFlag = 0x20
    cnBegin = 0x40
    cnEnd = 0x80

    notes = []

    numShort = 0
    numSlide = 0
    numLong = 0

    slideInCombine = 0

    for i in range(numNote):
        r, offset = ReadAndOffset('=BBiBi', data, offset)

        op = r[0]
        time = r[2]
        track = r[3]
        val = r[4]

        # if track != 0 and track != 1:
        #     continue
        beginTime = 0

        if (op & combineNoteFlag == combineNoteFlag):
            
            if op & cnBegin == cnBegin:
                # print("c%d begin time %d" % (i, time))
                beginTime = time
                cbNotes = []
                # notes.append((time, 3, 0))

            if op & slideNote == slideNote:
                # if time == beginTime:
                # print('combine note begin with slide time %d %d' % (time, slideNote))
                slideInCombine += 1
                cbNotes.append((time, slideNote, 0))
                # numSlide += 1

            if op & longNote == longNote:
                cbNotes.append((time, longNote, val))
                # print('c%d long time %d last %d' % (i, time, val))
                
            if op & cnEnd == cnEnd:
                notes.append((beginTime, combineNode, cbNotes))
                # print('combine note end at %d with %d elements' % (time, len(combineNode)))
                #print('c%d end time %d' % (i, time))

            continue
        
        if op & slideNote == slideNote:
            notes.append((time, slideNote, 0))
            numSlide += 1
            continue
            
        if op & longNote == longNote:
            notes.append((time, longNote, val))
            numLong += 1
            continue

        notes.append((time, 0, 0))
        numShort += 1
        #print("touch note %d %x %x" % (i, op, val))
        
    # print('begin count %d end count %d slide count %d' % (beginCount, endCount, slideInCombine))
    # print('got %d short %d slide %d long' % (numShort, numSlide, numLong))
    return notes
    

def SaveInstantValue(beats, filename, postfix=''):
    #保存时间点数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True


def SaveNote(notes, filename, postfix = ''):
    #保存音符数据
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    print('save', outname)
    with open(outname, 'w') as file:
        for v in notes:
            time, type, last = v[0], v[1], v[2]
            if type == 3:
                continue
            if last == 0:
                last = 30
            s = '%d,%d,%d\n' % (time, type * 6 + 20, last)
            file.write(s)
            if type == 1:
                s = '%d,%d,%d\n' % (time, type * 6 + 18, last)
                file.write(s)
                s = '%d,%d,%d\n' % (time, type * 6 + 22, last)
                file.write(s)
        print('done')

def LoadMidi(filename):
    with open(filename, 'rb') as file:        
        data = file.read()
        
    offset = 0
    r, offset = ReadAndOffset('4s', data, offset)
    if r[0].decode() != 'MThd':
        print(filename, 'is not midi file')
    
    r, offset = ReadAndOffset('>i3H', data, offset)
    filetype = r[1]
    numTrack = r[2]
    tpq = r[3]

    #for i in range(numTrack):
    r, offset = ReadAndOffset('4s', data, offset)
    print(r)
    assert r[0].decode() == 'MTrk'
    r, offset = ReadAndOffset('>i', data, offset)
    size = r[0]
    print(size)

def PackAndOffset(buffer, offset, fmt, *args):
    pack_into(fmt, buffer, offset, *args)
    size = calcsize(fmt)
    return size + offset

def CopyBuffer(target, offset, source):
    assert type(source) == bytes or type(source) == bytearray
    size = len(source)
    target[offset:offset+size] = source
    return offset + size

def WriteNote(buffer, deltaTime):
    pass

def MultibyteLength(n):
    numBytes = 1
    i = n // 128
    while i > 0:
        numBytes += 1
        i = i // 128
    r = bytearray(numBytes)
    for i in range(numBytes):
        r[-1-i] = ((n >> (i * 7)) & 0x7f) | 0x80
    r[-1] &= 0x7f
    print(r.hex())
    return r
    

def SaveMidi(filename, notes, bpm = 120.0):
    beatInterval = 60000 / (bpm * 960)
    noteLen = int(960 / 8)

    with open(filename, 'wb') as file:
        file.write('MThd'.encode())
        file.write(pack('>iHHH', 6, 1, 2, 960))

        buffer = bytearray(8192)
        file.write('MTrk'.encode())

        # track 0
        size = 0
        size = CopyBuffer(buffer, size, bytes.fromhex('00ff03'))
        trackname = 'meta track'
        size = PackAndOffset(buffer, size, 'b', len(trackname))
        size = CopyBuffer(buffer, size, trackname.encode())
        size = CopyBuffer(buffer, size, bytes.fromhex('00ff2f00'))
        file.write(pack('>i', size))
        file.write(buffer[:size])

        # track 1
        file.write('MTrk'.encode())
        size = 0
        # write a empty string
        size = CopyBuffer(buffer, size, bytes.fromhex('00ff03'))
        size = PackAndOffset(buffer, size, 'b', 0)
        timepoint = 0
        for note in notes:
            t = int(note[0] / beatInterval)
            deltaT = t - timepoint
            size = CopyBuffer(buffer, size, MultibyteLength(deltaT))
            # key press 9 chanel 0 note C4(48) velocity 127
            size = CopyBuffer(buffer, size, bytes.fromhex('90307f'))

            size = CopyBuffer(buffer, size, MultibyteLength(noteLen))
            # key press 9 chanel 0 note C4(48) velocity 127
            size = CopyBuffer(buffer, size, bytes.fromhex('903000'))
            timepoint = t + noteLen
        
        file.write(pack('>i', size))
        file.write(buffer[:size])
        file.flush()            

def CorrespondingTrack(track0):
    if track0 < 2:
        track1 = 1-track0
    elif track0 == 2:
        track1 = 3
    elif track0 == 3:
        track1 = 2
    return track1

def ConvertNoteToXml(note):
    trackName = ('Left2', 'Left1', 'Right1', 'Right2')
    bar, pos, type, value, track = note

    e = ElementTree.Element('Note')
    e.attrib['Bar'] = str(bar+1)
    e.attrib['Pos'] = str(pos*2)

    if type == shortNote:
        e.attrib['from_track'] = trackName[track]
        e.attrib['target_track'] = trackName[track]
        e.attrib['note_type'] = 'short'
    elif type == slideNote:
        e.attrib['target_track'] = trackName[track]
        target = CorrespondingTrack(track)
        e.attrib['end_track'] = trackName[target]
        e.attrib['note_type'] = 'slip'
    elif type == longNote:
        e.attrib['from_track'] = trackName[track]
        e.attrib['target_track'] = trackName[track]
        e.attrib['note_type'] = 'long'
        endBar, endPos = value
        e.attrib['EndBar'] = str(endBar+1)
        e.attrib['EndPos'] = str(endPos*2)
    else:
        assert False

    return e

def ConvertTimeBar(note, barInterval, posInterval, et):
    time, type, value, track = note
    time -= et

    bar = int(time // barInterval)
    pos = int((time % barInterval) // posInterval)

    if type == longNote:
        endTime = time + value
        endBar = int(endTime//barInterval)
        endPos = int((endTime % barInterval) // posInterval)
        value = (endBar, endPos)

    return (bar, pos, type, value, track)

def MakeActionListNode(startBar, danceLen, seqLen):
    e = ElementTree.Element('ActionList')
    e.attrib['start_bar'] = str(startBar)
    e.attrib['dance_len'] = str(danceLen)
    e.attrib['seq_len'] = str(seqLen)
    e.attrib['level'] = '2'
    e.attrib['type'] = ''

    return e

def GenerateIdolLevel(filename, notes, bpm, et, musicTime):
    '''
    保存到心动模式关卡文件
    '''
    barInterval = 240.0 / bpm
    barInterval *= 1000
    posInterval = barInterval / 32.0

    newNotes = []
    for n in notes:
        type = n[1]
        r = ConvertTimeBar(n, barInterval, posInterval, et)
        if type == combineNode:
            value = n[2]
            l = []
            for subNote in value:
                s = ConvertTimeBar(subNote, barInterval, posInterval, et)
                l.append(s)
            r = (r[0], r[1], r[2], l, r[4])
             
        newNotes.append(r)
    notes = newNotes

    text=open(r'D:\librosa\idol_template.xml', encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    lastNote = notes[-1]
    lastBar = lastNote[0]
    totalBar  = lastBar + 1

    # 删除前面4小节和最后4小节的音符
    enterBar = 4
    notes = [note for note in notes if note[0] > (enterBar-1) and note[0] < lastBar - 3]
    print('number of notes', len(notes))

    root = tree
    levelInfo = root.find('LevelInfo')
    node = levelInfo.find('BPM')
    node.text = str(bpm)
    node = levelInfo.find('EnterTimeAdjust')
    node.text = str(et)    
    node = levelInfo.find('LevelTime')
    node.text = str(musicTime)
    node = levelInfo.find('BarAmount')
    node.text = str(totalBar)
    node = levelInfo.find('BeginBarLen')
    node.text = str(enterBar)

    
    musicName = os.path.split(filename)[1]
    musicName = os.path.splitext(musicName)[0]
    MusicInfo = root.find('MusicInfo')
    node = MusicInfo.find('Title')
    node.text = str(musicName)
    node = MusicInfo.find('FilePath')
    node.text = str('audio/bgm/'+musicName)
    
    SectionSeq = root.find('SectionSeq')
    for node in SectionSeq:
        if node.attrib['type'] == 'note':
            node.attrib['endbar'] = str(totalBar - 4)
        if node.attrib['type'] == 'showtime':
            node.attrib['startbar'] = str(totalBar - 3)
            node.attrib['endbar'] = str(totalBar)

    notesNode = root.find('NoteInfo').find('Normal')
    notesNode.clear()
    for note in notes:
        type = note[2]
        
        if type == combineNode:
            e = ElementTree.Element('CombineNote')
            value = note[3]
            for subNote in value:
                s = ConvertNoteToXml(subNote)
                e.append(s)
        else:
            e = ConvertNoteToXml(note)

        notesNode.append(e)

    numseq = (totalBar-enterBar-4) // 2
    print('numseq', totalBar, numseq * 2)
    ActionSeq = root.find('ActionSeq')
    seq = enterBar+1
    if totalBar % 2 == 1:
        e = MakeActionListNode(seq, 1, 1)
        ActionSeq.append(e)
        seq += 1
    
    for i in range(numseq):
        e = MakeActionListNode(seq, 2, 2)
        ActionSeq.append(e)
        seq += 2
        
    e = MakeActionListNode(seq, 4, 4)
    ActionSeq.append(e)

    
    camera = root.find('CameraSeq').find('Camera')
    camera.attrib['end_bar'] = str(totalBar)
       
    DancerSort = root.find('DancerSort')
    DancerSort.clear()
    for i in range(15, totalBar-2, 2):
        e = ElementTree.Element('Bar')
        e.text = str(i)
        DancerSort.append(e)
    
    t = ElementTree.ElementTree(root)
    t.write(filename, encoding="utf-8",xml_declaration=True)
    print('done write file ' + filename)

def SaveSamplesToRegionFile(samples, filename, postfix=''):
    '''
    每秒100帧的采样数据转换为连续区间    
    '''
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    
    border =  np.nonzero(samples[:-1] != samples[1:])[0]

    region = []
    for b, e in zip(border[0::2], border[1::2]):
        # print(samples[b])
        if samples[b] == False:
            assert samples[e] == True
            begin = (b + 1) * 0.01
            length = (e - b) * 0.01
            region.append((begin, length))
    
    # print(region)
    with open(outname, 'w') as file:
        for r in region:
            line = '%f,1,%f\n' % r
            file.write(line)
        
        print('file', outname, 'done')


def GetSamplePath():
    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
    return path


def MakeMp3Pathname(song):
    path = GetSamplePath()
    pathname = '%s%s/%s.mp3' % (path, song, song)
    return pathname

def MakeLevelPathname(song, difficulty=2):
    path = GetSamplePath()
    diff = ['ez', 'nm', 'hd']
    pathname = '%s%s/%s_4k_%s.imd' % (path, song, song, diff[difficulty])
    return pathname   

def ProcessRythmMasterMusicInfo():
    # 读取关卡描述文件，将music info保存到目录下
    path = 'd:/librosa/RhythmMaster'
    files = os.listdir(path)
    dirs = []
    for f in files:
        if os.path.isdir(path + '/' + f):
            dirs.append(f)

    
    text=open(r'd:\librosa\RhythmMaster\mrock_song_client_android.xml', encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    nodes = tree.findall('SongConfig_Client')

    collects = {}    
    for node in nodes:
        name = node.find('m_szPath')
        name = name.text

        if name not in dirs:
            continue
        
        levelFileName = MakeLevelPathname(name)
        if not os.path.exists(levelFileName):
            continue

        level = LoadRhythmMasterLevel(levelFileName)

        duration = node.find('m_iGameTime')
        duration = float(duration.text)
        bpm = node.find('m_szBPM')
        bpm = float(bpm.text)

        t0, _, _ = level[0]        
        step = 60000.0 / (bpm * 8)
        remainder = t0 % step
        quotient = t0 // step

        r = (step - remainder) % step
        if abs(r) > 2:
            print(name, 'q', quotient, 'r', remainder, 'step', step, 'res', r)
        
        filename = path + '/' + name + '/' + 'info.txt'
        with open(filename, 'w') as file:
            file.write('duration=%f\nbpm=%f\net=%f' % (duration, bpm, 0))

            # to do
        # print(name)
        continue



        print(name, duration, bpm)

    # print(len(nodes))





if __name__ == '__main__':


    # ProcessRythmMasterMusicInfo()

    #import madmon_test
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

    if os.name == 'nt':
        path = r'D:/ab/QQX5_Mainland/exe/resources/level/game_level/'
    else :
        path = r'/Users/xuchao/Documents/python/MusicLevelAutoGen/'

    #LoadIdolInfo(path + 'idol_100002.xml')
    if False:
        pathname = r'D:\librosa\炫舞自动关卡生成\level\idol_100052.xml'
        notes = LoadIdolInfo(pathname)
        madmon_test.save_file(notes, pathname, '_notes')


    if False:    
        path = '/Users/xuchao/Documents/rhythmMaster/'
        if os.name == 'nt':
            path = 'D:/librosa/RhythmMaster/'

        songName = '4minuteshm'
        songName = '1987wbzhyjn'


        songList = ['inthegame', 'remains', 'ilikeit', 'haodan', 'ai', 
'1987wbzhyjn', 'tictic', 'aiqingkele', 'feiyuedexin', 'hewojiaowangba', 'myall', 'unreal', 'faraway',
'fengniao', 'wangfei', 'wodemuyang', 'mrx', 'rageyourdream', 'redhot', 'yilububian', 'yiqiyaobai', 'yiranaini', 'yiyejingxi']

        for songName in songList:
            pathname = '%s%s/%s_4k_hd.imd' % (path, songName, songName)
            notes = LoadRhythmMasterLevel(pathname)


        # SaveNote(notes, pathname, '_notes')


    # SaveInstantValue(notes, filename, '_time')
    # filename = r'd:/miditest_out.mid'
    # # LoadMidi(filename)

    # SaveMidi(filename, None)

    # a = input()
    # print(a)
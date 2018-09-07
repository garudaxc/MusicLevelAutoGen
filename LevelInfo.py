# codingd=UTF-8
import logger
from xml.etree import ElementTree  
import numpy as np
import os
import io
import sys
from struct import *
import util
rootDir = util.getRootDir()

shortNote   = 0x00
slideNote   = 0x01
longNote    = 0x02
combineNode = 0x03

posCountPerBeat = 8
beatPerBar = 4
posCountPerBar = posCountPerBeat * beatPerBar

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

    result = [(n[0] - 1) * barInterval + n[1] * posInterval + enterTime for n in notes]
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
        
        bar = int(n.attrib['Bar'])
        pos = int(n.attrib['Pos'])
        notes.append((bar, pos))
        # print(bar, pos)

    # unique
    notes = list(set(notes))
    notes = TransfromNotes(bpm, et, notes)
    return notes

    print(bpm, et)

def LoadBpmET(pathname):
    text=open(pathname, encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    root = tree
    levelInfo = root.find('LevelInfo')
    node = levelInfo.find('BPM')
    bpm = float(node.text)
    node = levelInfo.find('EnterTimeAdjust')
    et = float(node.text) / 1000.0
    return bpm, et

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

def ReadVariableLengthQuantity(buffer, offset):
    value = 0
    while True:
        # 按byte读取值，最高位不为1，需要读取下一个byte
        r, offset = ReadAndOffset('>B', buffer, offset)
        byteValue = r[0]
        value = (value << 7) | (byteValue & 0x7f)
        if (byteValue & 0x80) == 0:
            break
        
    return value, offset

def LoadMidi(filename, exInfo = None):
    with open(filename, 'rb') as file:        
        data = file.read()
    
    def TickToSecond(tick, division, microsecondsPerQuarterNoteArr):
        count = len(microsecondsPerQuarterNoteArr)
        if count <= 0:
            return 0

        usec = 0
        if count == 1:
            usec = tick / division * microsecondsPerQuarterNoteArr[0][1]
        else:
            idx = 1
            curTick = 0
            while idx < count:
                tickStart, interval = microsecondsPerQuarterNoteArr[idx]
                lastTickStart, lastInterval = microsecondsPerQuarterNoteArr[idx - 1]
                if tickStart < tick:
                    usec += (tickStart - lastTickStart) / division * lastInterval
                    curTick = tickStart
                else:
                    usec += (tick - lastTickStart) / division * lastInterval
                    curTick = tick
                    break
                idx += 1

            if tick > curTick:
                usec += (tick - curTick) / division * microsecondsPerQuarterNoteArr[-1][1]

        second = int(round(usec / 1000.0)) / 1000.0
        return second

    def TickToBarInfo(tick, prefix, division, microsecondsPerQuarterNoteArr, tickFunc = TickToSecond, printLog = True):
        beatInterval = division
        barNum = int((tick / beatInterval) / 4)
        beatNum = int((tick / beatInterval) - barNum * 4)
        subCount = tick - barNum * 4 * beatInterval - beatNum * beatInterval
        val = subCount / (beatInterval / (4 * 120))
        subVal1 = int(val / 120)
        subVal2 = (val - subVal1 * 120)
        second = tickFunc(tick, division, microsecondsPerQuarterNoteArr)
        minute = int(second // 60)
        second = second - minute * 60
        if printLog:
            print(prefix, barNum + 1, beatNum + 1, subVal1 + 1, subVal2, '  time: ', minute, ':', second)
        return barNum, beatNum, subVal1, subVal2

    def ProcessStr(text):
        emptyText = ''
        if len(text) <= 0:
            return emptyText

        text = text.replace(' ', '')
        text = text.replace('\r', '')
        text = text.replace('\n', '')
        text = text.replace('\x00', '')
        if len(text) <= 0:
            return emptyText

        def CutPair(text, charA, charB):
            while True:
                idxA = text.find(charA)
                if idxA < 0:
                    break

                idxB = text.find(charB)
                if idxB < 0:
                    break

                text = text[idxB+1:]

            return text

        text = CutPair(text, '<', '>')
        text = CutPair(text, '[', ']')

        return text
        

    def TryDecodeStr(strData):
        tryArr = ['gbk', 'ascii', 'utf-8']
        for encoding in tryArr:
            try:
                text = strData.decode(encoding)
                return ProcessStr(text)
            except:
                pass

        return '(err0)'

    def DecodeStr(strData):
        text = TryDecodeStr(strData)
        try:
            data = text.encode('gbk')
            return text
        except:
            pass

        return '(err1)'

    offset = 0
    r, offset = ReadAndOffset('4s', data, offset)
    if r[0].decode() != 'MThd':
        print(filename, 'is not midi file')
    
    r, offset = ReadAndOffset('>I3H', data, offset)
    filetype = r[1]
    numTrack = r[2]
    # x51的midi似乎只用了最高位为0的division情况
    division = r[3]
    print('track count:', numTrack)
    midiNotes = []
    microsecondsPerQuarterNoteArr = []
    for i in range(numTrack):
        print('parse track idx:', i)
        # chunk type: MTrk string
        r, offset = ReadAndOffset('4s', data, offset)
        print(r)
        assert r[0].decode() == 'MTrk'

        # length
        r, offset = ReadAndOffset('>I', data, offset)
        trackLength = r[0]
        print('trackLength', trackLength)

        if trackLength <= 0:
            continue

        # MTrk event
        eventsStart = offset
        isTrackEnd = False
        notePitch = 0
        noteStart = 0
        noteEnd = 0
        curTicks = 0
        wordCount = 0
        lastEventType = -1
        curWord = ''
        while (offset < eventsStart + trackLength) or (not isTrackEnd):
            deltaTime, offset = ReadVariableLengthQuantity(data, offset)
            curTicks += deltaTime 
            r, offset = ReadAndOffset('>B', data, offset)
            eventType = r[0]
            # MIDI event
            if eventType < 0x80:
                if lastEventType < 0:
                    print('first event type less than 0x80, failed')
                    return []
                else:
                    # print('event type less than 0x80, use last type.', lastEventType)
                    eventType = lastEventType
                    offset = offset - 1
            
            lastEventType = eventType
            if eventType == 0xF0 or eventType == 0xF7:
                # sysex event
                eventLength, offset = ReadVariableLengthQuantity(data, offset)
                offset = offset + eventLength
            elif eventType == 0xFF:
                # meta-event
                r, offset = ReadAndOffset('>B', data, offset)
                metaType = r[0]
                metaLength, offset = ReadVariableLengthQuantity(data, offset)
                metaOffset = offset
                if metaType == 0x2F:
                    isTrackEnd = True
                elif metaType == 0x51:
                    r, suboffset = ReadAndOffset('>3B', data, offset)
                    microsecondsPerQuarterNote = (r[0] << 16) | (r[1] << 8) | r[2]
                    microsecondsPerQuarterNoteArr.append([curTicks, microsecondsPerQuarterNote])
                elif metaType == 0x05:
                    wordCount += 1
                    if metaLength > 0:                        
                        strFmt = str(metaLength) + 's'
                        r, suboffset = ReadAndOffset(strFmt, data, offset)
                        text = DecodeStr(r[0])
                        if len(text) > 0:
                            curWord = text
                    # TickToBarInfo(curTicks, 'word', division, microsecondsPerQuarterNoteArr)

                offset += metaLength
            else:
                mainType = eventType & 0xF0
                if mainType == 0x80 or mainType == 0x90 or mainType == 0xA0 or mainType == 0xB0 or mainType == 0xE0 or eventType == 0xF2:
                    r, subOffset = ReadAndOffset('>B', data, offset)
                    val1 = r[0]
                    r, subOffset = ReadAndOffset('>B', data, subOffset)
                    val2 = r[0]
                    if mainType == 0x80 or (mainType == 0x90 and val2 == 0):
                        # Note Off event
                        noteEnd = curTicks
                        if (noteEnd < noteStart):
                            print('error: noteEnd < noteStart', noteEnd, noteStart)
                            continue

                        lenMidiNotes = len(midiNotes)
                        if lenMidiNotes > 0 and midiNotes[lenMidiNotes - 1][3] == wordCount:
                            # 一个词可能唱多个音符，如果是同一个词，更新长度
                            lastNote = midiNotes[lenMidiNotes - 1]
                            lastNote[1] = noteEnd
                        else:
                            midiNotes.append([noteStart, noteEnd, notePitch, wordCount, curWord])
                    elif mainType == 0x90:
                        # Note On event
                        noteStart = curTicks
                        notePitch = val1
                    offset += 2
                elif mainType == 0xC0 or mainType == 0xD0 or eventType == 0xF3:
                    offset += 1
                elif eventType == 0xF0:
                    while True:
                        r, offset = ReadAndOffset('>B', data, offset)
                        if r[0] == 0xF7:
                            break
                else:
                    pass
            
        offset = eventsStart + trackLength

    print('tempo change count', len(microsecondsPerQuarterNoteArr))
    wordArr = []
    for note in midiNotes:
        note[0] = TickToSecond(note[0], division, microsecondsPerQuarterNoteArr)
        note[1] = TickToSecond(note[1], division, microsecondsPerQuarterNoteArr)
        note[1] = note[1] - note[0]
        wordArr.append(note[4])

    # print(wordArr)
    midiNotes = np.array(midiNotes)
    midiNotes = midiNotes[:, 0:3]
    midiNotes = midiNotes.astype(float)
    if exInfo is not None:
        exInfo.append(microsecondsPerQuarterNoteArr)
        exInfo.append(wordArr)
    # singingStartTime = midiNotes[:, 0]
    # SaveInstantValue(singingStartTime, filename, 'singingstart')
    return midiNotes    

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

def CalcNotePos(bar, pos):
    return bar * posCountPerBar + pos

def AdjustBarAndPos(bar, pos, maxNotePerBeat):
    adjustBar = round(bar)
    minPosInterval = posCountPerBeat // maxNotePerBeat
    adjustPos = round(pos / minPosInterval) * minPosInterval
    maxPos = posCountPerBar
    if adjustPos >= maxPos:
        adjustBar = adjustBar + 1
        adjustPos = 0

    return adjustBar, adjustPos

def AlignShortNote(note, maxNotePerBeat):
    bar, pos, type, value, track = note
    adjustBar, adjustPos = AdjustBarAndPos(bar, pos, maxNotePerBeat)
    newNote = adjustBar, adjustPos, type, value, track
    start = CalcNotePos(adjustBar, adjustPos)
    return newNote, [(track, start, start)]

def AlignSlideNote(note, maxNotePerBeat):
    bar, pos, type, value, track = note
    adjustBar, adjustPos = AdjustBarAndPos(bar, pos, maxNotePerBeat)
    newNote = adjustBar, adjustPos, type, value, track
    start = CalcNotePos(adjustBar, adjustPos)

    trackInfo = []
    target = CorrespondingTrack(track)
    step = 1
    if target < track:
        step = -1
    for i in range(track, target, step):
        trackInfo.append((i, start, start))

    return newNote, trackInfo

def AlignLongNote(note, maxNotePerBeat):
    bar, pos, type, value, track = note
    endBar, endPos = value
    adjustStartBar, adjustStartPos = AdjustBarAndPos(bar, pos, maxNotePerBeat)
    adjustEndBar, adjustEndPos = AdjustBarAndPos(endBar, endPos, maxNotePerBeat)
    start = CalcNotePos(adjustStartBar, adjustStartPos)
    end = CalcNotePos(adjustEndBar, adjustEndPos)
    if start < end:
        newNote = (adjustStartBar, adjustStartPos, type, (adjustEndBar, adjustEndPos), track)
        trackInfo = [(track, start, end)]
    else:
        # 长音符如果调整后首位相接了，用短音符代替
        print('start == end, adjust long to short. origin pos:', bar, pos, endBar, endPos)
        newNote = (adjustStartBar, adjustStartPos, shortNote, 0, track)
        trackInfo = [(track, start, start)]
    return newNote, trackInfo

def AlignCombineNote(note, maxNotePerBeat):
    bar, pos, type, value, track = note
    isFirst = True
    isValid = True
    lastPosIdx = 0
    adjustSubNotes = []
    trackInfos = []
    combineStart = 0
    combineEnd = 0
    trackInfoDic = {}
    for subNote in value:
        subType = subNote[2]
        if subType == longNote:
            newNote, trackInfo = AlignLongNote(subNote, maxNotePerBeat)
            if newNote[2] != longNote:
                print('combine note adjust long note not vaild')
                isValid = False
                break
        elif subType == slideNote:
            newNote, trackInfo = AlignSlideNote(subNote, maxNotePerBeat)
        else:
            print('combine note contain invaild note type: ', subType)
            isValid = False
            break

        posIdx = CalcNotePos(newNote[0], newNote[1])
        adjustSubNotes.append(newNote)
        for info in trackInfo:
            trackInfoDic[trackInfo[0]] = True
        if isFirst:
            lastPosIdx = posIdx
            isFirst = False
            combineStart = trackInfo[0][0]
            combineEnd = trackInfo[0][1]
            continue
        if lastPosIdx < posIdx:
            lastPosIdx = posIdx
            combineEnd = trackInfo[0][1]
            continue
        print('combine node adjust not valid.')
        isValid = False
        break

    newNote = None
    if isValid and len(adjustSubNotes) > 0:
        firstNote = adjustSubNotes[0]
        newNote = (firstNote[0], firstNote[1], type, adjustSubNotes, track)
        for key in trackInfoDic:
            trackInfo.append((key, combineStart, combineEnd))

    return newNote, trackInfo

def updateTrackEndDic(trackEndDic, trackInfos):
    for trackInfo in trackInfos:
        track, start, end = trackInfo
        if track not in trackEndDic.keys():
            trackEndDic[track] = end
        elif trackEndDic[track] < end:
            trackEndDic[track] = end

def isTrackInfoValid(trackEndDic, trackInfos):
    isValid = True
    for trackInfo in trackInfos:
        track, start, end = trackInfo
        if track not in trackEndDic.keys():
            continue
        if trackEndDic[track] >= start:
            isValid = False
            break

    return isValid


def AlignNotesWithBeat(notes, maxNotePerBeat):
    '''
    maxNotePerBeat 对齐音符，1为整拍，2为半拍，4为1/4拍...
    '''
    alignedNotes = []
    trackEndDic = {}
    for note in notes:
        type = note[2]
        if type == combineNode:
            newNote, info = AlignCombineNote(note, maxNotePerBeat)
        elif type == longNote:
            newNote, info = AlignLongNote(note, maxNotePerBeat)
        elif type == slideNote:
            newNote, info = AlignSlideNote(note, maxNotePerBeat)
        else:
            newNote, info = AlignShortNote(note, maxNotePerBeat)

        if newNote and isTrackInfoValid(trackEndDic, info):
            updateTrackEndDic(trackEndDic, info)
            alignedNotes.append(newNote)
        else:
            if newNote:
                print('note conflict with other after align. Discard this note. type:', type)
            else:
                print('new note is none')

    return alignedNotes


def GetTrackPosInfo(notes, trackCount):
    trackPosDic = []
    for i in range(0, trackCount):
        trackPosDic.append({})

    for note in notes:
        bar, pos, type, value, track = note
        notePos = CalcNotePos(bar, pos)
        if type == shortNote:
            trackPosDic[track][notePos] = True
        elif type == slideNote:
            trackPosDic[track][notePos] = True
            trackPosDic[CorrespondingTrack(track)][notePos] = True
        elif type == longNote:
            endBar, endPos = value
            endNotePos = CalcNotePos(endBar, endPos)
            dic = trackPosDic[track]
            # 先把end的位置也算做是被占用了,要不不好处理同侧的轨道在end处出现音符的情况
            for i in range(notePos, endNotePos + 1):
                dic[i] = True
        else:
            # 组合音符
            pass
    return trackPosDic
    

def CheckNotes(notes, noteEndBar):
    '''
    最后检查音符，处理对齐音符等阶段可能出现的同一个位置多个轨道有音符等情况
    '''
    samePosNoteCount = 2
    trackCount = 4
    posDic = {}

    for note in notes:
        bar, pos, type, value, track = note
        notePos = CalcNotePos(bar, pos)
        if notePos in posDic:
            posDic[notePos].append(note)
        else:
            posDic[notePos] = [note]

    trackPosInfo = GetTrackPosInfo(notes, trackCount)

    for notePos in posDic:
        arr = posDic[notePos]
        noteArr = arr

        if len(arr) > samePosNoteCount:
            shortNotes = []
            otherNotes = []
            for note in arr:
                bar, pos, type, value, track = note
                if type == shortNote:
                    shortNotes.append(note)
                else:
                    otherNotes.append(note)
            otherNotesCount = len(otherNotes)
            if otherNotesCount >= samePosNoteCount:
                noteArr = otherNotes[0:samePosNoteCount]
            else:
                count = samePosNoteCount - otherNotesCount
                noteArr = shortNotes[0:count] + otherNotes

        if len(noteArr) == samePosNoteCount:
            noteA = noteArr[0]
            noteB = noteArr[1]
            trackA = noteA[4]
            trackB = noteB[4]
            if int(trackA / 2) == int(trackB / 2):
                typeA = noteA[3]
                typeB = noteB[3]
                if typeA != shortNote and typeB != shortNote:
                    print('[CheckNotes] two long note in same side. remove one')
                    noteArr = [noteA]
                elif typeA == shortNote:
                    trackA = trackCount - 1 - trackB
                    pos = CalcNotePos(noteA[0], noteA[1])
                    if pos in trackPosInfo[trackA]:
                        print('[CheckNotes] short note conflict with other, remove. a')
                        noteArr = [noteB]
                    else:
                        trackPosInfo[trackA][pos] = True
                        noteArr[0] = (noteA[0], noteA[1], noteA[2], noteA[3], trackA)
                else:
                    trackB = trackCount - 1 - trackA
                    pos = CalcNotePos(noteB[0], noteB[1])
                    if pos in trackPosInfo[trackB]:
                        print('[CheckNotes] short note conflict with other, remove. b')
                        noteArr = [noteA]
                    else:
                        trackPosInfo[trackB][pos] = True
                        noteArr[1] = (noteB[0], noteB[1], noteB[2], noteB[3], trackB)
                
        posDic[notePos] = noteArr

    tempNotes = []
    for note in notes:
        bar, pos, type, value, track = note
        notePos = CalcNotePos(bar, pos)
        noteArr = posDic[notePos]
        for tempNote in noteArr:
            tempNotes.append(tempNote)

        posDic[notePos] = []

    trackPosInfo = GetTrackPosInfo(tempNotes, trackCount)
    validNotes = []
    for note in tempNotes:
        bar, pos, type, value, track = note
        if type != shortNote:
            if type == longNote:
                endBar, endPos = value
                if endBar >= noteEndBar:
                    endBar = noteEndBar
                    endPos = 0
                    startNotePos = CalcNotePos(bar, pos)
                    endNotePos = CalcNotePos(endBar, endPos)
                    if endNotePos - startNotePos < 8:
                        print('skip long note too short after clip out of endbar')
                        continue
                    note = (bar, pos, type, (endBar, endPos), track)
                    print('clip long note out of endbar')
            validNotes.append(note)
            continue

        curNote = note
        notePos = CalcNotePos(bar, pos)
        correspondingTrack = CorrespondingTrack(track)
        if notePos in trackPosInfo[correspondingTrack]:
            targetTrack = trackCount - 1 - track
            targetCorrespondingTrack = CorrespondingTrack(track)
            if notePos in trackPosInfo[targetTrack] or notePos in trackPosInfo[targetCorrespondingTrack]:
                print('[CheckNotes] short note conflict with other, remove. c')
                continue

            curNote = (note[0], note[1], note[2], note[3], targetTrack)            
        
        validNotes.append(curNote)

    return validNotes


def GenerateIdolLevel(filename, notes, bpm, et, musicTime, templateFilePath = None):
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

    if templateFilePath is None:
        templateFilePath = rootDir + 'data/idol_template.xml'
    text=open(templateFilePath, encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    lastNote = notes[-1]
    lastBar = lastNote[0]
    totalBar  = lastBar + 1

    # 删除前面4小节和最后4小节的音符
    enterBar = 4
    notes = [note for note in notes if note[0] > (enterBar-1) and note[0] < lastBar - 3]
    print('number of notes', len(notes))

    notes = AlignNotesWithBeat(notes, 2)
    notes = CheckNotes(notes, lastBar - 3)

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

def GenerateIdolLevelForTangoDebug(filename, notes, bpm, et, musicTime):
    '''
    炫舞一tango调试用
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

    text=open(rootDir + 'data/idol_template.xml', encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    lastNote = notes[-1]
    lastBar = lastNote[0]
    totalBar  = lastBar + 1

    # 删除前面4小节和最后4小节的音符
    enterBar = 0
    # notes = [note for note in notes if note[0] > (enterBar-1) and note[0] < lastBar - 3]
    # print('number of notes', len(notes))

    notes = AlignNotesWithBeat(notes, 4)
    # notes = CheckNotes(notes)

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
    
    # SectionSeq = root.find('SectionSeq')
    # for node in SectionSeq:
    #     if node.attrib['type'] == 'note':
    #         node.attrib['endbar'] = str(totalBar - 4)
    #     if node.attrib['type'] == 'showtime':
    #         node.attrib['startbar'] = str(totalBar - 3)
    #         node.attrib['endbar'] = str(totalBar)

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

    # numseq = (totalBar-enterBar-4) // 2
    # print('numseq', totalBar, numseq * 2)
    # ActionSeq = root.find('ActionSeq')
    # seq = enterBar+1
    # if totalBar % 2 == 1:
    #     e = MakeActionListNode(seq, 1, 1)
    #     ActionSeq.append(e)
    #     seq += 1
    
    # for i in range(numseq):
    #     e = MakeActionListNode(seq, 2, 2)
    #     ActionSeq.append(e)
    #     seq += 2
        
    # e = MakeActionListNode(seq, 4, 4)
    # ActionSeq.append(e)

    
    # camera = root.find('CameraSeq').find('Camera')
    # camera.attrib['end_bar'] = str(totalBar)
       
    # DancerSort = root.find('DancerSort')
    # DancerSort.clear()
    # for i in range(15, totalBar-2, 2):
    #     e = ElementTree.Element('Bar')
    #     e.text = str(i)
    #     DancerSort.append(e)
    
    t = ElementTree.ElementTree(root)
    t.write(filename, encoding="utf-8",xml_declaration=True)
    print('done write file ' + filename)

def GenerateIdolLevelForMidiLabel(filename, notes, bpm, et, musicTime, posPerBeat = 8):
    barInterval = 240.0 / bpm
    barInterval *= 1000
    posInterval = barInterval / (4 * posPerBeat)

    newNotes = []
    for n in notes:
        if len(n) <= 4:
            type = n[1]
            r = ConvertTimeBar(n, barInterval, posInterval, et)
        else:
            type = n[2]
            r = n
        if type == combineNode:
            value = n[2]
            l = []
            for subNote in value:
                s = ConvertTimeBar(subNote, barInterval, posInterval, et)
                l.append(s)
            r = (r[0], r[1], r[2], l, r[4])
             
        newNotes.append(r)
    notes = newNotes

    text=open(rootDir + 'data/idol_template.xml', encoding='utf-8').read()
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.fromstring(text)

    lastNote = notes[-1]
    lastBar = lastNote[0]
    totalBar  = lastBar + 4

    # 删除前面4小节和最后4小节的音符
    enterBar = 0
    # notes = [note for note in notes if note[0] > (enterBar-1) and note[0] < lastBar - 3]
    # print('number of notes', len(notes))

    # notes = AlignNotesWithBeat(notes, 4)
    # notes = CheckNotes(notes)

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


    def ConvertNoteToXmlEx(note):
        trackName = ('Left2', 'Left1', 'Right1', 'Right2')
        bar, pos, type, value, track = note

        e = ElementTree.Element('Note')
        e.attrib['Bar'] = str(bar+1)
        e.attrib['Pos'] = str(pos)

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
            e.attrib['EndPos'] = str(endPos)
        else:
            assert False

        return e

    convertFunc = ConvertNoteToXml
    if posPerBeat == 16:
        convertFunc = ConvertNoteToXmlEx

    notesNode = root.find('NoteInfo').find('Normal')
    notesNode.clear()
    for note in notes:
        type = note[2]
        
        if type == combineNode:
            e = ElementTree.Element('CombineNote')
            value = note[3]
            for subNote in value:
                s = convertFunc(subNote)
                e.append(s)
        else:
            e = convertFunc(note)

        notesNode.append(e)
    
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
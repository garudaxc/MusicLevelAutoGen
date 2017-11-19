# coding=UTF-8
import logger
from xml.etree import ElementTree  
import numpy as np
import os
import io
import sys
from struct import *

class tradition_mode_autogen:

    def __init__(self, file):
        self.filename = file


    def calc_beats(self):
        return

def parse_level_info(tree):
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

def save_level_info(filename, infos):

    root = ElementTree.Element('LevelInfo')
    tree = ElementTree.ElementTree(root)
    for level in infos:
        node = ElementTree.Element('level', attrib=level)       
        root.append(node)

    tree.write(filename, encoding="utf-8",xml_declaration=True)

def load_levelinfo_file(filename):
    if not os.path.exists(filename):
        return None
    
    xmlparser = ElementTree.XMLParser(encoding='utf-8')
    tree = ElementTree.parse(filename, parser=xmlparser)
    #tree = ElementTree.fromstring(text)
    root = tree.getroot()
    infos = {}
    for l in root.iter('level'):
        attri = l.attrib
        #print(attri)
        id = attri['id']
        infos[id] = [attri['bpm'], attri['et']]
    return infos

def test1(path):   

    infos = load_levels(path)
    save_level_info(path + '../level-infos.xml', infos)

def TransfromNotes(bpm, enterTime, notes):
    barInterval = (60.0 / bpm) * 4
    posInterval = barInterval / 64

    result = [n[0] * barInterval + n[1] * posInterval + enterTime for n in notes]
    return result

def LoadIdolInfo(pathname):
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


def ReadAndOffset(fmt, buffer, offset):
    r = unpack_from(fmt, buffer, offset)
    offset += calcsize(fmt)
    return r, offset


def LoadRhythmMasterLevel(pathname):
    with open(pathname, 'rb') as file:
        data = file.read()
        
    offset = 0
    r, offset = ReadAndOffset('2i', data, offset)
    totalTime = r[0]
    numTrunk = r[1]
    print('total time', totalTime, 'trunk', numTrunk)

    r = unpack_from('4i', data, offset)
    interval = r[-1]
    print('interval', interval)

    # 跳过一段数据
    offset += calcsize('3i') * numTrunk
    r, offset = ReadAndOffset('h', data, offset)
    assert r[0] == 0x0303

    r, offset = ReadAndOffset('i', data, offset)
    numNote = r[0]
    print('numNote', numNote)
    
    combineNoteFlag = 0x20
    cnBegin = 0x40
    cnEnd = 0x80

    slideNote = 0x01
    longNote = 0x02

    notes = []
    for i in range(numNote):
        r, offset = ReadAndOffset('=BBiBi', data, offset)

        op = r[0]
        time = r[2]
        track = r[3]
        val = r[4]

        # if track != 0 and track != 1:
        #     continue

        if (op & combineNoteFlag == combineNoteFlag):
            if op & op & cnBegin:
                notes.append((time, 3, 0))
               # print("combine note %x time %f" % (op, time))
            continue
        
        if op & slideNote == slideNote:
            notes.append((time, 1, 0))
            print('slide note track ', track)
            continue
            
        if op & longNote == longNote:
            notes.append((time, 2, val))
            #print('long note during %d' % (val))
            continue

        notes.append((time, 0, 0))
        #print("touch note %d %x %x" % (i, op, val))
        
    notes = list(set(notes))
    return notes
    

def SaveInstantValue(beats, filename, postfix = ''):
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    with open(outname, 'w') as file:
        for obj in beats:
            file.write(str(obj) + '\n')

    return True

def SaveNote(notes, filename, postfix = ''):
    outname = os.path.splitext(filename)[0]
    outname = outname + postfix + '.csv'
    print('save', outname)
    with open(outname, 'w') as file:
        for v in notes:
            time, type, last = v[0], v[1] * 4 + 10, v[2]
            if last == 0:
                last = 50
            s = '%d,%d,%d\n' % (time, type, last)
            file.write(s)
            if type == 14:
                s = '%d,%d,%d\n' % (time, 13, last)
                file.write(s)
                s = '%d,%d,%d\n' % (time, 12, last)
                file.write(s)
        print('done')

if __name__ == '__main__':
    
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

    filename = r'E:\download\AI\4minuteshm\4minuteshm_4k_nm_copy.imd'
    filename = r'E:\download\AI\aibujieshi\aibujieshi_4k_nm.imd'
    filename = '/Users/xuchao/Documents/rhythmMaster/4minuteshm/4minuteshm_4k_nm.imd'
    filename = '/Users/xuchao/Documents/rhythmMaster/abracadabra/abracadabra_4k_hd.imd'
    notes = LoadRhythmMasterLevel(filename)
    SaveNote(notes, filename, '_notes')
    # SaveInstantValue(notes, filename, '_time')


    # test1(path)

    #load_levelinfo_file(path + 'level-infos.xml')

    # dir = os.listdir('/Users/xuchao/Documents/python/MusicLevelAutoGen/level')
    # for f in dir:
    #     print(os.path.join('/Users/xuchao/Documents/python/MusicLevelAutoGen/level', f))
    # print(dir)


    # with open('eggs.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    #     a = [[1, 2], [3, 4]]
    #     list(map(spamwriter.writerow, a))

    # a = input()
    # print(a)

    # scipy six decorator scikit-learn audioread numpy, resampy, joblib


    
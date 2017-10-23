import logger
from xml.etree import ElementTree  
import numpy as np
import os
import io
import sys

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
        print(pathname)
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


if __name__ == '__main__':
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

    if os.name == 'nt':
        path = r'D:/ab/QQX5_Mainland/exe/resources/level/test/'
    else :
        path = r'/Users/xuchao/Documents/python/MusicLevelAutoGen/'

    test1(path)

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


    
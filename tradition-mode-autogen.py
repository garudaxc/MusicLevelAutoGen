
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
    root = tree.getroot()
    levelInfo = root.find('LevelInfo')
    node = levelInfo.find('BPM')
    bpm = float(node.text)
    # num bar node = levelInfo.find('BarAmount')
    node = levelInfo.find('EnterTime')
    et = float(node.text)
    print(bpm, et)
    return bpm, et


def load_levels(path):
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]

        if (ext != '.xml'):
            continue
        
        pathname = os.path.join(path, f)
        xmlparser = ElementTree.XMLParser(encoding='utf-8')
        tree = ElementTree.parse(pathname, parser=xmlparser)
        parse_level_info(tree)



if __name__ == '__main__':
    print(__name__)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
    # xmlp = ElementTree.XMLParser(encoding="utf-8")  
    # f = ElementTree.parse('/Users/xuchao/Documents/python/MusicLevelAutoGen/level/sync0018.xml',parser=xmlp)

    load_levels('/Users/xuchao/Documents/python/MusicLevelAutoGen/level')
    # dir = os.listdir('/Users/xuchao/Documents/python/MusicLevelAutoGen/level')
    # for f in dir:
    #     print(os.path.join('/Users/xuchao/Documents/python/MusicLevelAutoGen/level', f))
    # print(dir)

    # tree = ElementTree.parse('/Users/xuchao/Documents/python/MusicLevelAutoGen/test.xml')
    # root = tree.getroot()
    # for node in root:
    #     print(node.tag, node.attrib)


    # with open('eggs.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    #     a = [[1, 2], [3, 4]]
    #     list(map(spamwriter.writerow, a))

    print(__name__)
    gen = tradition_mode_autogen('sdfs')
    print(gen.filename)

    # a = input()
    # print(a)

    # scipy six decorator scikit-learn audioread numpy, resampy, joblib


    
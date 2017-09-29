import csv
import sys
import logger
from xml.etree import ElementTree  



class tradition_mode_autogen:

    hop_lengh = 512
    
    # classify parameter
    k = 4

    def __init__(self, file):
        self.filename = file
        # self.__y, self.__sr = librosa.load(file, sr = None)
        # print('load ' + file, 'sr', self.__sr)


    def calc_beats(self):
        return

test_value = 101


def foo(*arg):
    for a in arg: print(a, end=' ')
    print()



if __name__ == '__main__':
    print(__name__)

    tree = ElementTree.parse('/Users/xuchao/Documents/python/MusicLevelAutoGen/test.xml')
    root = tree.getroot()
    for node in root:
        print(node.tag, node.attrib)


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

    
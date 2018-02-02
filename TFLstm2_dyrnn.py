import pickle
import os
import LevelInfo
import numpy as np
import lstm.myprocesser
from lstm import postprocess
import tensorflow as tf
from tensorflow.contrib import rnn
from lstm import postprocess
import time
import TFLstm


runList = []
def run(r):
    runList.append(r)
    return r


def ProcessInpuData():    
    #获取输入 bpm等
    pass

@run
def Test():
    songList = ['aiqingmaimai']
    fileanme = TFLstm.MakeLevelPathname(songList[0])
    level = LevelInfo.LoadRhythmMasterLevel(fileanme)





if __name__ == '__main__':
    for fun in runList:
        fun()

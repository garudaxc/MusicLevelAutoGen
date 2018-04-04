# coding=UTF-8
import numpy as np

import os
import madmom
# import librosa
# import librosa.display
import matplotlib.pyplot as plt
import pickle

#madmom.audio.signal

from madmom.processors import ParallelProcessor, Processor, SequentialProcessor
#from functools import partial
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)


def CreateProcesser(fps=100):
    # define pre-processing chain
    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    # process the multi-resolution spec & diff in parallel
    # process the multi-resolution spec & diff in parallel
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))

    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    return pre_processor



def TestWrite(pathname):
    pre_processor = CreateProcesser()

    d = pre_processor(pathname)
    print(type(d))
    print (d.shape)

    output = 'd:/work/signal_data.pk'
    if os.name != 'nt':
        output = '/Users/xuchao/Documents/work/signal_data.pk'

    with open(output, 'wb') as file:
        pickle.dump(d, file)

    return d


def LoadAndProcessAudio(pathname, fps=100):
    pre_processor = CreateProcesser(fps)
    d = pre_processor(pathname)
    print('audio processed ', d.shape)
    return d

def TestRead(pathname):
    d = None
    with open(pathname, 'rb') as file:
        d = pickle.load(file)
    
    print(type(d))
    print (d.shape)

    # librosa.display.specshow(d)
    # #plt.colorbar()
    # plt.show()     
    return d


def MainTest():
    idlist = [100052] #bpm有点不准
    #idlist = []
    path = r'd:\librosa\炫舞自动关卡生成\music\%d.mp3' % idlist[0]

    path = '/Users/xuchao/Documents/rhythmMaster/'
    if os.name == 'nt':
        path = 'D:/librosa/RhythmMaster/'
        path = r'D:/work/rythmmaster/'

    songName = '4minuteshm'

    pathname = '%s%s/%s.mp3' % (path, songName, songName)

    TestWrite(pathname)
    return

    d2 = TestRead(pathname)
    print(d2.shape)

    t = d2[6000:6002]
    t = np.array(t)
    #print(t)

    t = t.reshape(t.shape[0], 1, t.shape[1])
    print(t)
    print(t.shape)

    print(t[0].reshape(1, 1, 314))


if __name__ == '__main__':
    MainTest()
    
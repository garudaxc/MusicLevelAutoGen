
import numpy as np
import madmom

#madmom.audio.signal

from madmom.processors import ParallelProcessor, Processor, SequentialProcessor
#from functools import partial
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)


# define pre-processing chain
sig = SignalProcessor(num_channels=1, sample_rate=44100)
# process the multi-resolution spec & diff in parallel
# process the multi-resolution spec & diff in parallel
multi = ParallelProcessor([])
frame_sizes = [1024, 2048, 4096]
num_bands = [3, 6, 12]
for frame_size, num_bands in zip(frame_sizes, num_bands):
    frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
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


idlist = [1254, 1400, 1446, 1447, 1449, 1462, 1463, 1465, 1475, 1478, 1488, 1491] #拍子减半
idlist = [1262, 1279, 1374, 1391] #差两拍
idlist = [1245] #bpm有点不准
#idlist = []
path = r'D:\ab\QQX5_Mainland\exe\resources\media\audio\Music\song_%d.ogg' % idlist[0]

d = pre_processor(path)

print(type(d))
print (d.shape)



#processer = madmom.features.downbeats.RNNDownBeatProcessor()
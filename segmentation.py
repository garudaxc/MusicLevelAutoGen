import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = r'D:\librosa\炫舞自动关卡生成\music\100001.mp3'
file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\张杰 - 逆战.mp3'
file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Lily Allen - Hard Out Here.mp3'
y, sr = librosa.load(file, mono=True)
print('number of samples=' + str(len(y)))
print('sr=' + str(sr))

#y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
bounds = librosa.segment.agglomerative(chroma, 3)
bound_times = librosa.frames_to_time(bounds, sr=sr)


plt.figure()
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',linewidth=2, alpha=0.9, label='Segment boundaries')
plt.axis('tight')
plt.legend(frameon=True, shadow=True)
plt.title('Power spectrogram')
plt.tight_layout()
plt.show()

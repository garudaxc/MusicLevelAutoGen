import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# file = '/Users/xuchao/Music/网易云音乐/金志文 - 哭给你听.mp3'

file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\BOYFRIEND (보이프렌드) - On & On (온앤온).mp3'
file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Sam Tsui - Make It Up.mp3'
file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\QQ炫舞 - Love You.mp3'
file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Nightwish - I Want My Tears Back.mp3'
file = '/Users/xuchao/Music/网易云音乐/G.E.M.邓紫棋 - 后会无期.mp3'

def down_sample(chromas, rate):
    new = np.array([np.average(chromas[i:i+rate], 0) for i in range(0, len(chromas), rate)])
    return new


def path_smooth(ssm, l):
    m = len(ssm)
    newssm = np.empty([m, m])
    for i in range(m):
        for j in range(m):
            ll = min(l, m-i, m-j)
            newssm[i, j] = np.average([ssm[i+k, j+k] for k in range(ll)])
    return newssm

def calc_ssm(file, threshold = 0.25, sampleRate = 10, downsample = 10, smooth = 5):    
    y, sr = librosa.load(file, mono=True)
    print('number of samples=', len(y))
    print('sr=', sr)

    exp = np.log2(sr / float(sampleRate))
    hop = int(2 ** round(exp))
    print('hop lenth', hop)

    fmin = librosa.note_to_hz('C2')
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, fmin=fmin, n_octaves=5, n_chroma=36, hop_length=hop)
    print('numsamples', len(chroma_cens[0]))
	
    chroma_cens = down_sample(chroma_cens.T, downsample).T  
    plt.figure(1)
    librosa.display.specshow(chroma_cens, y_axis='chroma')

    m = len(chroma_cens[0])
    print('new chroma size', m)

    ssm = np.empty([m, m])

    for i in range(m):
        for j in range(m):
            ssm[i, j] = 1 - np.dot(chroma_cens[:,i], chroma_cens[:,j])

    minValue = np.min(ssm)
    maxValue = np.max(ssm)
    print('min max', minValue, maxValue)
    ssm = ssm - minValue
    ssm = ssm / (maxValue - minValue)    

    ssm = path_smooth(ssm, smooth)
                
##    plt.figure(1)
##    librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
    plt.figure(2)
    librosa.display.specshow(ssm)
    plt.colorbar()
    plt.show()        

calc_ssm(file)

##R = librosa.segment.recurrence_matrix(chroma_cens, mode='affinity')
##plt.show()


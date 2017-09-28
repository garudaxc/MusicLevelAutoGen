# -*- coding: utf-8 -*-
"""
======================
Laplacian segmentation
======================

This notebook implements the laplacian segmentation method of
`McFee and Ellis, 2014 <http://bmcfee.github.io/papers/ismir2014_spectral.pdf>`_,
with a couple of minor stability improvements.

Throughout the example, we will refer to equations in the paper by number, so it will be
helpful to read along.
"""

# Code source: Brian McFee
# License: ISC


###################################
# Imports
#   - numpy for basic functionality
#   - scipy for graph Laplacian
#   - matplotlib for visualization
#   - sklearn.cluster for K-Means
#
from __future__ import print_function

import numpy as np
import scipy
import matplotlib.pyplot as plt

import sklearn.cluster

import librosa
import librosa.display
import logger

#############################
# First, we'll load in a song

def calc_segment(y, sr, k = 4):
    ##############################################
    # Next, we'll compute and plot a log-power CQT
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                ref=np.max)
    
    # plt.figure(figsize=(12, 4))
    # librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
    #                          bins_per_octave=BINS_PER_OCTAVE,
    #                          x_axis='time')
    # plt.tight_layout()


    ##########################################################
    # To reduce dimensionality, we'll beat-synchronous the CQT
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # For plotting purposes, we'll need the timing of the beats
    # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                x_min=0,
                                                                x_max=C.shape[1]),
                                        sr=sr)

    # plt.figure(figsize=(12, 4))
    # librosa.display.specshow(Csync, bins_per_octave=12*3,
    #                          y_axis='cqt_hz', x_axis='time',
    #                          x_coords=beat_times)
    # plt.tight_layout()


    #####################################################################
    # Let's build a weighted recurrence matrix using beat-synchronous CQT
    # (Equation 1)
    # width=3 prevents links within the same bar
    # mode='affinity' here implements S_rep (after Eq. 8)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                        sym=True)

    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))


    ###################################################################
    # Now let's build the sequence matrix (S_loc) using mfcc-similarity
    #
    #   :math:`R_\text{path}[i, i\pm 1] = \exp(-\|C_i - C_{i\pm 1}\|^2 / \sigma^2)`
    #
    # Here, we take :math:`\sigma` to be the median distance between successive beats.
    #
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)


    ##########################################################
    # And compute the balanced combination (Equations 6, 7, 9)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path


    ###########################################################
    # Plot the resulting graphs (Figure 1, left and center)
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 3, 1)
    # librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time',
    #                          y_coords=beat_times)
    # plt.title('Recurrence similarity')
    # plt.subplot(1, 3, 2)
    # librosa.display.specshow(R_path, cmap='inferno_r')
    # plt.title('Path similarity')
    # plt.subplot(1, 3, 3)
    # librosa.display.specshow(A, cmap='inferno_r')
    # plt.title('Combined graph')
    # plt.tight_layout()


    #####################################################
    # Now let's compute the normalized Laplacian (Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # If we want k clusters, use the first k normalized eigenvectors.
    # Fun exercise: see how the segmentation changes as you vary k

    X = evecs[:, :k] / Cnorm[:, k-1:k]

    # Plot the resulting representation (Figure 1, center and right)

    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 2)
    # librosa.display.specshow(Rf, cmap='inferno_r')
    # plt.title('Recurrence matrix')

    # plt.subplot(1, 2, 1)
    # librosa.display.specshow(X,
    #                          y_axis='time',
    #                          y_coords=beat_times)
    # plt.title('Structure components')
    # plt.tight_layout()

    #############################################################
    # Let's use these k components to cluster beats into segments
    # (Algorithm 1)
    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)
    #logger.info(seg_ids)

    # and plot the results
    # plt.figure(figsize=(12, 4))
    colors = plt.get_cmap('Paired', k)

    # plt.subplot(1, 3, 2)
    # librosa.display.specshow(Rf, cmap='inferno_r')
    # plt.title('Recurrence matrix')
    # plt.subplot(1, 3, 1)
    # librosa.display.specshow(X,
    #                          y_axis='time',
    #                          y_coords=beat_times)
    # plt.title('Structure components')
    # plt.subplot(1, 3, 3)
    # librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors)
    # plt.title('Estimated segments')
    # plt.colorbar(ticks=range(k))
    # plt.tight_layout()

    ###############################################################
    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = seg_ids[bound_beats]
    
    my_bound_beats = librosa.util.fix_frames(bound_beats, x_min=0, x_max = len(beats) - 1)
    # 确保分段数量之间的关系
    bound_segs = bound_segs[:my_bound_beats.shape[0]-1]
    
    #logger.info('my bound beats', my_bound_beats)
    my_bound_frames = beats[my_bound_beats]


    logger.info('num bound seg', len(bound_segs), 'num bound frames', len(my_bound_frames))

    # Convert beat indices to frames
    bound_frames = beats[bound_beats]

    # Make sure we cover to the end of the track
    bound_frames = librosa.util.fix_frames(bound_frames,
                                        x_min=None,
                                        x_max=C.shape[1]-1)

    # logger.info(bound_segs)
    # logger.info(bound_frames)

    ###################################################
    # And plot the final segmentation over original CQT

    # sphinx_gallery_thumbnail_number = 5

    # from os import listdir
    # import os.path
    # def save_file(beats, mp3filename, postfix = ''):
    #     outname = os.path.splitext(mp3filename)[0]
    #     outname = outname + postfix + '.csv'
    #     librosa.output.times_csv(outname, beats)
    #     logger.info('output beat time file ' + outname)

    import matplotlib.patches as patches
    plt.figure(figsize=(12, 4))

    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                    fmin=librosa.note_to_hz('C1'),
                                    bins_per_octave=BINS_PER_OCTAVE)

    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                            bins_per_octave=BINS_PER_OCTAVE,
                            x_axis='time')
    ax = plt.gca()

    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        ax.add_patch(patches.Rectangle((interval[0], freqs[0]),
                                    interval[1] - interval[0],
                                    freqs[-1],
                                    facecolor=colors(label),
                                    alpha=0.50))

    plt.tight_layout()
    return my_bound_frames, bound_segs

def calc_power(y, sr, bound_frames, bound_segs):
    # 计算音乐强度
    S = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(S)
    db = (db - np.min(db)) / (np.max(db) - np.min(db))
    l = np.sum(db * db, 0)**0.5

    # 在每个分段内累计强度
    ss = np.array(list(sum(l[bound_frames[i]:bound_frames[i+1]]) for i in range(bound_frames.size - 1)))

    db_seg = [sum(ss[np.nonzero(bound_segs == i)]) for i in range(np.max(bound_segs) + 1)]
    db_seg = np.array(db_seg)
    
    frame_count = bound_frames[1:] - bound_frames[:-1]
    
    frame_count_seg = [sum(frame_count[np.nonzero(bound_segs == i)]) for i in range(np.max(bound_segs) + 1)]
    frame_count_seg = np.array(frame_count_seg)
    frame_count_seg = np.maximum(frame_count_seg, 1)
    
    db_aver = db_seg / frame_count_seg

    #归一化
    db_aver = (db_aver - np.min(db_aver)) / (np.max(db_aver) - np.min(db_aver))
        
    # draw line
    y_db = db_aver[bound_segs]
    y_db = np.array([y_db, y_db]).flatten('F')
    x_db = np.array([bound_frames[:-1], bound_frames[1:] - 1]).flatten('F')

    logger.info('y_db size ', y_db.size, 'bound frame size ', bound_frames.size)

    #onset_frames = librosa.onset.onset_detect(y=y, sr=sr, delta = 0.1)

    plt.figure(figsize=(12, 4))
    #plt.plot(l)
    plt.plot(x_db, y_db)
    #plt.vlines(onset_frames, 0, np.max(l))
    plt.tight_layout()

    x_db = librosa.frames_to_time(x_db, sr=sr)
    power_data = np.array([x_db, y_db]).T

    #plt.figure()
    seg_power = db_aver[bound_segs]

    return seg_power, power_data

    #librosa.display.specshow(db, sr=sr, y_axis='log', x_axis='time')

def gen_seg_probability(x_max, x_point, y_min=0.5):

    x = x_point / float(x_max - 1)

    y = (np.sin(6*np.pi*x + np.pi*0.5)) / 4 + 0.75
    y[np.flatnonzero(x < 1.0 / 6)] = y_min
    y[np.flatnonzero(x > 5.0 / 6)] = y_min

    #plt.figure(figsize=(12, 4))
    plt.plot(x * x_max, y)
    return y



if __name__ == '__main__':

    file = 'D:/librosa/炫舞自动关卡生成/music/100018.mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\BOYFRIEND (보이프렌드) - On & On (온앤온).mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Sam Tsui - Make It Up.mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Frankmusik - Wrecking Ball.mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\Lily Allen - Hard Out Here.mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\张杰 - 逆战.mp3'
    file = r'd:\librosa\炫舞自动关卡生成\测试歌曲\张靓颖 - 我是我的.mp3'
    file = '/Users/xuchao/Music/网易云音乐/金志文 - 哭给你听.mp3'
    file = '/Users/xuchao/Music/网易云音乐/G.E.M.邓紫棋 - 后会无期.mp3'
    file = '/Users/xuchao/Music/网易云音乐/Good Time.mp3'

    y, sr = librosa.load(file)

    bound_frames, bound_segs = calc_segment(y, sr)

    calc_power(y, sr, bound_frames, bound_segs)

    #plt.show()

    
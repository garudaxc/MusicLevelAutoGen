import numpy as np
import NotePreprocess
import TFLstm

def CompareData(arrA, arrB):
    idA = id(arrA)
    idB = id(arrB)
    print('compare object id', idA, idB)
    if idA == idB:
        print('compare object id equal. may be error!')
        return False

    arrA = np.array(arrA)
    arrB = np.array(arrB)
    shapeA = np.shape(arrA)
    shapeB = np.shape(arrB)
    print('compare shape', shapeA, shapeB)
    if len(shapeA) != len(shapeB):
        print('compare shape failed')
        return False

    for valA, valB in zip(shapeA, shapeB):
        if valA != valB:
            print('compare shape val failed')
            return False

    reshapeA = np.reshape(arrA, [-1])
    reshapeB = np.reshape(arrB, [-1])
    print('compare length', len(arrA), len(arrB))
    if len(arrA) != len(arrB):
        print('compare length failed')
        return False

    for vA, vB in zip(reshapeA, reshapeB):
        if vA != vB:
            print('compare arr val failed')
            return False

    print('compare res true')
    return True

def CheckSplitFuncValid():
    fileList, trainList, validateList = TFLstm.LoadCsvSingingFileList()
    for song, _ in fileList:
        print('check song', song)
        songFilePath = TFLstm.MakeMp3Pathname(song)
        sampleRate = 44100
        audioData = NotePreprocess.LoadAudioFile(songFilePath, sampleRate)
        audioData = np.array(audioData)
        specDiff, melLogSpec = NotePreprocess.SpecDiffAndMelLogSpecEx(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
        specDiffSrc, melLogSpecSrc = NotePreprocess.SpecDiffAndMelLogSpec(audioData, sampleRate, [1024, 2048, 4096], [3, 6, 12], 100)
        if not CompareData(specDiffSrc, specDiff):
            print(song, 'check failed')
            return False

        if not CompareData(melLogSpecSrc, melLogSpec):
            print(song, 'check failed')
            return False

        print(song, 'check valid')

    return True

if __name__ == '__main__':

    print('TestCase end')

import os
import pickle
import myprocesser

class BasicNoteFeatureProcessor():
    def __init__(self):
        self.posfix = '_feature'

    def makeFeatureFilePath(self, filePath):
        arr = os.path.splitext(filePath)
        if arr[1] == '.raw':
            return filePath

        return arr[0] + self.posfix + '.raw'

    def extract(self, audioFilePath):
        print('BasicNoteFeatureProcessor extract nothing')
        return []

    def save(self, filePath, features, labels):
        targetFilePath = self.makeFeatureFilePath(filePath)
        print('NoteFeatureProcessor save', targetFilePath)
        with open(targetFilePath, 'wb') as file:
            pickle.dump(features, file)
            pickle.dump(labels, file)
            print('raw file saved', features.shape)

    def load(self, filePath):
        targetFilePath = self.makeFeatureFilePath(filePath)
        if not os.path.exists(targetFilePath):
            print('filePath not exist', targetFilePath)
            return [], []

        print('NoteFeatureProcessor load', targetFilePath)
        with open(targetFilePath, 'rb') as file:
            features = pickle.load(file)
            labels = pickle.load(file)

        return features, labels

class ShortNoteFeatureProcessor(BasicNoteFeatureProcessor):
    def __init__(self):
        self.posfix = '_feature_short'

    def extract(self, audioFilePath):
        print('ShortNoteFeatureProcessor extract')
        return myprocesser.LoadAndProcessAudio(audioFilePath)

class LongNoteFeatureProcessor(BasicNoteFeatureProcessor):
    def __init__(self):
        self.posfix = '_feature_long'

    def extract(self, audioFilePath):
        print('LongNoteFeatureProcessor extract')
        return myprocesser.LoadAndProcessAudio(audioFilePath)
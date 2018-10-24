import os
import numpy as np
import tensorflow as tf
import madmom
import NotePreprocess
import NoteEnvironment
import NoteModel
import ModelTool
import time
import myprocesser

class NoteLevelGenerator():
    def __init__(self):
        self.graph = None
        self.sess = None
        self.fps = 100
        self.sampleRate = 44100
        self.hopSize = int(self.sampleRate / self.fps)
        self.frameSizeArr = [1024, 2048, 4096]
        self.numBandArr = [3, 6, 12]
        self.noteModelBatchSize = 8
        self.noteModelOverlap = 128
        self.noteModelInputDim = 314
        self.bpmModelInputDim = 314
        self.bpmModelBatchSize = 1
        self.bpmModelCount = 8

        # todo limit time duraion
        self.minDuration = 10
        self.maxDuration = 60 * 20

    def initialize(self, resourceDir=None):
        if resourceDir is None:
            resourceDir = self.defaultResourceDir()

        if not os.path.exists(resourceDir):
            return False

        shortModelPath = self.getModelPath(resourceDir, 'model_singing')
        longModelPath = self.getModelPath(resourceDir, 'model_longnote')
        onsetModelPath = self.getModelPath(resourceDir, 'onset')
        bpmModelPathArr = []
        for i in range(self.bpmModelCount):
            bpmModelPathArr.append(self.getModelPath(resourceDir, 'downbeats_' + str(i)))

        graph = tf.Graph()

        sampleRate = self.sampleRate
        fps = self.fps 
        hopSize = self.hopSize
        frameSizeArr = self.frameSizeArr
        numBandArr = self.numBandArr

        with graph.as_default():
            # todo use enviroment setting
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            # gpu_options.allow_growth = False
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess = tf.Session()
            signalTensorArr, preprocessTensorArr, diffFrameArr = ModelTool.BuildPreProcessGraph('preprocess', sampleRate, frameSizeArr, numBandArr, hopSize)
            self.signalTensorArr = signalTensorArr
            self.preprocessTensorArr = preprocessTensorArr
            self.diffFrameArr = diffFrameArr

            self.shortModel = NoteModel.NoteDetectionModel('short_note', self.noteModelBatchSize, 0, 3, 26, self.noteModelInputDim, 3, timeMajor=False, useCudnn=True, restoreCudnnWithGPUMode=True)
            self.shortModel.Restore(sess, shortModelPath)
            self.shortTensorDic = self.shortModel.GetTensorDic()
            self.shortInitialStatesZero = self.shortModel.InitialStatesZero()
            
            self.longModel = NoteModel.NoteDetectionModel('long_note', self.noteModelBatchSize, 0, 3, 26, self.noteModelInputDim, 4, timeMajor=False, useCudnn=True, restoreCudnnWithGPUMode=True)
            self.longModel.Restore(sess, longModelPath)
            self.longTensorDic = self.longModel.GetTensorDic()
            self.longInitialStatesZero  = self.longModel.InitialStatesZero()

            onsetVariableScopeName = 'onset'
            self.onsetTensorDic = ModelTool.BuildOnsetModelGraph(onsetVariableScopeName)
            varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=onsetVariableScopeName)
            saver = tf.train.Saver(var_list=varList)
            saver.restore(sess, onsetModelPath)

            bpmModelTensorDicArr = []
            for idx, bpmModelPath in enumerate(bpmModelPathArr):
                bpmModelTensorDicArr.append(self.initBPMModel(sess, bpmModelPath, 'downbeats_' + str(idx)))
            self.bpmModelTensorDicArr = bpmModelTensorDicArr

        self.graph = graph
        self.sess = sess
        # just for gpu version init
        fakeAudioData = [0.0] * self.sampleRate * 60 * 1
        self.run('', '', fakeAudioData=fakeAudioData)
        print('initialize end')

        return True

    def run(self, inputFilePath, outputFilePath, fakeAudioData=None, outputDebugInfo=False):
        if self.graph is None or self.sess is None:
            return False

        sampleRate = self.sampleRate

        if outputDebugInfo:
            timeStart = time.time()

        if fakeAudioData is None:
            audioData = NotePreprocess.LoadAudioFile(inputFilePath, sampleRate)
        else:
            audioData = fakeAudioData
        audioData = np.array(audioData)
        minSampleCount = self.minDuration * sampleRate
        maxSampleCoun = self.maxDuration * sampleRate
        lenAudioData = len(audioData)
        if audioData is None or lenAudioData < minSampleCount or lenAudioData > maxSampleCoun:
            return False

        if outputDebugInfo:
            print('cost audio decode', time.time() - timeStart)
            timeStart = time.time()

        frameCount = ModelTool.FrameCount(audioData, self.hopSize)
        scaleValue = ModelTool.ScaleValue(audioData)
        tfAudioData = audioData / scaleValue
        tfAudioDataArr = ModelTool.PaddingAudioData(tfAudioData, self.frameSizeArr)

        graph = self.graph
        sess = self.sess
        with graph.as_default():
            if outputDebugInfo:
                print('cost audio padding', time.time() - timeStart)
                timeStart = time.time()

            preprocessRes = self.runSess(sess, self.preprocessTensorArr, self.signalTensorArr, tfAudioDataArr)

            if outputDebugInfo:
                print('cost process', time.time() - timeStart)
                timeStart = time.time()

            tfLogMel = []
            tfSpecDiff = []
            for idx in range(0, len(preprocessRes), 2):
                tfLogMel.append(preprocessRes[idx])
                tfSpecDiff.append(preprocessRes[idx + 1])
            tfLogMel = ModelTool.PostProcessTFLogMel(tfLogMel, frameCount, swap=True)
            tfSpecDiff = ModelTool.PostProcessTFSpecDiff(tfSpecDiff, frameCount, self.diffFrameArr)

            noteModelInputData = myprocesser.FeatureStandardize(tfSpecDiff)
            noteModelInputData = NotePreprocess.SplitData(noteModelInputData, self.noteModelBatchSize, self.noteModelOverlap)
            noteModelInputData = noteModelInputData.reshape(self.noteModelBatchSize, -1, self.noteModelInputDim)

            bpmModelInputData = np.reshape(tfSpecDiff, (-1, self.bpmModelBatchSize, self.bpmModelInputDim))
            bpmSeqLen = [len(bpmModelInputData)] * self.bpmModelBatchSize

            runTensorArr = []
            inputTensorArr = []
            inputDataArr = []

            self.addNoteModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.shortTensorDic, noteModelInputData, self.shortInitialStatesZero)
            self.addNoteModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.longTensorDic, noteModelInputData, self.longInitialStatesZero)
            self.addOnsetModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.onsetTensorDic, tfLogMel)
            for bpmTensorDic in self.bpmModelTensorDicArr:
                self.addBPMModelSessData(runTensorArr, inputTensorArr, inputDataArr, bpmTensorDic, bpmModelInputData, bpmSeqLen)

            if outputDebugInfo:
                print('cost temp cpu', time.time() - timeStart)
                timeStart = time.time()
                
            res = self.runSess(sess, runTensorArr, inputTensorArr, inputDataArr)
            shortModelRes = res[0]
            longModelRes = res[1]
            onsetModelRes = res[2]

            shortModelRes = self.postProcessNoteModelRes(shortModelRes, len(tfSpecDiff))
            longModelRes = self.postProcessNoteModelRes(longModelRes, len(tfSpecDiff))

            if outputDebugInfo:
                print('cost model', time.time() - timeStart)
                print('shape onset res', np.shape(onsetModelRes))
                timeStart = time.time()

        return True

    def releaseResource(self):
        if self.sess is not None:
            self.sess.close()


    def runSess(self, sess, runTensorArr, inputTensorArr, inputDataArr):
        dic = {}
        for inputTensor, inputData in zip(inputTensorArr, inputDataArr):
            dic[inputTensor] = inputData

        res = sess.run(runTensorArr, feed_dict=dic)
        return res

    def defaultResourceDir(self):
        filePath = os.path.abspath(__file__)
        fileDir = os.path.dirname(filePath)
        resourceDir = os.path.join(fileDir, 'resource')
        return resourceDir

    def getModelPath(self, resourceDir, modelName):
        modelDir = os.path.join(resourceDir, 'model')
        modelDir = os.path.join(modelDir, modelName)
        modelFilePath = os.path.join(modelDir, modelName + '.ckpt')
        return modelFilePath

    def getLevelTemplatePath(self, resourceDir):
        fileDir= os.path.join(resourceDir, 'template')
        filePath = os.path.join(fileDir, 'idol_template.xml')
        return filePath

    def initBPMModel(self, sess, modelPath, variableScopeName):
        batchSize = 1
        numUnits = 25
        inputDim = 314
        usePeepholes = True
        useLSTMBlockFusedCell = True
        tensorDic = ModelTool.BuildDownbeatsModelGraph(variableScopeName, 3, batchSize, numUnits, inputDim, usePeepholes, [numUnits * 2, 3], [3], tf.nn.softmax, useLSTMBlockFusedCell)

        varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variableScopeName)
        saver = tf.train.Saver(var_list=varList)
        saver.restore(sess, modelPath)
        return tensorDic

    def addNoteModelSessData(self, runTensorArr, inputTensorArr, inputDataArr, tensorDic, inputData, initState):
        runTensorArr.append(tensorDic['predict_op'])
        inputTensorArr.append(tensorDic['X'])
        inputTensorArr.append(tensorDic['initial_states'])
        inputDataArr.append(inputData)
        inputDataArr.append(initState)

    def addOnsetModelSessData(self, runTensorArr, inputTensorArr, inputDataArr, tensorDic, inputData):
        runTensorArr.append(tensorDic['output'])
        inputTensorArr.append(tensorDic['X'])
        inputDataArr.append(inputData)

    def addBPMModelSessData(self, runTensorArr, inputTensorArr, inputDataArr, tensorDic, inputData, seqLen):
        runTensorArr.append(tensorDic['output'])
        inputTensorArr.append(tensorDic['X'])
        inputTensorArr.append(tensorDic['sequence_length'])
        inputDataArr.append(inputData)
        inputDataArr.append(seqLen)

    def postProcessNoteModelRes(self, res, frameCount):
        outputDim = len(res[0])
        res = res.reshape(self.noteModelBatchSize, -1, outputDim)
        res = res[:, self.noteModelOverlap : (len(res[0]) - self.noteModelOverlap), :]
        res = res.reshape(-1, outputDim)
        res = res[0:frameCount]
        return res

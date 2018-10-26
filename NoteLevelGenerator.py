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
from functools import partial
import postprocess
from tensorflow.contrib import signal

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

        self.bpmModelUseMerged = True
        self.bpmModelCount = 8
        self.bpmModelInputDim = 314
        self.bpmModelBatchSize = 16
        self.bpmModelOverlap = 500

        # todo limit time duraion
        self.minDuration = 10
        self.maxDuration = 60 * 20

    def initialize(self, resourceDir=None):
        if resourceDir is None:
            resourceDir = self.defaultResourceDir()

        if not os.path.exists(resourceDir):
            return False

        self.resourceDir = resourceDir

        shortModelPath = self.getModelPath(resourceDir, 'model_singing')
        longModelPath = self.getModelPath(resourceDir, 'model_longnote')
        onsetModelPath = self.getModelPath(resourceDir, 'onset')
        bpmModelPathArr = []
        bpmModelVariableScopeNameArr = []
        if self.bpmModelUseMerged:
            varScopeName = 'downbeats'
            bpmModelVariableScopeNameArr.append(varScopeName)
            bpmModelPathArr.append(self.getModelPath(resourceDir, varScopeName))
        else:
            for i in range(self.bpmModelCount):
                varScopeName = 'downbeats_' + str(i)
                bpmModelVariableScopeNameArr.append(varScopeName)
                bpmModelPathArr.append(self.getModelPath(resourceDir, varScopeName))

        graph = tf.Graph()

        sampleRate = self.sampleRate
        fps = self.fps 
        hopSize = self.hopSize
        frameSizeArr = self.frameSizeArr
        numBandArr = self.numBandArr

        restoreCudnnWithGPUMode = NoteEnvironment.IsGPUAvailable()

        with graph.as_default():
            # todo use enviroment setting
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            # gpu_options.allow_growth = False
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess = tf.Session()
            signalTensor, preprocessTensorArr, diffFrameArr, logMelSpecTensor, specDiffTensor, frameCountTensor, scaleValueTensor = ModelTool.BuildPreProcessGraph('preprocess', sampleRate, frameSizeArr, numBandArr, hopSize)
            self.signalTensor = signalTensor
            self.preprocessTensorArr = preprocessTensorArr
            self.diffFrameArr = diffFrameArr
            self.logMelSpecTensor = logMelSpecTensor
            self.specDiffTensor = specDiffTensor
            self.frameCountTensor = frameCountTensor
            self.scaleValueTensor = scaleValueTensor

            noteModelInputDataTensor, bpmInputDataTensor = self.tfFeatureToModelInput(logMelSpecTensor, specDiffTensor)
            self.noteModelInputDataTensor = noteModelInputDataTensor
            self.bpmInputDataTensor = bpmInputDataTensor

            self.shortModel = NoteModel.NoteDetectionModel('short_note', self.noteModelBatchSize, 0, 3, 26, self.noteModelInputDim, 3, timeMajor=False, useCudnn=True, restoreCudnnWithGPUMode=restoreCudnnWithGPUMode)
            self.shortModel.Restore(sess, shortModelPath)
            self.shortTensorDic = self.shortModel.GetTensorDic()
            self.shortInitialStatesZero = self.shortModel.InitialStatesZero()
            
            self.longModel = NoteModel.NoteDetectionModel('long_note', self.noteModelBatchSize, 0, 3, 26, self.noteModelInputDim, 4, timeMajor=False, useCudnn=True, restoreCudnnWithGPUMode=restoreCudnnWithGPUMode)
            self.longModel.Restore(sess, longModelPath)
            self.longTensorDic = self.longModel.GetTensorDic()
            self.longInitialStatesZero  = self.longModel.InitialStatesZero()

            onsetVariableScopeName = 'onset'
            self.onsetTensorDic = ModelTool.BuildOnsetModelGraph(onsetVariableScopeName)
            varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=onsetVariableScopeName)
            saver = tf.train.Saver(var_list=varList)
            saver.restore(sess, onsetModelPath)

            bpmModelTensorDicArr = []
            for bpmModelPath, bpmModelVariableScopeName in zip(bpmModelPathArr, bpmModelVariableScopeNameArr):
                bpmModelTensorDicArr.append(self.initBPMModel(sess, bpmModelPath, bpmModelVariableScopeName))
            self.bpmModelTensorDicArr = bpmModelTensorDicArr

        self.graph = graph
        self.sess = sess
        # just for gpu version init
        fakeAudioData = [0.0] * self.sampleRate * 60 * 1
        self.run('', '', fakeAudioData=fakeAudioData)
        print('initialize end')

        return True

    def run(self, inputFilePath, outputFilePath, isTranscodeByQAAC=False, fakeAudioData=None, outputDebugInfo=False, saveDebugFile=False):
        if self.graph is None or self.sess is None:
            return False

        templateFilePath = self.getLevelTemplatePath(self.resourceDir)
        if not os.path.exists(templateFilePath):
            return False

        sampleRate = self.sampleRate

        if outputDebugInfo:
            timeStart = time.time()
            allTimeStart = timeStart

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
            tfTimeStart = timeStart

        frameCount = ModelTool.FrameCount(audioData, self.hopSize)
        scaleValue = ModelTool.ScaleValue(audioData)

        graph = self.graph
        sess = self.sess
        with graph.as_default():
            preprocessRes = self.runSess(sess, 
                    [self.noteModelInputDataTensor, self.bpmInputDataTensor, self.logMelSpecTensor, self.specDiffTensor], 
                    [self.signalTensor, self.frameCountTensor, self.scaleValueTensor], 
                    [audioData, frameCount, scaleValue]
            )
            noteModelInputData = preprocessRes[0]
            bpmModelInputData = preprocessRes[1]
            tfLogMel = preprocessRes[2]
            tfSpecDiff = preprocessRes[3]
            if outputDebugInfo:
                print('cost process', time.time() - timeStart)
                timeStart = time.time()

            noteModelSeqLen = [len(noteModelInputData[0])] * self.noteModelBatchSize
            bpmSeqLen = [len(bpmModelInputData)] * self.bpmModelBatchSize

            runTensorArr = []
            inputTensorArr = []
            inputDataArr = []

            self.addNoteModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.shortTensorDic, noteModelInputData, noteModelSeqLen, self.shortInitialStatesZero)
            self.addNoteModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.longTensorDic, noteModelInputData, noteModelSeqLen, self.longInitialStatesZero)
            self.addOnsetModelSessData(runTensorArr, inputTensorArr, inputDataArr, self.onsetTensorDic, tfLogMel)
            for bpmTensorDic in self.bpmModelTensorDicArr:
                self.addBPMModelSessData(runTensorArr, inputTensorArr, inputDataArr, bpmTensorDic, bpmModelInputData, bpmSeqLen)

            if outputDebugInfo:
                print('cost temp cpu', time.time() - timeStart)
                timeStart = time.time()
                
            res = self.runSess(sess, runTensorArr, inputTensorArr, inputDataArr)
            if outputDebugInfo:
                curTime = time.time()
                print('cost model', curTime - timeStart)
                print('tf cost', curTime - tfTimeStart)
                timeStart = curTime

        if fakeAudioData is not None:
            return True
        
        shortModelRes = self.postProcessNoteModelRes(res[0], len(tfSpecDiff))
        longModelRes = self.postProcessNoteModelRes(res[1], len(tfSpecDiff))
        onsetModelRes = res[2]
        bpm, et, duration = self.postProcessBPMModelRes(res, audioData, tfSpecDiff, inputFilePath, isTranscodeByQAAC)
        if outputDebugInfo:
            print('cost post', time.time() - timeStart)

        postprocess.GenerateLevelImp(inputFilePath, duration, bpm, et, 
                    shortModelRes, longModelRes, outputFilePath, templateFilePath, 0.7, 0.7, 
                    saveDebugFile=saveDebugFile, 
                    onsetActivation=onsetModelRes, enableDecodeOffset=isTranscodeByQAAC)
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
        numUnits = 25
        inputDim = self.bpmModelInputDim
        outputDim = 3
        tfActivationFunc = tf.nn.softmax
        inputDimTile = None
        if self.bpmModelUseMerged:
            numUnits = numUnits * self.bpmModelCount
            inputDim = inputDim * self.bpmModelCount
            tfActivationFunc = partial(ModelTool.TFSoftMaxForMergeAll, modelCount=self.bpmModelCount, outputDimPerModel=outputDim)
            outputDim = outputDim * self.bpmModelCount
            inputDimTile = self.bpmModelCount

        usePeepholes = True
        useLSTMBlockFusedCell = True
        tensorDic = ModelTool.BuildDownbeatsModelGraph(variableScopeName, 3, self.bpmModelBatchSize, 
                    numUnits, inputDim, usePeepholes, [numUnits * 2, outputDim], [outputDim], 
                    tfActivationFunc, useLSTMBlockFusedCell, inputDimTile=inputDimTile)

        varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variableScopeName)
        saver = tf.train.Saver(var_list=varList)
        saver.restore(sess, modelPath)
        return tensorDic

    def addNoteModelSessData(self, runTensorArr, inputTensorArr, inputDataArr, tensorDic, inputData, seqLen, initState):
        runTensorArr.append(tensorDic['predict_op'])
        inputTensorArr.append(tensorDic['X'])
        inputTensorArr.append(tensorDic['sequence_length'])
        inputTensorArr.append(tensorDic['initial_states'])
        inputDataArr.append(inputData)
        inputDataArr.append(seqLen)
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

    def postProcessBPMModelRes(self, sessRes, audioData, featureData, audioFilePath, isTranscodeByQAAC):
        if self.bpmModelUseMerged:
            bpmRes = sessRes[3]
            bpmOutputDim = bpmRes.shape[-1]
            bpmRes = bpmRes.reshape(len(bpmRes), -1, self.bpmModelBatchSize, bpmOutputDim)
            bpmRes = bpmRes[:, self.bpmModelOverlap: len(bpmRes[0]) - self.bpmModelOverlap, :, :]
            bpmRes = bpmRes.transpose(0, 2, 1, 3)
            bpmRes = bpmRes.reshape(len(bpmRes), -1, bpmOutputDim)
        else:
            bpmRes = sessRes[len(sessRes) - self.bpmModelCount:]

        predict = madmom.ml.nn.average_predictions(bpmRes)
        act = partial(np.delete, obj=0, axis=1)
        downBeatOutput = act(predict)

        duration = len(audioData) / self.sampleRate
        analysisRange = (0, int(duration))
        bpm, et = NotePreprocess.CalcBpmET(audioData, self.sampleRate, duration, downBeatOutput, downBeatOutput, analysisRange)
        if isTranscodeByQAAC:
            et = et + NotePreprocess.DecodeOffset(audioFilePath)

        duration = int(duration * 1000)
        et = int(et * 1000)

        # proc = NotePreprocess.CustomRNNDownBeatProcessor()
        # srcRes = proc(featureData)
        # srcBPM, srcET = NotePreprocess.CalcBpmET(audioData, self.sampleRate, duration, srcRes, srcRes, analysisRange)
        # if isTranscodeByQAAC:
        #     srcET = srcET + NotePreprocess.DecodeOffset(audioFilePath)
        # srcDuration = duration
        # srcET = int(srcET * 1000)
        # print('src bpm', srcBPM, 'et', srcET, 'duration', srcDuration)
        # import DownbeatTracking
        # DownbeatTracking.SaveInstantValue(downBeatOutput[:, 0], audioFilePath, '_bpm_split_0_%d_overlap_%d' % (self.bpmModelBatchSize, self.bpmModelOverlap))
        # DownbeatTracking.SaveInstantValue(downBeatOutput[:, 1], audioFilePath, '_bpm_split_1_%d_overlap_%d' % (self.bpmModelBatchSize, self.bpmModelOverlap))
        # DownbeatTracking.SaveInstantValue(srcRes[:, 0], audioFilePath, '_bpm_src_0')
        # DownbeatTracking.SaveInstantValue(srcRes[:, 1], audioFilePath, '_bpm_src_1')

        return bpm, et, duration

    def tfSplitData(self, arr, splitCount, overlap):
        srcArrShape = tf.shape(arr)
        subArrDataLength = tf.cast((tf.ceil(srcArrShape[0] / splitCount)), tf.int32)
        appendLength = subArrDataLength * splitCount - srcArrShape[0]
        arr = tf.pad(arr, [[overlap, overlap + appendLength], [0, 0]])
        doubleOverlap = 2 * overlap
        return signal.frame(arr, subArrDataLength + doubleOverlap, subArrDataLength, pad_end=False, axis=-2)

    def tfStandardize(self, data):
        featureMin = tf.reduce_min(data)
        featureMax = tf.reduce_max(data)
        delta = featureMax - featureMin
        data = data * (1.0 / delta) - featureMin / delta
        return data

    def tfFeatureToModelInput(self, tfLogMel, tfSpecDiff):
        noteModelInputData = self.tfStandardize(tfSpecDiff)
        noteModelInputData = self.tfSplitData(noteModelInputData, self.noteModelBatchSize, self.noteModelOverlap)
        noteModelInputData = tf.reshape(noteModelInputData, (self.noteModelBatchSize, -1, self.noteModelInputDim))

        bpmModelInputData = self.tfSplitData(tfSpecDiff, self.bpmModelBatchSize, self.bpmModelOverlap)
        bpmModelInputData = tf.reshape(bpmModelInputData, (self.bpmModelBatchSize, -1, self.bpmModelInputDim))
        bpmModelInputData = tf.transpose(bpmModelInputData, (1, 0, 2))
        return noteModelInputData, bpmModelInputData
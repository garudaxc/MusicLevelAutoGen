#coding=utf-8
import xml.dom.minidom
import random
import math
from enum import Enum
import os.path
#动作
class DanceClip:
	def __init__(self):
		self.type = "normal"
		self.bar = 0
		self.level = 1;
		self.ActionStyle ="Common"
		self.ActionLevel = 1
		self.ActionLength = 1

#镜头
class CameraClip:
	def __init__(self):
		self.id="front_01"
		self.bar = 0

#关卡
class LevelDocument:
	def __init__(self):
		self.LevelInfo = dict(
			Type="sync",
			Difficulty=1,
			EXPRatio="1.0000",
			BPM =120.0,
			BeatPerBar=4,
			BeatLen=8,
			BarAmount=200,
			EnterTime=0
			)
		self.MusicInfo = dict(
			Title="xxx",
			# Artist="xxx",
			FilePath="xxx"
			)
		self.DanceSegment_1 = []
		self.DanceSegment_2 = []
		self.DanceSegment_3 = []
		self.DanceSegment_4 = []
		self.FaceSequence = []
		self.CameraSequence = []

#关卡xml转换器
class XmlConverter:
	def __init__(self,levelDocument):
		self.levelDocument = levelDocument
		self.xmlDocument = xml.dom.minidom.Document() 

	def run(self):
		
		LevelNode =self.xmlDocument.createElement('Level') 
		self.xmlDocument.appendChild(LevelNode) 
		# 1 LevelInfo
		LevelNode.appendChild(self.__DictDataToXmlNode('LevelInfo',self.levelDocument.LevelInfo))

		# 2 MusicInfo
		LevelNode.appendChild(self.__DictDataToXmlNode('MusicInfo',self.levelDocument.MusicInfo))

		# 3 DanceSequence
		LevelNode.appendChild(self.__DanceSequenceToXmlNode())

		# 4 FaceSequence
		LevelNode.appendChild(self.__FaceSequenceToXmlNode())

		# 5 CameraSequence
		LevelNode.appendChild(self.__CameraSequenceToXmlNode())
	
	def writeToFile(self,filePath):
		fp = open(filePath, 'w')
		self.xmlDocument.writexml(fp,  addindent='\t',  newl='\n', encoding="utf-8")

	def toXmlStr(self):
		return self.xmlDocument.toxml()

	def __DictDataToXmlNode(self,rootNodeName,dictData):
		rootNode = self.xmlDocument.createElement(rootNodeName)
		for (key,value) in dictData.items():
			keyNode = self.xmlDocument.createElement(key)
			keyNode.appendChild(self.xmlDocument.createTextNode(str(value)))
			rootNode.appendChild(keyNode)
		if rootNodeName is 'LevelInfo':
			rootNode.appendChild(self.xmlDocument.createElement("Remark"))
		return rootNode

	def __DanceSequenceToXmlNode(self):
		DanceSequenceNode = self.xmlDocument.createElement('DanceSequence')

		DanceSequenceNode.setAttribute("UnSync","0")
		DanceSequenceNode.setAttribute("Eight_Key_P","0.5")
		DanceSequenceNode.setAttribute("AllRythm","0")
		
		DanceSequenceNode.appendChild(self.__DanceSegmentToXmlNode(self.levelDocument.DanceSegment_1))
		DanceSequenceNode.appendChild(self.__DanceSegmentToXmlNode(self.levelDocument.DanceSegment_2))
		DanceSequenceNode.appendChild(self.__DanceSegmentToXmlNode(self.levelDocument.DanceSegment_3))
		DanceSequenceNode.appendChild(self.__DanceSegmentToXmlNode(self.levelDocument.DanceSegment_4))
		return DanceSequenceNode

	def __DanceSegmentToXmlNode(self,danceSegmentLst):
		DanceSegmentNode = self.xmlDocument.createElement("DanceSegment")
		DanceSegmentNode.setAttribute("startBar",str(danceSegmentLst[0].bar))
		DanceSegmentNode.setAttribute("type","normal")

		for danceClip in danceSegmentLst:
			DanceSegmentNode.appendChild(self.__DanceClipToXmlNode(danceClip))

		return DanceSegmentNode

	def __DanceClipToXmlNode(self,danceClip):
		DanceNode = self.xmlDocument.createElement("Dance")
		
		DanceNode.setAttribute("type",str(danceClip.type))
		DanceNode.setAttribute("bar",str(danceClip.bar))
		DanceNode.setAttribute("level",str(danceClip.level))
		if danceClip.type is not "Blank":
			ActionSelectorNode =  self.xmlDocument.createElement("ActionSelector")
			DanceNode.appendChild(ActionSelectorNode)

			ActionRuleNode = self.xmlDocument.createElement("ActionRule")
			ActionSelectorNode.appendChild(ActionRuleNode)

			ActionLengthNode = self.xmlDocument.createElement("ActionLength")
			ActionRuleNode.appendChild(ActionLengthNode)
			ActionStyleNode = self.xmlDocument.createElement("ActionStyle")
			ActionRuleNode.appendChild(ActionStyleNode)
			ActionLevelNode = self.xmlDocument.createElement("ActionLevel")
			ActionRuleNode.appendChild(ActionLevelNode)

			ActionLengthNode.appendChild(self.xmlDocument.createTextNode(str(danceClip.ActionLength)))
			ActionStyleNode.appendChild(self.xmlDocument.createTextNode(str(danceClip.ActionStyle)))
			ActionLevelNode.appendChild(self.xmlDocument.createTextNode(str(danceClip.ActionLevel)))
		return DanceNode

	def __FaceSequenceToXmlNode(self):

		FaceSequenceNode = self.xmlDocument.createElement("FaceSequence")
		FACIAL_TRACK_Node = self.xmlDocument.createElement("FACIAL_TRACK")
		FaceSequenceNode.appendChild(FACIAL_TRACK_Node)
		return FaceSequenceNode

	def __CameraSequenceToXmlNode(self):

		CameraSequenceNode = self.xmlDocument.createElement("CameraSequence")

		for cameraClip in self.levelDocument.CameraSequence:
			CameraNode = self.xmlDocument.createElement("Camera")
			CameraNode.setAttribute("id",cameraClip.id)
			CameraNode.setAttribute("bar",str(cameraClip.bar))
			CameraSequenceNode.appendChild(CameraNode)
		
		return CameraSequenceNode

#生成结果
class GeneratorResult(Enum):
	SUCCESS = 0                     #生成成功
	MUSIC_LENGTH_EORROR = 1			#歌曲时长不合规范
	INPUT_EORROR = 2				#转换参数错误
	SECTION_BAR_AMOUNT_SHORT = 3    #段落过短低于16小结

#关卡生成器
class Generator:
	def __init__(self,inputData,ruler,logger):
		self.__document = LevelDocument()
		self.__generatorInput = inputData
		self.__ruler = ruler
		self.__camer_get_index = 0
		self.__fixed_camera_shot_container =[]
		self.__Logger = logger
		for cameraShot in self.__ruler["camera_shot"]["fixed_shot_container"]:
  			self.__fixed_camera_shot_container.append(cameraShot)

	def run(self):
		result = self.__CheckInput()
		if result is not GeneratorResult.SUCCESS:
			return result

		result = self.__InputDoumnetInfo()
		if result is not GeneratorResult.SUCCESS:
			return result

		result = self.__CalculateBasicInformation()
		if result is not GeneratorResult.SUCCESS:
			return result

		self.__FillActionClipInfomation()
		self.__FillCameraClipInfomation()
		return GeneratorResult.SUCCESS

	@property
	def doument(self):
		return self.__document

	def __CheckInput(self):
		if self.__generatorInput["musicLength"]<self.__ruler["mini_music_length"] or self.__generatorInput["musicLength"]>self.__ruler["max_music_length"]:
			return GeneratorResult.MUSIC_LENGTH_EORROR
		if self.__generatorInput["musicBPM"]<=0:
			return GeneratorResult.INPUT_EORROR
		if self.__generatorInput["musicET"]<0 or self.__generatorInput["musicET"]>self.__generatorInput["musicLength"]:
			return GeneratorResult.INPUT_EORROR
		if self.__generatorInput["musicSectionBreakPointOne"]<=self.__generatorInput["musicET"] or  self.__generatorInput["musicSectionBreakPointTwo"]<=self.__generatorInput["musicSectionBreakPointOne"]:
			return GeneratorResult.INPUT_EORROR
		if self.__generatorInput["musicSectionBreakPointTwo"]>=self.__generatorInput["musicLength"]:
			return GeneratorResult.INPUT_EORROR
		dif =self.__generatorInput["muiscDifficult"]
		if dif<0 or dif >10:
			return GeneratorResult.INPUT_EORROR
		return GeneratorResult.SUCCESS

	def __InputDoumnetInfo(self):
		self.__document.LevelInfo["Difficulty"] = self.__generatorInput["muiscDifficult"]
		self.__document.LevelInfo["BPM"] =self.__generatorInput["musicBPM"]
		self.__document.LevelInfo["EnterTime"] = self.__generatorInput["musicET"]
		self.__document.MusicInfo["Title"] = self.__generatorInput["musicTitle"]
		# self.__document.MusicInfo["Artist"] = self.__generatorInput["musicArtist"]
		self.__document.MusicInfo["FilePath"]= self.__generatorInput["filePath"]
		return GeneratorResult.SUCCESS

	def __CalculateBasicInformation(self):

		self.__barTime = (60*1000)/self.__generatorInput["musicBPM"]*4
		self.__barAmount = math.floor((self.__generatorInput["musicLength"] -self.__generatorInput["musicET"])/self.__barTime)

		self.__sectionOneBarAmount = math.floor((self.__generatorInput["musicSectionBreakPointOne"] -self.__generatorInput["musicET"])/self.__barTime)
		self.__sectionTwoBarAmount = math.floor((self.__generatorInput["musicSectionBreakPointTwo"] -self.__sectionOneBarAmount*self.__barTime)/self.__barTime)
		self.__sectionThereBarAoumt =math.floor((self.__generatorInput["musicLength"] -(self.__sectionOneBarAmount+self.__sectionTwoBarAmount)*self.__barTime)/self.__barTime) -1 # 留出1个空白
		# print("bartime = ",self.__barTime,"BarAmount = ",self.__barAmount)
		# print("段落","[",self.__sectionOneBarAmount,self.__sectionTwoBarAmount,self.__sectionThereBarAoumt,"]")

		maxBarAmountForInTwoAndThere = max(self.__sectionTwoBarAmount,self.__sectionThereBarAoumt)
		halfMaxBar =math.floor(maxBarAmountForInTwoAndThere/2)
		maxSapeceBar = self.__sectionOneBarAmount -max(self.__ruler["mini_section_barAmount"],halfMaxBar)
		if maxSapeceBar<self.__ruler["begin_blank_space_barAmount"][0]:
			return GeneratorResult.SECTION_BAR_AMOUNT_SHORT

		minBarAmountForInTwoAndThere = min(self.__sectionTwoBarAmount,self.__sectionThereBarAoumt)
		doubleMaxBar =minBarAmountForInTwoAndThere*2
		minSpaceBar = self.__sectionOneBarAmount -doubleMaxBar
		# print ("good",minSpaceBar)

		if minSpaceBar >self.__ruler["begin_blank_space_barAmount"][1]:
			return GeneratorResult.SECTION_BAR_AMOUNT_SHORT


		if self.__sectionTwoBarAmount<self.__sectionThereBarAoumt and self.__sectionTwoBarAmount<math.floor(self.__sectionThereBarAoumt/2):
			return GeneratorResult.SECTION_BAR_AMOUNT_SHORT
		if self.__sectionThereBarAoumt<self.__sectionTwoBarAmount and self.__sectionThereBarAoumt<math.floor(self.__sectionTwoBarAmount/2):
			return GeneratorResult.SECTION_BAR_AMOUNT_SHORT

		self.__beginBlackSpaceBarAmount = random.randint(  max(minSpaceBar ,self.__ruler["begin_blank_space_barAmount"][0]),min(self.__ruler["begin_blank_space_barAmount"][1],maxSapeceBar))

		self.__sectionOneBarAmount = self.__sectionOneBarAmount -self.__beginBlackSpaceBarAmount

		if self.__sectionOneBarAmount<self.__ruler["mini_section_barAmount"] or self.__sectionTwoBarAmount<self.__ruler["mini_section_barAmount"] or self.__sectionThereBarAoumt< (self.__ruler["mini_section_barAmount"]+1):
			return GeneratorResult.SECTION_BAR_AMOUNT_SHORT

		# print("段落","[",self.__beginBlackSpaceBarAmount, self.__sectionOneBarAmount,self.__sectionTwoBarAmount,self.__sectionThereBarAoumt,"]")

		return GeneratorResult.SUCCESS

	def __FillActionClipInfomation(self):
		self.__actionBarCursor = 0
		self.__actionBarCursor = self.__beginBlackSpaceBarAmount

		self.__ActionGenerator(1,self.__sectionOneBarAmount,self.__document.DanceSegment_1)
		self.__ActionGenerator(2,self.__sectionTwoBarAmount,self.__document.DanceSegment_2)
		self.__ActionGenerator(3,self.__sectionThereBarAoumt,self.__document.DanceSegment_3)
		self.__ActionGenerator(4,1,self.__document.DanceSegment_4)

		self.__SegmentLevelGenerator(1,self.__document.DanceSegment_1)
		self.__SegmentLevelGenerator(2,self.__document.DanceSegment_2)
		self.__SegmentLevelGenerator(3,self.__document.DanceSegment_3)

	#segmentType 1 2 3 4 分别表示4个段落
	def __ActionGenerator(self,segmentType ,segBarAmount ,danceSegmentLst):

	    #填充空白动作
		if segmentType == 4:
			BlankAction = DanceClip()
			BlankAction.type ="Blank"
			BlankAction.level = "1"
			BlankAction.bar = self.__actionBarCursor
			danceSegmentLst.append(BlankAction)
			return

		#计算一个段落中动作长度为1的个数
		if segmentType == 1:
			if segBarAmount&1 == 0:
				oneLengthActionNum = 8
			else:
				oneLengthActionNum = 7
		else:
			if segBarAmount&1 == 0:
				oneLengthActionNum = 0
			else:
				oneLengthActionNum = 1

        # 填充动作长度为1的动作
		for index in range(0,oneLengthActionNum):
			oneLengthAction = DanceClip()
			oneLengthAction.ActionLength = 1
			oneLengthAction.bar = self.__actionBarCursor
			self.__actionBarCursor += oneLengthAction.ActionLength
			if index == 0:
				oneLengthAction.level = oneLengthAction.ActionLevel = 1

			else:
				oneLengthAction.ActionLevel = 3
			danceSegmentLst.append(oneLengthAction)

		#计算一个段落中动作长度为2的个数

		twoLengthActionNum = int((segBarAmount -oneLengthActionNum-4)/2)

		#填充动作长度为2的动作
		if segmentType == 1:
			actionLevelLst = self.__ActionLevelGenerator(self.__ruler["one_segment_actionLevel_range"],twoLengthActionNum)
		elif segmentType == 2:
			actionLevelLst = self.__ActionLevelGenerator(self.__ruler["two_segment_actionLevel_range"],twoLengthActionNum)
		elif segmentType == 3:
			actionLevelLst = self.__ActionLevelGenerator(self.__ruler["there_segment_actionLevel_range"],twoLengthActionNum)



		for index in range(0,twoLengthActionNum ):
			twoLengthAction = DanceClip()
			twoLengthAction.ActionLength = 2
			twoLengthAction.bar = self.__actionBarCursor
			self.__actionBarCursor += twoLengthAction.ActionLength
			twoLengthAction.ActionLevel =actionLevelLst[index]  
			danceSegmentLst.append(twoLengthAction)

 		#填充动作长度为4的动作
		fourLengthAction = DanceClip()
		fourLengthAction.ActionLength = 4
		fourLengthAction.ActionLevel = 9
		fourLengthAction.bar = self.__actionBarCursor
		self.__actionBarCursor += fourLengthAction.ActionLength
		danceSegmentLst.append(fourLengthAction)

	def __ActionLevelGenerator(self,actionLevelRange,actionNum):
		actionLevelLst =[]

		levelNum = len(actionLevelRange)
		oneNum =  math.floor(actionNum * self.__ruler["actionLevel_weight_1"])
		for i in range(0,oneNum):
			actionLevelLst.append(actionLevelRange[0])


		twoNum = math.ceil(actionNum * self.__ruler["actionLevel_weight_2"])
		for i in range(0,twoNum):
			actionLevelLst.append(actionLevelRange[1])

		if levelNum ==3:
			thereNum = actionNum -oneNum-twoNum
			for i in range(0,thereNum):
				actionLevelLst.append(actionLevelRange[2])

		else:
			thereNum = math.ceil(actionNum * self.__ruler["actionLevel_weight_3"])
			for i in range(0,thereNum):
				actionLevelLst.append(actionLevelRange[2])

			fourNum = actionNum -oneNum-twoNum-thereNum
			for i in range(0,fourNum):
				actionLevelLst.append(actionLevelRange[3])
				
		return actionLevelLst

	def __SegmentLevelGenerator(self,segmentType,danceList):
		dif = self.__document.LevelInfo["Difficulty"] 
		danceLen = len(danceList)
		segment_ruler = self.__ruler["level"]["dif"+str(dif)]["Segment_"+str(segmentType)]
		# print(segment_ruler,"dif"+str(dif))
		beginArea = segment_ruler["begin"]
		endArea = segment_ruler["end"]

		area_1_range_num = segment_ruler["area_1"]["range_num"]
		area_1_range_weights = segment_ruler["area_1"]["weights"]
		area_1_weight = segment_ruler["area_1"]["weight"]

		area_2_range_num = segment_ruler["area_2"]["range_num"]
		area_2_range_weights = segment_ruler["area_2"]["weights"]
		area_2_weight = segment_ruler["area_2"]["weight"]

		beginLen = len(beginArea)
		rn_1 = len(area_1_range_num)
		rn_2 = len(area_2_range_num)
		endLen = len(endArea)

		levelList =[]

		minLen = rn_1 + rn_2+beginLen +endLen

		if minLen >=danceLen:
			levelList += beginArea
			for item in area_1_range_num:
				levelList.append(item)
			for item in area_2_range_num:
				levelList.append(item)

			levelList +=endArea
			for i in range(danceLen):
				danceList[i].level = levelList[i]
			if minLen>danceLen:
				self.__Logger.info("some level may be lost "+str(minLen)+" "+str(danceLen))
			return


		otherLen = danceLen -minLen

		area_1_num = round(otherLen *area_1_weight)
		area_2_num = otherLen -area_1_num


		levelList += beginArea

		# print(danceLen,area_1_num,area_2_num,area_1_weight)
		levelList += self.__rangeLevelGenerator(area_1_range_num,area_1_range_weights,area_1_num)

		levelList += self.__rangeLevelGenerator(area_2_range_num,area_2_range_weights,area_2_num)

		levelList += endArea


		# news_ids = list(set(levelList))
		# news_ids.sort()
		# print(news_ids)
		# if segmentType == 1 and len(news_ids)!=9:
		# 	print("________________________",levelList)

		# print (levelList,len(levelList),danceLen)
		for i in range(danceLen):
			danceList[i].level = levelList[i]


	def __rangeLevelGenerator(self,range_num,weights,num):
		levelList =[]
		# print(range_num,weights)
		for i in range(len(range_num)-1):
			rn = round(num * weights[i])
			for k in range(rn):
				levelList.append(range_num[i])
			levelList.append(range_num[i])

		rangeLen = len(range_num)

		lastNum =rangeLen + num -len(levelList)

		for i in range(lastNum):
			levelList.append(range_num[rangeLen-1])


		# levelList = levelList[:num]
		
		return levelList


	def __FillCameraClipInfomation(self):
		self.__special_shot_container = []
		for special_shot_pair in self.__ruler ["camera_shot"]["special_shot_container"]:
			self.__special_shot_container.append(special_shot_pair)

		firstShotClip = CameraClip()
		firstShotClip.id ="front_01"
		firstShotClip.bar = 4
		self.__document.CameraSequence.append(firstShotClip)

		segmentOneEndShotClip = CameraClip()
		segmentOneEndShotClip.id ="right_04"
		segmentOneEndShotClip.bar = self.__beginBlackSpaceBarAmount+self.__sectionOneBarAmount
		self.__FillSegmentCameraClip(1,firstShotClip, segmentOneEndShotClip)
		self.__document.CameraSequence.append(segmentOneEndShotClip)

		segmentTwoEndShotClip = CameraClip()
		segmentTwoEndShotClip.id ="left_04"
		segmentTwoEndShotClip.bar = segmentOneEndShotClip.bar +self.__sectionTwoBarAmount
		self.__FillSegmentCameraClip(2,segmentOneEndShotClip ,segmentTwoEndShotClip)
		self.__document.CameraSequence.append(segmentTwoEndShotClip)

		segmentThereEndShotClip = CameraClip()
		segmentThereEndShotClip.id ="front_04"
		segmentThereEndShotClip.bar = segmentTwoEndShotClip.bar+self.__sectionThereBarAoumt
		self.__FillSegmentCameraClip(3,segmentTwoEndShotClip,segmentThereEndShotClip)

		self.__document.CameraSequence.append(segmentThereEndShotClip)

	def __FillSegmentCameraClip(self,segmentType,beginClip,endClip):
		self.__resetCameraContainer()
		segmentFirst_01_shot  =CameraClip()
		if segmentType == 1:
			segmentFirst_01_shot = beginClip
		elif segmentType == 2:
			segmentFirst_01_shot.bar = beginClip.bar+1
			segmentFirst_01_shot.id = "left_01"
			self.__document.CameraSequence.append(segmentFirst_01_shot)
		elif segmentType == 3:
			segmentFirst_01_shot.bar = beginClip.bar+1
			segmentFirst_01_shot.id = "right_01"
			self.__document.CameraSequence.append(segmentFirst_01_shot)

		BPM = self.__document.LevelInfo["BPM"]
		special_shot_length = self.__ruler["special_shot_length_range"]
		if BPM<120:
			move_shot_length = self.__ruler["bpm120_move_shot_length_range"]
			fix_shot_length = self.__ruler["bpm150_fix_shot_length_range"]
		elif BPM<150:
			move_shot_length = self.__ruler["bpm120_150_move_shot_length_range"]
			fix_shot_length = self.__ruler["bpm120_150_fix_shot_length_range"]
		else:
			move_shot_length = self.__ruler["bpm150_move_shot_length_range"]
			fix_shot_length = self.__ruler["bpm150_fix_shot_length_range"]

		# print(move_shot_length,special_shot_length,fix_shot_length)
		curEndBar = endClip.bar
		curBeginBar = segmentFirst_01_shot.bar
		curSpaceBarNum = curEndBar -curBeginBar

		#添加移动镜头
		min_shot_space_bar_num =  math.floor(curSpaceBarNum/2)
		move_shot =CameraClip()
		move_shot.bar = endClip.bar -  min( random.randint(move_shot_length[0],move_shot_length[1]),min_shot_space_bar_num )+1
		move_shot_container = self.__ruler["camera_shot"]["move_shot_container"][endClip.id]
		move_shot_index = random.randint(0,len(move_shot_container)-1)
		move_shot.id = move_shot_container[move_shot_index]
		

		curEndBar = move_shot.bar
		curSpaceBarNum = curEndBar -curBeginBar
		# 添加特效镜头
		if curSpaceBarNum  >=special_shot_length[1]*2+fix_shot_length*2 and segmentType != 3: 
			special_shot_container_index = random.randint(0,len(self.__special_shot_container)-1)
			special_shot_pair = self.__special_shot_container[special_shot_container_index]
			special_shot_1 =CameraClip()
			special_shot_1.bar = curBeginBar + fix_shot_length
			curBeginBar = special_shot_1.bar
			special_shot_1.id = special_shot_pair[0]
			special_shot_2 =CameraClip()
			special_shot_2.bar = curBeginBar + random.randint(special_shot_length[0],special_shot_length[1])
			special_shot_2.id = special_shot_pair[1]
			curBeginBar = special_shot_2.bar
			self.__special_shot_container.remove(special_shot_pair)
			self.__document.CameraSequence.append(special_shot_1)
			self.__document.CameraSequence.append(special_shot_2)
			specailLength = random.randint(special_shot_length[0],special_shot_length[1])
			curBeginBar = special_shot_2.bar + specailLength
		else:
			curBeginBar = curBeginBar +fix_shot_length



		# 添加固定镜头

		curEndClip = self.__document.CameraSequence[len(self.doument.CameraSequence)-1]
		beginID = curEndClip.id
		endID = move_shot.id
		curSpaceBarNum =curEndBar -curBeginBar

		if curSpaceBarNum>0:
			# print(curSpaceBarNum,curBeginBar,curEndBar, fix_shot_length)
			fixedShotLengthNum = round(curSpaceBarNum/fix_shot_length)
			if fixedShotLengthNum>0:
				fixedShotBarNum =math.ceil(curSpaceBarNum/fixedShotLengthNum)
				for i in range(0,fixedShotLengthNum-1):
					new_fix_shot = CameraClip()
					new_fix_shot.bar = curBeginBar
					new_fix_shot.id = self.__GetCamerShot(beginID)
					beginID = new_fix_shot.id
					self.__document.CameraSequence.append(new_fix_shot)
					curBeginBar  =new_fix_shot.bar+ fixedShotBarNum

				new_fix_segEnd_shot = CameraClip()
				new_fix_segEnd_shot.bar = curBeginBar
				new_fix_segEnd_shot.id = self.__GetCamerShot(beginID,endID)    # 段落中最后一个固定镜头
				self.__document.CameraSequence.append(new_fix_segEnd_shot)


		self.__document.CameraSequence.append(move_shot)

	def __resetCameraContainer(self):

		self.__camer_get_index=0
		self.__fixed_camera_shot_container =[]
		for cameraShot in self.__ruler["camera_shot"]["fixed_shot_container"]:
			self.__fixed_camera_shot_container.append(cameraShot)

	def __GetCamerShot(self,shot_A,shot_B=None):
		if self.__camer_get_index>=3:
			self.__resetCameraContainer()
		else:
			self.__camer_get_index += 1

		distancesA = [ self.__DistanceCamerShot(shot_A,p) for p in self.__fixed_camera_shot_container ]
		if shot_B is None:
			index = distancesA.index(max(distancesA))
			result  = self.__fixed_camera_shot_container[index]
		else:
			distancesB = [ self.__DistanceCamerShot(p,shot_B) for p in self.__fixed_camera_shot_container ]
			distancesN = [a*b for a,b in zip(distancesA,distancesB)]
			# print(self.__fixed_camera_shot_container,shot_A,shot_B)
			# print(self.__document.LevelInfo["Difficulty"], distancesA,distancesB,distancesN)
			index = distancesN.index(max(distancesN))
			result  = self.__fixed_camera_shot_container[index]

		del self.__fixed_camera_shot_container[index]
		return result

	def __DistanceCamerShot(self,shot_A,shot_B):
		# print ("C ",shot_A,shot_B)
		adata = shot_A.split("_")
		bdata = shot_B.split("_")
		if len(adata)== 4:
			shot_A = adata[2]+"_02"
		if len(bdata)== 4:
			shot_B = bdata[0]+"_"+bdata[1]

		if "special" in shot_A:
			pos_A = self.__ruler["camera_shot"]["special_shot_container_pos"]["x"+shot_A[-1:]]
		else:
			pos_A = self.__ruler["camera_shot"]["fixed_shot_container_pos"][shot_A]

		if "special" in shot_B:
			pos_B = self.__ruler["camera_shot"]["special_shot_container_pos"]["x"+shot_B[-1:]]
		else:
			pos_B = self.__ruler["camera_shot"]["fixed_shot_container_pos"][shot_B]

		# print (pos_A,shot_A,pos_B,shot_B,abs(pos_A[0]-pos_B[0])*abs(pos_A[1]-pos_B[1]))
		
		return abs(pos_A[0]-pos_B[0])*abs(pos_A[1]-pos_B[1])

#规则配置
GeneratorRuler = dict(
	mini_music_length = 165000,    						#歌曲时长范围最短不能短于2分45秒
	max_music_length = 3600000,    						#最长不能超过六分钟
	mini_section_barAmount  = 16,  						#段落最短不能低于16小节
	begin_blank_space_barAmount = (5,11),        		#关卡的第一个动作前必须空出不少于4小节、不大于12小节的歌曲

	one_segment_actionLevel_range = (4,5,6),			#情绪为4，5，6
	two_segment_actionLevel_range = (4,5,6,7),			#情绪为4，5，6，7
	there_segment_actionLevel_range = (5,6,7,8),     	#情绪为5，6，7，8 
	actionLevel_weight_1 = 0.25,               			#前两个比重>= 60%
	actionLevel_weight_2 = 0.35,						#前两个比重>= 60%
	actionLevel_weight_3 = 0.16,

    level = dict(
    	dif1=dict(
    		Segment_1 = dict(
    			begin=[1,1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.7
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.5,0.3,0.2),
    				weight= 0.3
    			),
    			end =[9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.7
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.9,0.1),
    				weight= 0.3
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.7
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.8,0.2),
    				weight= 0.3
    			),
    			end =[],
    			)
    		),
    	dif2=dict(
    		Segment_1 = dict(
    			begin=[1,1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.65
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.45,0.4,0.15),
    				weight= 0.35
    			),
    			end =[9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.65
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.85,0.15),
    				weight= 0.35
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.65
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.75,0.25),
    				weight= 0.35
    			),
    			end =[],
    			)
    		),
    	dif3=dict(
    		Segment_1 = dict(
    			begin=[1,1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.6
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.40,0.45,0.15),
    				weight= 0.4
    			),
    			end =[9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.65
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.80,0.20),
    				weight= 0.35
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.60
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.70,0.30),
    				weight= 0.40
    			),
    			end =[],
    			)
    		),
    	dif4=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.55
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.35,0.50,0.15),
    				weight= 0.45
    			),
    			end =[9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.55
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.75,0.25),
    				weight= 0.45
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.55
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.65,0.35),
    				weight= 0.45
    			),
    			end =[],
    			)

    		),
    	dif5=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.5
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.3,0.50,0.2),
    				weight= 0.5
    			),
    			end =[9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.50
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.70,0.30),
    				weight= 0.50
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.50
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.60,0.40),
    				weight= 0.50
    			),
    			end =[],
    			)
    		),
    	dif6=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.45
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.25,0.50,0.25),
    				weight= 0.55
    			),
    			end =[9,9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.45
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.65,0.35),
    				weight= 0.55
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.45
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.55,0.45),
    				weight= 0.55
    			),
    			end =[],
    			)
    		),
    	dif7=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.4
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.20,0.50,0.30),
    				weight= 0.6
    			),
    			end =[9,9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(4,5,6,7),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.4
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.60,0.40),
    				weight= 0.6
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.4
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.50,0.50),
    				weight= 0.6
    			),
    			end =[],
    			)
    		),
    	dif8=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.35
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.20,0.45,0.35),
    				weight= 0.65
    			),
    			end =[9,9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(5,6,7),
	    			weights = (0.33,0.33,0.33),
	    			weight = 0.35
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.55,0.45),
    				weight= 0.65
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.35
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.45,0.55),
    				weight= 0.65
    			),
    			end =[],
    			)
    		),
    	dif9=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.3
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.20,0.40,0.40),
    				weight= 0.7
    			),
    			end =[9,9,9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(5,6,7),
	    			weights = (0.33,0.33,0.33),
	    			weight = 0.3
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.50,0.50),
    				weight= 0.7
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.3
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.4,0.6),
    				weight= 0.7
    			),
    			end =[],
    			)
    		),
    	dif10=dict(
    		Segment_1 = dict(
    			begin=[1],
    			area_1=dict(
	    			range_num =(2,3,4,5),
	    			weights = (0.25,0.25,0.25,0.25),
	    			weight = 0.25
    			),
    			area_2=dict(
    				range_num =(6,7,8),
    				weights = (0.20,0.35,0.45),
    				weight= 0.75
    			),
    			end =[9,9,9],
    			),
    		Segment_2 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(5,6,7),
	    			weights = (0.33,0.33,0.33),
	    			weight = 0.25
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.45,0.55),
    				weight= 0.75
    			),
    			end =[],
    			),
    		Segment_3 = dict(
    			begin=[],
    			area_1=dict(
	    			range_num =(6,7),
	    			weights = (0.5,0.5),
	    			weight = 0.25
    			),
    			area_2=dict(
    				range_num =(8,9),
    				weights = (0.35,0.65),
    				weight= 0.75
    			),
    			end =[],
    			)
    		),
    	),

    #特写镜头应当在歌曲比较舒缓的部分加入，一首歌中不宜增加太多，以一到两处，每处变换两个镜头总共4-8小节为宜，且特写镜头不添加于第三段落
    special_shot_length_range =(2,4), 
	# 120BPM以下：固定镜头4小节，移动镜头4～8小节均可；
	# 120BPM以上：固定镜头8小节，移动镜头8小节
	# 150BPM以上：固定镜头8小节，移动镜头可以用16小节。 
	bpm120_move_shot_length_range =(4,8),
	bpm120_150_move_shot_length_range = (8,8),
	bpm150_move_shot_length_range = (8,16),

	bpm120_fix_shot_length_range =4,
	bpm120_150_fix_shot_length_range =8,
	bpm150_fix_shot_length_range = 8,

	camera_shot=dict(
		long_shot =("left_04","front_04","right_04"),
		clos_ups_shot =("left_02","front_02","right_02"),
        fixed_shot_container_pos = dict(
        	left_02=(-1,1),
			front_02=(0,1),
			right_02=(1,1),
			left_01=(-1,0),
			front_01=(0,0),
			right_01=(1,0),
			left_03=(-1,-1),
			front_03=(0,-1),
			right_03=(1,-1),
        	),

		special_shot_container_pos = dict(
			x6=(-1,1),
			x4=(0,1),
			x5=(1,1),
			x3=(-1,0),
			x1=(0,0),
			x2=(1,0),
			x9=(-1,-1),
			x7=(0,-1),
			x8=(1,-1),
		),
			
		fixed_shot_container =(
			"left_02",
			"front_02",
			"right_02",
			"left_01",
			# "front_01",                            # front_01 只出现在第五小节
			"right_01",
			"left_03",
			"front_03",
			"right_03",
			),

		special_shot_container =(  						#特写镜头 每处变换两个镜头总共4-8小节为宜

				("special_18","special_23"),
				("special_18","special_26"),
				("special_18","special_21"),
				("special_18","special_24"),

				("special_12","special_26"),
				("special_12","special_24"),
				("special_12","special_29"),
				("special_12","special_27"),

				("special_15","special_23"),
				("special_15","special_29"),
				("special_15","special_21"),
				("special_15","special_27"),

				("special_19","special_23"),
				("special_19","special_29"),
				("special_19","special_21"),
				("special_19","special_27"),

				("special_13","special_24"),
				("special_13","special_25"),
				("special_13","special_27"),
				("special_13","special_28"),

				("special_16","special_21"),
				("special_16","special_27"),
				("special_16","special_22"),
				("special_16","special_28"),

				("special_17","special_22"),
				("special_17","special_25"),
				("special_17","special_23"),
				("special_17","special_26"),

				("special_11","special_25"),
				("special_11","special_28"),
				("special_11","special_29"),
				("special_11","special_26"),

				("special_14","special_22"),
				("special_14","special_28"),
				("special_14","special_23"),
				("special_14","special_29"),
			),
	    move_shot_container = dict(                            #移动镜头
	    	front_04=
				(
				"left_01_front_04",
				"right_01_front_04",
				"left_03_front_04",
				"right_03_front_04"
				),
			left_04 =
				(
				# "front_01_left_04",
				"right_01_left_04",
				"front_03_left_04",
				"right_03_left_04"
				),
	        right_04 = 	
				(
				# "front_01_right_04",
				"left_01_right_04",
				"front_03_right_04",
				"left_03_right_04"
				)
			)
		)

	)

#test 输入
GeneratorInput = dict(
	musicLength = 240*1000,
	musicBPM = 120,
	musicET = 0,
	musicSectionBreakPointOne = 80*1000,
	musicSectionBreakPointTwo = 160*1000,
	muiscDifficult= 10,
	musicTitle ="xxx",
	filePath="Media\\audio\\Music\\song_1000.ogg"
	)

def GenerateLevelFile(filename,musicInfo,logger):

	GeneratorInput["musicLength"] =musicInfo["duration"]*1000
	GeneratorInput["musicBPM"] = musicInfo["bpm"]
	GeneratorInput["musicET"] =musicInfo["EnterTime"]*1000
	GeneratorInput["musicSectionBreakPointOne"]=musicInfo["seg0"]*1000
	GeneratorInput["musicSectionBreakPointTwo"]=musicInfo["seg1"]*1000
	GeneratorInput["muiscDifficult"] = musicInfo["diffculty"]

	GeneratorInput["musicTitle"] = os.path.splitext( os.path.basename(filename))[0]

	creator =  Generator(GeneratorInput,GeneratorRuler,logger)
	result =creator.run()
	if result is  GeneratorResult.SUCCESS:
		convertor = XmlConverter(creator.doument)
		convertor.run()
		convertor.writeToFile(filename)
		logger.info(result)
	else:
		logger.error(result.value,result)




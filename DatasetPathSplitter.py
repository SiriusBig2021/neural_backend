import os
import random
import shutil




class DatasetPathSplitter:

    def __init__(self, folder, percentTrainImg, outputTrainDir, outputTestDir):

        if not os.path.exists(outputTrainDir):
            raise FileNotFoundError(outputTrainDir + ' is not exists')

        if not os.path.exists(outputTestDir):
            raise FileNotFoundError(outputTestDir + ' is not exists')

        self.outputTrainDir = outputTrainDir
        self.outputTestDir = outputTestDir

        imagesFolder = sorted(os.listdir(folder), key=lambda *args: random.random())  # shuffle img
        numbersImg = len(imagesFolder)

        if numbersImg == 0:
            raise FileNotFoundError(folder + ' is empty')

        #print(numbersImg)
        numbersImgForTrain = round(numbersImg * percentTrainImg)
        #print(numbersImgForTrain)

        for i in range(numbersImgForTrain):
            shutil.move(folder + '/' + imagesFolder[i], outputTrainDir)
        for i in range(numbersImgForTrain, numbersImg):
            shutil.move(folder + '/' + imagesFolder[i], outputTestDir)

    def getTrainFolder(self):
        if not os.path.exists(self.outputTrainDir):
            raise FileNotFoundError(self.outputTrainDir + ' is not exists')

        return self.outputTrainDir

    def getTestFolder(self):
        if not os.path.exists(self.outputTestDir):
            raise FileNotFoundError(self.outputTestDir + ' is not exists')

        return self.outputTestDir


# path = '/home/sauce-chili/Sirius/neural_backend/data/videos'
# outTrain = '/home/sauce-chili/Sirius/neural_backend/data/outTrain'
# outTest = '/home/sauce-chili/Sirius/neural_backend/data/outTest'
#
# pathSplitter = DatasetPathSplitter(path, 0.75, outTrain, outTest)
#
# print(pathSplitter.getTrainFolder())
# print(pathSplitter.getTestFolder())

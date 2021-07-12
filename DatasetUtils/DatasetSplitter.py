import os
import random

class DatasetSplitter:

    def __init__(self, datasetMetaFile, percentTrainImg, outTrainMetaFile, outTestMetaFile):

        if not os.path.exists(datasetMetaFile):
            raise FileExistsError(datasetMetaFile + ' is not exists')

        if not os.path.exists(outTrainMetaFile):
            f = open(outTrainMetaFile, 'r')
            f.close()
            print('[WARNING] ' + outTrainMetaFile + ' is not exists.The folder was created.')
            # raise FileExistsError(outputTrainDir + ' is not exists')

        if not os.path.exists(outTestMetaFile):
            f = open(outTestMetaFile, 'r')
            f.close()
            print('[WARNING] ' + outTestMetaFile + ' is not exists.The folder was created.')
            # raise FileNotFoundError(outputTestDir + ' is not exists')

        self.__datasetMetaFile = datasetMetaFile
        self.__outTrainMetaFile = outTrainMetaFile
        self.__outTestMetaFile = outTestMetaFile
        self.__percentTrainImg = percentTrainImg

    def split(self):

        notesMetaFile = []
        numbersNoteInMetaFile = 0
        with open(self.__datasetMetaFile, 'r') as dmf:
            notesMetaFile = dmf.readlines()
            numbersNoteInMetaFile = len(notesMetaFile)

        notesMetaFile = sorted(notesMetaFile, key=lambda *args: random.random())  # shuffle notes
        numbersImgForTrain = round(numbersNoteInMetaFile * self.__percentTrainImg)

        if numbersNoteInMetaFile == 0:
            print('[WARNING] Folder ' + self.__datasetMetaFile + ' is empty')
            return

        with open(self.__outTrainMetaFile, 'w') as train_mf:
            for i in range(numbersImgForTrain):
                if (i + 1) == numbersImgForTrain:
                    train_mf.write(notesMetaFile[i][:-1])
                else:
                    train_mf.write(notesMetaFile[i])

        with open(self.__outTestMetaFile, 'w') as test_mf:
            for i in range(numbersImgForTrain, numbersNoteInMetaFile):
                if (i + 1) == numbersNoteInMetaFile:
                    test_mf.write(notesMetaFile[i][:-1])
                else:
                    test_mf.write(notesMetaFile[i])

    def getPathTrainFolder(self):
        if not os.path.exists(self.__outTrainMetaFile):
            return None

        return self.__outTrainMetaFile

    def getPathTestFolder(self):
        if not os.path.exists(self.__outTestMetaFile):
            return None

        return self.__outTestMetaFile




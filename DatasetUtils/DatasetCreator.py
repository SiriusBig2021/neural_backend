import cv2
import os
from DatasetUtils.DatasetSplitter import DatasetSplitter


class DatasetCreator:
    __DEBUG = True

    def __init__(self, absoluteVideoPath, absolutePathOutFolder, absolutePathToMetaFile, labels: dict,
                 splitter: DatasetSplitter = None):

        if not os.path.exists(absoluteVideoPath):
            raise FileExistsError(absoluteVideoPath + ' is not exists')

        if not os.path.exists(absolutePathToMetaFile):
            os.mkdir(absolutePathToMetaFile)
            print('[WARNING] ' + absolutePathToMetaFile + ' is not exists.The folder was created.')

        self.videoPath = absoluteVideoPath
        self.outFolder = absolutePathOutFolder
        self.metaFile = absolutePathToMetaFile
        self.labels = labels
        self.splitter = splitter

    def create(self):

        if not os.path.exists(self.outFolder):
            os.mkdir(self.outFolder)
            print('[WARNING] ' + self.outFolder + ' is not exists.The folder was created.')

        cam = cv2.VideoCapture(self.videoPath)
        videoname = self.videoPath.split("/")[-1].split(".")[0]

        with open(self.metaFile, 'a') as mf:
            while cam.isOpened():
                grabbed, frame = cam.read()

                frameid = int(cam.get(cv2.CAP_PROP_POS_FRAMES))

                if not grabbed:
                    break

                name = videoname + '_' + str(frameid) + '.jpg'
                path = os.path.join(self.outFolder, name)
                cv2.imwrite(path, frame)

                note = name + ' ' + self.labels['class'] + ' ' + self.labels['status'] + "\n"

                mf.write(note)

                if self.__DEBUG:
                    print('[INFO] recorded: ' + note)

        cam.release()

        if self.splitter is not None:
            self.splitter.split()
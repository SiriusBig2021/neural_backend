from DenseOpticalFlow import DenseOpticalFlow
import cv2

import glob
import os

from datetime import datetime


class VideoCutter:

    def __init__(self, outputPath, opt_param: dict, inputPath=None):

        if inputPath is None:
            self.handlerFolder = []
        else:
            if not os.path.exists(inputPath):
                raise FileExistsError(inputPath + 'is not exists')

        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
            print('[WARNING] ' + outputPath + ' is not exists.The folder was created.')

        self.handlerFolder = []
        self.optflow = DenseOpticalFlow(opt_param=opt_param)
        self.outPath = outputPath
        self.videoWriter = None

    def __initVideoWriter(self, nameVideoFile, fourcc, fps, size):
        self.videoWriter = cv2.VideoWriter(nameVideoFile, fourcc, fps, size)

    def __destroyVideoWriter(self):
        self.videoWriter.release()
        self.videoWriter = None

    def setHandlerFolder(self, absolutePath):
        self.handlerFolder = [videoPath for videoPath in glob.glob(absolutePath + '*.mp4')]

    def addVideoInHandlerFolder(self, absolutePathToVideo):
        self.handlerFolder.append(absolutePathToVideo)

    def execute(self):
        for videoPath in self.handlerFolder:
            cam = cv2.VideoCapture(videoPath)
            print('[INFO] Current video: ' + videoPath)
            while cam.isOpened():
                grabbed, frame = cam.read()
                if not grabbed:
                    print('[INFO] video ' + videoPath + ' is not grabbed')
                    break

                direction = self.optflow.getMoveDirection(frame)
                print('[INFO] direction: ' + direction)

                cv2.imshow(videoPath, frame)
                cv2.waitKey(1)

                if direction != 'wait':
                    if self.videoWriter is None:
                        file = videoPath.split('/')[-1]
                        fileName = file.split('.')[0]
                        fileSuffix = file.split('.')[1]
                        name = fileName + '_' \
                               + datetime.now().strftime("%d-%m-%Y-%H:%M") \
                               + '.' + fileSuffix
                        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                        fps = cam.get(cv2.CAP_PROP_FPS)
                        size = frame.shape[:2][::-1]
                        self.__initVideoWriter(self.outPath + name, fourcc, fps, size)

                    self.videoWriter.write(frame)

                elif direction == 'wait':
                    if self.videoWriter is not None:
                        self.__destroyVideoWriter()

            cam.release()
            cv2.destroyWindow(videoPath)

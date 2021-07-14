from DenseOpticalFlow import DenseOpticalFlow
import cv2

import glob
import os

from datetime import datetime


class VideoCutter:
    __amountJumpByFrameLine = 25

    def __init__(self, outputPath, opt_param: dict, inputPath=None):

        if inputPath is None:
            self.handlerFolder = []
        else:
            if not os.path.exists(inputPath):
                raise FileExistsError(inputPath + 'is not exists')
            else:
                self.setHandlerFolder(inputPath)

        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
            print('[WARNING] ' + outputPath + ' is not exists.The folder was created.')

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

    def __getAccumulatedFrames(self, cam) -> (list, None):

        frameList = []

        while cam.isOpened:
            grabbed, frame = cam.read()

            if not grabbed:
                break

            frameList.append(frame)

        if len(frameList) == 0:
            return None

        return frameList

    def execute(self):

        for videoPath in self.handlerFolder:

            cam = cv2.VideoCapture(videoPath)

            frameList = self.__getAccumulatedFrames(cam)

            if frameList is None:
                print('[WARNING] Failed to collect video. Video will be skipped.')
                continue

            print('[INFO] Current video: ' + videoPath)

            nbFrames = len(frameList)
            counterFrame = 0

            while counterFrame < nbFrames:

                print('[INFO] Frame line: [' + str(counterFrame) + '/' + str(nbFrames - 1) + ']')

                key = cv2.waitKey(33)

                if key == ord('s'):
                    print('[INFO] skipping video')
                    break

                elif key == ord('e'):

                    self.optflow.clearHash()
                    if counterFrame + self.__amountJumpByFrameLine >= nbFrames - 1:
                        counterFrame = nbFrames - 1
                    else:
                        counterFrame += self.__amountJumpByFrameLine

                elif key == ord('q'):

                    self.optflow.clearHash()
                    if counterFrame - self.__amountJumpByFrameLine <= 0:
                        counterFrame = 0
                    else:
                        counterFrame -= self.__amountJumpByFrameLine

                direction = self.optflow.getMoveDirection(frameList[counterFrame][140:, :])
                # print('[INFO] direction: ' + direction)

                cv2.imshow('In Opt', frameList[counterFrame][140:, :])
                cv2.imshow(videoPath, frameList[counterFrame])
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
                        size = frameList[counterFrame].shape[:2][::-1]
                        self.__initVideoWriter(self.outPath + name, fourcc, fps, size)

                    self.videoWriter.write(frameList[counterFrame])

                elif direction == 'wait':
                    if self.videoWriter is not None:
                        self.__destroyVideoWriter()

                counterFrame += 1

            cam.release()
            cv2.destroyWindow(videoPath)
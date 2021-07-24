from DenseOpticalFlow import DenseOpticalFlow
import cv2

import glob
import os

from datetime import datetime

opt_param = {
    # Не лезь если ты не Женя
    'threshold_magnitude': 6,  # пороговая амплитуда движения
    'size_accumulation': 6,  # кол-во накапливаемых изображения
    # Не лезть
    'opticflow_param': {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 100,
        'iterations': 4,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
    }
}


class VideoCutter:
    __amountJumpByFrameLine = 30
    __amountJumpMaxByFrameLine = 600

    def __init__(self, outputPath, opt_param: dict, useTools: bool = True, inputPath=None):

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
        self.useTools = useTools
        self.videoWriter = None

    def __initVideoWriter(self, nameVideoFile, fourcc, fps, size):
        self.videoWriter = cv2.VideoWriter(nameVideoFile, fourcc, fps, size)

    # def __initVideoWriter(self, camReader, size, currentVideoPath):
    #     file = currentVideoPath.split('/')[-1]
    #     fileName = file.split('.')[0]
    #     fileSuffix = file.split('.')[1]
    #     name = fileName + '_' \
    #            + datetime.now().strftime("%d-%m-%Y-%H:%M") \
    #            + '.' + fileSuffix
    #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #     fps = camReader.get(cv2.CAP_PROP_FPS)
    #     self.videoWriter = cv2.VideoWriter(self.outPath + name, fourcc, fps, size)

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

            nbFrames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
            indFrame = 0

            while cam.isOpened() and indFrame < nbFrames:

                ####################### Block handle press ##########################
                key = cv2.waitKey(33)

                # skip video
                if self.useTools and key == ord('n'):
                    print('[INFO] skipping video')
                    break

                # roll video on amountJumpByFrameLine
                elif self.useTools and key == ord('e'):

                    self.optflow.clearСash()
                    if indFrame + self.__amountJumpByFrameLine >= nbFrames - 1:
                        indFrame = nbFrames - 1
                    else:
                        indFrame += self.__amountJumpByFrameLine

                # roll back video on amountJumpByFrameLine
                elif key == ord('q') and self.useTools:

                    self.optflow.clearСash()
                    if indFrame - self.__amountJumpByFrameLine <= 0:
                        indFrame = 0
                    else:
                        indFrame -= self.__amountJumpByFrameLine

                # roll video on amountJumpMaxByFrameLine
                elif key == ord('d') and self.useTools:

                    self.optflow.clearСash()
                    if indFrame + self.__amountJumpMaxByFrameLine >= nbFrames - 1:
                        indFrame = nbFrames - 1
                    else:
                        indFrame += self.__amountJumpMaxByFrameLine

                # roll back video on amountJumpMaxByFrameLine
                elif key == ord('a') and self.useTools:

                    self.optflow.clearСash()
                    if indFrame - self.__amountJumpMaxByFrameLine <= 0:
                        indFrame = 0
                    else:
                        indFrame -= self.__amountJumpMaxByFrameLine

                ####################################################################

                print('[INFO] Frame line: [' + str(indFrame) + '/' + str(nbFrames - 1) + ']')

                # set current frame by index
                cam.set(cv2.CAP_PROP_POS_FRAMES, indFrame)
                grabbed, frame = cam.read()

                if not grabbed:
                    print('[INFO] Failed to capture frame')
                    break

                direction = self.optflow.getMoveDirection(frame[140:, :])
                # print('[INFO] direction: ' + direction)

                if self.useTools:
                    cv2.imshow('In Opt', frame[140:, :])
                    cv2.imshow(videoPath, frame)
                cv2.waitKey(1)

                if direction != 'wait':

                    # if self.videoWriter is None:
                    #     self.__initVideoWriter(camReader=cam,
                    #                            size=frame[:2][::-1],
                    #                            currentVideoPath=videoPath)

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

                indFrame += 1

            cam.release()
            cv2.destroyWindow(videoPath)


#inp = '/home/sauce-chili/Sirius/neural_backend/data/archive/mid1/'
outp = '/home/sauce-chili/Sirius/neural_backend/data/VideoWithMove/mid_1/'

pathToVideo = '/home/sauce-chili/Sirius/neural_backend/data/archive/mid1/mid1_14-07-2021_19:37:41.mp4dddddd'
# next
# /home/sauce-chili/Sirius/neural_backend/data/archive/mid1/mid1_13-07-2021_11:31:02.mp4

vc = VideoCutter(outputPath=outp, opt_param=opt_param)
vc.addVideoInHandlerFolder(pathToVideo)
vc.execute()

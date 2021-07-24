import cv2
from scipy.stats import mode

import numpy as np


# opt_param = {
#     'threshold_magnitude': 6,
#     'size_accumulation': 6,
#     'opticflow_param': {
#         'pyr_scale': 0.5,
#         'levels': 3,
#         'winsize': 100,
#         'iterations': 4,
#         'poly_n': 5,
#         'poly_sigma': 1.1,
#         'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
#     }
# }


class DenseOpticalFlow:

    def __init__(self, opt_param: dict):
        self.opt_param = opt_param
        self.directions_map = np.zeros([self.opt_param['size_accumulation'], 5])
        self.frame_previous = None
        self.gray_previous = None

    # желательно подавать обрезанную картинку

    def clearСash(self):
        self.directions_map.fill(0)
        self.frame_previous = None
        self.gray_previous = None

    def getMoveDirection(self, img):
        if self.frame_previous is None or self.gray_previous is None:
            self.frame_previous = img
            self.gray_previous = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(self.frame_previous)
            hsv[:, :, 1] = 255
            return 'wait'

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        optflow = cv2.calcOpticalFlowFarneback(self.gray_previous, gray, None, **self.opt_param['opticflow_param'])
        magnitude, ang = cv2.cartToPolar(optflow[:, :, 0], optflow[:, :, 1], angleInDegrees=True)
        # ang_180 = ang / 2
        self.gray_previous = gray
        move_sense = ang[magnitude > self.opt_param['threshold_magnitude']]
        # print(move_sense)
        move_mode = mode(move_sense)[0]
        # print(move_mode)

        if move_mode.size == 0:
            # print('None move')
            return 'wait'

        if 10 < move_mode <= 100:
            self.directions_map[-1, 0] = 1
            self.directions_map[-1, 1:] = 0
            self.directions_map = np.roll(self.directions_map, -1, axis=0)
        elif 100 < move_mode <= 190:
            self.directions_map[-1, 3] = 1
            self.directions_map[-1, :3] = 0
            self.directions_map[-1, 4:] = 0
            self.directions_map = np.roll(self.directions_map, -1, axis=0)
        elif 190 < move_mode <= 280:
            self.directions_map[-1, 2] = 1
            self.directions_map[-1, :2] = 0
            self.directions_map[-1, 3:] = 0
            self.directions_map = np.roll(self.directions_map, -1, axis=0)
        elif 280 < move_mode or move_mode < 10:
            self.directions_map[-1, 1] = 1
            self.directions_map[-1, :1] = 0
            self.directions_map[-1, 2:] = 0
            self.directions_map = np.roll(self.directions_map, -1, axis=0)
        else:
            self.directions_map[-1, -1] = 1
            self.directions_map[-1, :-1] = 0
            self.directions_map = np.roll(self.directions_map, 1, axis=0)

        # print(self.directions_map)
        loc = self.directions_map.mean(axis=0).argmax()
        if loc == 0:
            text = 'down'
        elif loc == 1:
            text = 'right'
        elif loc == 2:
            text = 'up'
        elif loc == 3:
            text = 'left'
        else:
            text = 'wait'

        return text

# For Tests

# videoFile = '/home/sauce-chili/Sirius/neural_backend/data/archive/top_10-07-2021_17:48:39.mp4'
# cap = cv2.VideoCapture(videoFile)
# opt = DenseOpticalFlow(opt_param=opt_param)

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (1280, 720)

# writer = cv2.VideoWriter('presentation.mp4', fourcc, fps, size)

# while cap.isOpened():
#     grabbed, frame = cap.read()
#     if not grabbed:
#         continue
#
#     frame = frame[140:, :]
#
#     text = opt.getMoveDirection(frame)
#
#     print(text)
#
#     cv2.imshow('frame', frame)
#     k = cv2.waitKey(1) & 0xff
#     if k == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

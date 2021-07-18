import numpy as np
import cv2

from utils import *

videoFile = '/home/sauce-chili/Sirius/neural_backend/data/archive/mid2_10-07-2021_00:35:01.mp4'

cap = cv2.VideoCapture(videoFile)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=300,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (300, 3))

# Take first frame and find  strong corners(contours) in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Starting point for comparison
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# print('p0: ')
# print(p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()

    try:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        continue

    # calculate optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # print('p1: ')
    # print(p1)
    # print('status:')
    # print(status)

    # Select good points
    good_new = p1[status == 1]
    # print(good_new)
    good_old = p0[status == 1]
    # print(good_old)
    print('=' * 20)
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    show_image(img, delay=1)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

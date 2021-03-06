import os
import sys
import uuid
import cv2
import numpy as np
import string
import random
import base64
import shutil
import glob
import logging
import time
import traceback
import multiprocessing
import signal
from logging.handlers import RotatingFileHandler


"""
                    __________________________________________________
                    ___________________________¶¶¶____________________
                    _______________________¶¶¶¶__¶¶¶__________________
                    ______________________¶¶_______¶__________________
                    _____________________¶¶_______¶¶__________________
                    ____________________¶¶_______¶¶_______________¶¶__
                    ____________________¶________¶¶¶_____________¶¶¶__
                    ____________________¶_________¶¶____________¶¶¶___
                    ___________________¶¶_________¶¶___________¶¶_____
                    ____________________¶_________¶¶__________¶¶______
                    ____________________¶¶________¶¶_________¶¶_______
                    ____________________¶¶________¶¶________¶¶________
                    _____________________¶¶¶_______¶¶_______¶_________
                    _____________________¶¶_________¶¶¶____¶¶_________
                    _____________________¶¶___________¶¶__¶¶__________
                    _____________________¶¶____________¶¶¶¶___________
                    _____________¶¶¶¶¶¶¶¶¶______________¶¶____________
                    __________¶¶¶¶_¶¶¶¶_¶¶______________¶¶¶___________
                    _______¶¶¶¶__¶¶¶___¶¶_______________¶¶¶___________
                    ______¶¶¶____¶¶___¶¶_________________¶¶___________
                    ____¶¶¶_¶¶¶¶¶¶¶_________¶¶¶__________¶¶________¶¶¶
                    ___¶¶¶_______¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶________¶¶______¶¶¶¶_
                    __¶¶____________¶¶¶¶_____¶¶¶¶¶_______¶______¶¶¶__¶
                    __¶¶____________________¶¶__¶¶______¶¶____¶¶¶__¶¶¶
                    __¶¶¶___________________¶¶__¶¶__________¶¶¶__¶¶¶__
                    ____¶¶¶_________________¶¶¶¶¶¶¶_________¶¶__¶¶¶___
                    ___¶¶¶¶¶¶______________¶¶¶___¶¶¶_______¶¶_¶¶¶_____
                    __¶¶_¶¶¶¶¶¶¶______¶¶__¶¶___¶¶_¶¶_______¶_¶¶_______
                    ___¶___¶¶¶¶¶¶¶¶¶__¶¶¶¶¶____¶¶_¶¶_____¶¶__¶________
                    ___¶¶¶¶_____¶¶¶¶¶¶¶¶¶¶¶_____¶¶¶¶_____¶¶¶¶¶________
                    _____¶¶¶¶¶¶¶¶_______________¶¶¶¶¶_____¶¶__________
                    _____¶¶_________________¶¶¶¶¶¶¶¶¶_____¶___________
                    ______¶¶¶¶______________¶¶¶__¶¶¶¶____¶¶___________
                    ________¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶____¶¶_____¶¶___________
                    _________¶¶____¶¶¶¶¶¶¶¶______¶¶¶____¶¶____________
                    __________¶¶_______________¶¶¶¶_¶¶¶¶¶_____________
                    __________¶¶¶¶¶__________¶¶¶¶_¶¶¶¶¶_______________
                    ____________¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶__¶¶¶__________________
                    ______________¶¶¶___________¶¶____________________
                    ________________¶¶¶¶¶¶_¶¶¶¶¶¶_____________________
                    __________________¶¶¶¶¶¶¶¶________________________
                    __________________________________________________
                    
"""


# collapse/expand all funcs --> 'ctrl+shift+NumPad -/+' for PyCharm

# ----------------  Utils

class Logger:

    def __init__(self, filename=None):

        self.format = '%(levelname)s -- %(asctime)s -- %(message)s'

        self.to_std = self.init_std_logger()
        self.to_std.info("Log method to std initialised.")

        if filename is not None and filename != "":
            self.filename = filename
            self.to_file = self.init_file_logger()
            self.to_std.info("Log method to {} initialised.".format(filename))

        self.printer = self.init_print()

    def init_std_logger(self):

        my_handler = logging.StreamHandler()
        my_handler.setFormatter(logging.Formatter(self.format, datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("std")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(my_handler)

        return logger

    def init_print(self):

        my_handler = logging.StreamHandler()
        my_handler.setFormatter(logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("printer")
        logger.setLevel(logging.INFO)
        logger.addHandler(my_handler)

        return logger

    def print(self, msg):
        self.printer.info(msg)

    def init_file_logger(self):

        my_handler = RotatingFileHandler(self.filename, mode='a', maxBytes=5 * 1024 * 1024,
                                         backupCount=0, encoding=None, delay=0)
        my_handler.setFormatter(logging.Formatter(self.format, datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("file_logger")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(my_handler)

        return logger


def get_random_name(name_len=22):
    """generate random name string without extension"""
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(name_len))


def get_filelist(directory, ext):
    """
    get files list with required extensions list
    """

    ret_list = []
    for folder, subs, files in os.walk(directory):

        for filename in files:

            if filename.split(".")[-1] in ext:
                ret_list.append(os.path.join(folder, filename))

    return ret_list


def get_random_uuid():
    return str(uuid.uuid4())


def get_mac():
    return str(uuid.getnode())


def get_format_date(date_format="%d-%m-%Y_%H:%M:%S"):
    return time.strftime(date_format)


def import_something(py_path, obj_name):
    from importlib import import_module
    try:
        return getattr(import_module(py_path.replace("./", "").replace("/", ".").replace(".py", "")), obj_name)
    except:
        print("Can`t import {} object from {}. Exit.".format(obj_name, py_path))
        exit()


def find_free_port(p_from, p_to):
    """
    get random web socket service port

    p_from  -  int
    p_to    -  int
    """

    def try_port(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except:
            return False
        finally:
            sock.close()

    try:
        if not len(range(int(p_from), int(p_to))) > 0:
            raise Exception("Port range is wrong - from {} to {}".format(int(p_from), int(p_to)))
        for i in range(int(p_from), int(p_to)):
            if try_port(i):
                return i
        return -1
    except Exception as e:
        print(e)


# ----------------  Image/Bbox utils

def xyxy_to_xcycwh(imh, imw, bbox):
    """
    convert [xmin, ymin, xmax, ymax] to relative coordinates [x_center, y_center, width, height]

    imh   -  image height
    imw   -  image width
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    """

    dw = 1. / imw
    dh = 1. / imh
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = x * dw
    if x < 0:
        x = 0.0001
    if x > imw:
        x = imw * 0.9999

    y = y * dh
    if y < 0:
        y = 0.0001
    if y > imh:
        y = imh * 0.9999

    w = w * dw
    if w < 0:
        w = 0.0001
    if w > imw:
        w = imw * 0.9999

    h = h * dh
    if h < 0:
        h = 0.0001
    if h > imh:
        h = imh * 0.9999

    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def xyxy_to_xywh(imh, imw, bbox):
    """
    convert [xmin, ymin, xmax, ymax] to relative coordinates [x_top, y_top, width, height]

    imh   -  image height
    imw   -  image width
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    """
    return [round(int(bbox[0]) / imw, 4), round(int(bbox[1]) / imh, 4),
            round((int(bbox[2]) - int(bbox[0])) / imw, 4), round((int(bbox[3]) - int(bbox[1])) / imh, 4)]


def xywh_to_xyxy(imh, imw, bbox):
    """
    convert relative coordinates [x_top, y_top, width, height] to absolute [xmin, ymin, xmax, ymax]

    imh   -  image height
    imw   -  image width
    bbox  -  [x,y,w,h]
    """
    return [int(bbox[0] * imw), int(bbox[1] * imh),
            int(bbox[0] * imw) + int(bbox[2] * imw), int(bbox[1] * imh) + int(bbox[3] * imh)]


def xcycwh_to_xyxy(imh, imw, bbox):
    """
    convert relative coordinates [x_center, y_center, width, height] to absolute [xmin, ymin, xmax, ymax]

    imh   -  image height
    imw   -  image width
    bbox  -  [xc,yc,w,h]
    """
    return [int((float(bbox[0]) - float(bbox[2]) / 2) * imw), int((float(bbox[1]) - float(bbox[3]) / 2) * imh),
            int((float(bbox[0]) + float(bbox[2]) / 2) * imw), int((float(bbox[1]) + float(bbox[3]) / 2) * imh)]


def draw_bbox(img, bbox, label="", bbox_color=(255, 25, 25), text_color=(225, 255, 255)):
    """
    draw bbox on image

    img   -  np.array
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    label -  str, detected class and probability for example "{} {:.4f}".format(cls, prob)
    """
    c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    cv2.rectangle(img, c1, c2, bbox_color, thickness=tl)
    if label != "":
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0]+ t_size[0] + 2, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, bbox_color, -1)  # filled
        cv2.putText(img, label, (c1[0] + 2, c1[1] + t_size[1]), 0, tl / 3, text_color, thickness=tf, lineType=cv2.LINE_AA)


def draw_text(img, x, y, text="", text_color=(225, 255, 255)):
    tl = round(0.002 * (x + y) / 2) + 1
    tf = max(tl - 1, 1)
    cv2.putText(img, text, (x, y), 0, tl / 3, text_color, thickness=tf, lineType=cv2.LINE_AA)


def get_iou(bb1, bb2):
    """
    calculate iou of two bboxes with absolute coordinates

    bb1  -  [xmin, ymin, xmax, ymax]
    bb2  -  [xmin, ymin, xmax, ymax]
    """

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    # if bb1[0] < bb1[2] or bb1[1] < bb1[3] or bb2[0] < bb2[2] or bb2[1] < bb2[3]:
    #     return 0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_overlap(bbox1, bbox2):
    """
    calculate how many bbox1 percents inside of bbox2

    bb1  -  [xmin, ymin, xmax, ymax]
    bb2  -  [xmin, ymin, xmax, ymax]
    """

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    if xB < xA or yB < yA:
        return 0

    inter_area = (xB - xA) * (yB - yA)
    bbox_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox_area == 0.:
        return 0

    res = inter_area / bbox_area
    return res


def cut_bbox(img, bbox, x_expand=.0, y_expand=.0):

    """
    return new image == bbox + expand coefficients

    img  -  np.array
    bbox  -  [xmin, ymin, xmax, ymax]
    x_expand=2 means increase new image width twice

    also return new image coordinates inside original image
    """

    if x_expand > 1.:
        x_exp = x_expand % 1
    else:
        x_exp = 1 - x_expand

    if y_expand > 1.:
        y_exp = y_expand % 1
    else:
        y_exp = 1 - y_expand

    yK = + int(((bbox[3] - bbox[1]) * y_exp) / 2)
    xK = + int(((bbox[2] - bbox[0]) * x_exp) / 2)

    new_bbox = [bbox[0] - xK if x_expand > 1. else bbox[0] + xK, bbox[1] - yK if y_expand > 1. else bbox[1] + yK,
                bbox[2] + xK if x_expand > 1. else bbox[2] - xK, bbox[3] + yK if y_expand > 1. else bbox[3] - yK]

    if new_bbox[0] < 0:
        new_bbox[0] = 1
    if new_bbox[1] < 0:
        new_bbox[1] = 1
    if new_bbox[2] > img.shape[1]:
        new_bbox[2] = img.shape[1] - 1
    if new_bbox[3] > img.shape[0]:
        new_bbox[3] = img.shape[0] - 1

    return img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], new_bbox


def show_image(img, win_name="show", delay=0):

    """easy imshow"""

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(delay=delay)


def resize_image(img, height=None, width=None, letterbox=False, lb_color=(128, 128, 128), inter=cv2.INTER_AREA):

    """
    img        -  np.array
    height     -  new H
    width      -  new W
    letterbox  -  add areas to lowest side for escaping image distortion
    lb_color   -  color of areas, default is gray
    """

    if letterbox:
        h, w = img.shape[:2]
        if h > w:
            border_y = 0
            border_x = round((h - w + .5) / 2.)
        else:
            border_x = 0
            border_y = round((w - h + .5) / 2.)
        img = cv2.copyMakeBorder(img, top=border_y, bottom=border_y, left=border_x, right=border_x,
                                 borderType=cv2.BORDER_CONSTANT, value=lb_color)

    dim = (width, height)
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=inter)


def rotate_image(img, ang):

    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, ang, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def warp_image(img, pts):
    """
    create rotated image using four points [top_left, top_right, bot_right, bot_left] absolute coords.

    Usage:

    roi = [(x_min, ymin), (x_max, y_max)]
    warped_img = warp_image(frame, np.array(eval(str([(roi[0][0], roi[0][1]), (roi[1][0], roi[0][1]),
                                                      (roi[1][0], roi[1][1]), (roi[0][0], roi[1][1])])),
                                            dtype="float32"))

    """
    # pts = np.array(eval(str(pts)), dtype="float32")
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    # tl, tr, br, bl = self.pts[0], self.pts[1], self.pts[2], self.pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)

    # return the warped image
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


# ----------------  Other

class MotionDetector:

    """
    Detect motion on image via difference between masks of key frame and other next frame

    key_frame      -  np.array (frame without objects)
    binary_thresh  -  0/255 more than this thresh value will be on mask
    area_thresh    -  minimal white blob area on difference image

    """

    def __init__(self, key_frame, binary_thresh=80, area_thresh=500, debug=False):

        self.key_frame = cv2.GaussianBlur(cv2.cvtColor(key_frame, cv2.COLOR_BGR2GRAY), (3, 3), 2)
        self.binary_thresh = binary_thresh
        self.area_thresh = area_thresh
        self.debug = debug

    @staticmethod
    def grab_contours(cnts):
        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(("Contours tuple must have length 2 or 3, "
                             "otherwise OpenCV changed their cv2.findContours return "
                             "signature yet again. Refer to OpenCV's documentation "
                             "in that case"))

        # return the actual contours array
        return cnts

    def detect_motion(self, img):

        results = []

        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 2)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imh, imw = gray.shape[:2]

        frameDelta = cv2.absdiff(self.key_frame, gray)

        threshed = cv2.threshold(frameDelta, self.binary_thresh, 255, cv2.THRESH_BINARY)[1]
        threshed = cv2.dilate(threshed, None, iterations=3)

        cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.area_thresh or cv2.contourArea(c) > 1200:
                continue

            # compute the bounding box for the contour, draw it on the frame, and update the text+
            x, y, w, h = cv2.boundingRect(c)
            # results.append([x, y, w, h])
            results.append([x / imw, y / imh, w / imw, h / imh])

        if self.debug:

            show_image(self.key_frame, "key_frame", delay=1)
            show_image(frameDelta, "delta", delay=1)
            show_image(threshed, "threshed", delay=1)

            if len(results):
                for box in results:
                    box = xywh_to_xyxy(img.shape[0], img.shape[1], box)
                    img = draw_bbox(img, box, bbox_color=(100, 10, 10))

            # stacked_img = stack_images_one_row(imgs_list=[self.key_frame, frameDelta, threshed, gray], stack_shape=224,
            #                      messages=["keyframe", "delta", "threshed", "motions"])

            show_image(img, "motions")

            # show_image(stacked_img)

        return results


class Reader:

    def __init__(self, name, src, type):

        self.name = name
        self.src = src
        self.type = type

        if type not in ["webcam", "rtsp_stream", "file", "directory"]:
            raise Exception("Not implemented reader type")

        self.connected = False

        self.q = multiprocessing.Queue(maxsize=1)
        self.proc = None

        self.connect_and_start_read()

    def read_rtsp(self, name, src):

        try:

            proc_out = {"name": name}

            cap = cv2.VideoCapture(src)

            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            proc_out["meta"] = {"h": h, "w": w, "fps": fps, "fourcc": fourcc}

            while True:

                status, frame = cap.read()
                proc_out["status"], proc_out["frame"] = status, frame
                proc_out["ts"] = time.time()

                if self.q.empty():
                    self.q.put(proc_out)

        except:
            self.q.put({"error": traceback.format_exc()})
            sys.exit()

    def read_from_files(self, name, src, type):

        files_list = []
        if type == "directory":
            files_list.extend(get_filelist(src, ["jpg", "png"]))
            files_list.extend(get_filelist(src, ["mp4", "avi", "mkv"]))

        elif type == "file":
            files_list.append(src)

        try:

            proc_out = {"name": name}

            for n, file in enumerate(files_list):

                if file.split(".")[-1] in ["mp4", "avi", "mkv"]:

                    cap = cv2.VideoCapture(file)

                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

                    proc_out["meta"] = {"h": h, "w": w, "fps": fps, "fourcc": fourcc}

                    while True:
                        status, frame = cap.read()
                        proc_out["status"], proc_out["frame"] = status, frame
                        proc_out["ts"] = time.time()

                        self.q.put(proc_out)

                elif file.split(".")[-1] in ["jpg", "png"]:

                    proc_out["meta"] = {}

                    cap = cv2.VideoCapture(file)
                    status, frame = cap.read()
                    proc_out["status"], proc_out["frame"] = status, frame
                    self.q.put(proc_out)

            sys.exit()

        except:
            self.q.put({"error": traceback.format_exc()})
            sys.exit()

    def get_frame(self):

        empty_q_time = 0.

        frame = None
        frame_meta = None
        st = time.time()

        while frame is None:

            if not self.proc.is_alive():
                self.connected = False
                return {"error": "DEAD PROCESS. RESTART READING"}

            if self.q.empty():
                empty_q_time += 0.1
                time.sleep(0.1)

                if empty_q_time > 15:
                    self.connected = False
                    return {"error": "EMPTY QUEUE TOO LONG TIME. RESTART READING"}

                continue
            out = self.q.get()

            if "error" in out:
                self.connected = False
                return {"error": f"READER PROCESS ERROR : {out['error']}. RESTART READING"}

            elif out["status"] is False:
                self.connected = False
                return {"error": "READER FALSE STATUS. RESTART READING"}

            elif out["frame"] is None:
                self.connected = False
                return {"error": "READER NONE FRAME. RESTART READING"}

            self.connected = True
            frame = out["frame"]
            frame_meta = out["meta"]

        return {"frame": frame, "meta": frame_meta, "time": round(time.time() - st, 4)}

    def kill(self):
        try:
            os.kill(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def connect_and_start_read(self):

        if self.type in ["webcam", "rtsp_stream"]:
            self.proc = multiprocessing.Process(target=self.read_rtsp, args=(self.name, self.src, ))

        elif self.type in ["file", "directory"]:
            self.proc = multiprocessing.Process(target=self.read_from_files, args=(self.name, self.src, self.type, ))

        self.proc.start()

    def reset(self):
        self.connected = False
        self.q = multiprocessing.Queue(maxsize=1)
        self.kill()
        self.connect_and_start_read()


class Writer:

    def __init__(self, file_name, fps, height, width, fourcc='mp4v'):
        self.wr = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*fourcc), int(fps), (width, height))

    def write_to_file(self, frame):
        self.wr.write(frame)

    def finish_writing(self):
        self.wr.release()

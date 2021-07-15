import torch
import torch.nn as nn
import torch.nn.functional as F
from easyocr import Reader
import numpy as np
import time
from utils import warp_image, show_image, get_filelist, draw_bbox
import cv2
import pytesseract


class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0],
                               out_channels=self.input_shape[1],
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.input_shape[1],
                               out_channels=self.input_shape[1]*2,
                               kernel_size=3,
                               padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(self.input_shape[1]*2*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.fl(x)
        # x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        prediction = self.fc2(x)
        return prediction


class MLP(nn.Module):

    def __init__(self, input_shape):
        super(MLP, self).__init__()

        self.input_shape = input_shape

        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.input_shape[0]*self.input_shape[1]*self.input_shape[2], out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=10)
        self.relu = nn.ReLU()
        # self.softmax = nn.LogSoftmax()
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.LogSigmoid()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fl(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu(x)

        x = self.fc2(x)
        # print(x.shape)
        x = self.relu(x)

        x = self.fc3(x)
        # x = self.softmax(x)
        # x = self.sigmoid(x)

        # print(x.shape)
        return x


class EasyOcr:
    """class with EasyOcr and sorting algorithm"""
    def __init__(self):
        self.model = Reader(["en"], gpu=True, verbose=False)
        self.buff = {}
        self.ID = []

    def predict(self, image, minsize):
        """
        minsize - minimum numbers of the answer that would be used in the final choosing

        return info - dict with keys (bbox - bounding box, prob - prediction, number - text, frame - picture)
        """
        results = self.model.readtext(
            image,
            contrast_ths=10,
            low_text=0.15,
        )
        all_items = {}
        clefs = []
        ans = {}
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            res = {"bbox": (tr, tl, br, bl), "number": text, "prob": prob, "frame": image}

            clef = tr[0] - tl[0]
            all_items[clef] = res
            if len(text) < minsize:
                continue
            else:
                clefs.append(clef)
        clefs.sort()
        if len(clefs) > 0:
            ans = all_items[clefs[-1]]
        return ans

    def saver(self, ans):
        self.buff[ans["prob"]] = ans
        self.ID.append(ans["prob"])
        # print(self.ID)

    def choose(self):
        if self.ID:
            self.ID.sort()
            id = self.ID[-1]
            ans = self.buff[id]
            self.buff.clear()
            self.ID.clear()
            return ans

    def cleaner(self):
        self.buff.clear()
        self.ID.clear()


class OCRReader:
    def __init__(self, src, type="mp4", format_directory="jpg", all_info=None):
        """
        ---this class should be used for comfortable reading text-information from frames---

        > There are three types ["mp4" or "rtsp", "img", "dir"] - video, image, directory.
        (format_directory - it is name-type of all images in the directory)

        > all_info - dict, list, tuple or something else (structure where would be saved all info)
        """
        self.type = type
        self.src = src
        self.all_info = all_info
        self.video = None
        self.empty_frames = 0
        self.model = EasyOcr()
        if type == "mp4" or type == "rtsp":
            self.video = cv2.VideoCapture(src)
        elif type == "dir":
            self.counter = 0
            self.imgs = get_filelist(src, [format_directory])
        elif type == "img":
            self.img = cv2.imread(src)
        else:
            raise Exception("Not implemented reader type")

    def video_run(self, max_wait_iterations=20, cut_box=[(196, 400), (1235, 400), (1235, 1041), (196, 1041)], watch=True):
        """
        > max_wait_iterations - delay after choosing final info for saving
        > watch - (True or False) watching video
        > cut_box - coordinates in pixels for cutting frame [(top_left), (top_right), (bot_right), (bot_left)] (x, y)
        """
        _, img = self.video.read()
        warped_img = warp_image(img, np.array(eval(str(cut_box)), dtype="float32"))
        results = self.model.predict(warped_img, 7)  # , draw_bbox
        if len(results) == 0:
            self.empty_frames += 1
        if len(results):
            self.model.saver(results)
            self.image_show(warped_img, results)
        elif self.empty_frames == max_wait_iterations:
            pif_paf = self.model.choose()
            if pif_paf:
                self.all_info[time.time()] = pif_paf
                print(self.empty_frames)
            self.empty_frames = 0
        if watch:
            show_image(warped_img, delay=0)

    def image_run(self, local_src, cut_box=[(196, 400), (1235, 400), (1235, 1041), (196, 1041)], watch=False, max_wait_iterations=20):
        """
        >
        > watch - (True or False) watching image
        > cut_box - coordinates in pixels for cutting frame [(top_left), (top_right), (bot_right), (bot_left)] (x, y)
        """
        warped_img = warp_image(local_src, np.array(eval(str(cut_box)), dtype="float32"))
        results = self.model.predict(warped_img, 7)  # , draw_bbox
        if len(results) == 0:
            self.empty_frames += 1
        if len(results):
            self.model.saver(results)
            self.image_show(warped_img, results)
        elif self.empty_frames == max_wait_iterations:
            pif_paf = self.model.choose()
            if pif_paf:
                self.all_info[time.time()] = pif_paf
                print(self.empty_frames)
            self.empty_frames = 0
        if watch:
            show_image(warped_img, delay=1)

    def images_run(self, cut_box=[(196, 400), (1235, 400), (1235, 1041), (196, 1041)], watch=True):
        """
        > watch - (True or False) watching images from directory
        > cut_box - coordinates in pixels for cutting frame [(top_left), (top_right), (bot_right), (bot_left)] (x, y)
        """
        try:
            img = cv2.imread(self.imgs[self.counter])
        except:
            exit()
        self.counter += 1
        warped_img = warp_image(img, np.array(eval(str(cut_box)), dtype="float32"))
        results = self.model.predict(warped_img, 7)  # , draw_bbox
        if not (len(results)):
            self.empty_frames += 1
        if len(results):
            self.model.saver(results)
            self.image_show(warped_img, results)
        elif self.empty_frames == 1:
            pif_paf = self.model.choose()
            if pif_paf:
                self.all_info[time.time()] = pif_paf
                print(self.empty_frames)
            self.empty_frames = 0
        if watch:
            show_image(warped_img, delay=0)

    def image_show(self, img, results):
        tl = results["bbox"][1]
        br = results["bbox"][2]
        text = results["number"]
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def cleaner(self):
        self.empty_frames = 0

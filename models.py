import multiprocessing
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from easyocr import Reader
import numpy as np
import time
from torchsummary import summary
from utils import warp_image, show_image, get_filelist
import cv2
from Firebase import *


# NN for classification wagon status(Fill or Empty)
class FENN(nn.Module):

    def __init__(self, input_shape, classes: list, deviceType):
        super(FENN, self).__init__()

        self.input_shape = input_shape
        self.classes = classes
        self.device = torch.device(deviceType)
        self.to(device=self.device)
        self.network = nn.Sequential(

            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(self.input_shape[1] * 2 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(classes))
        )
        # summary(self, self.input_shape)

    def forward(self, x):
        return self.network(x)

    def predict(self, image):

        predict = {}

        with torch.no_grad():
            # print(self.input_shape[1:])
            # print(image)
            img = cv2.resize(image, self.input_shape[1:]) / 255.0
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.permute(2, 0, 1).float()
            img_tensor = torch.unsqueeze(img_tensor, 0)

            # forward + backward + optimize

            img_tensor = img_tensor.to(self.device)
            predicts = torch.softmax(self.forward(img_tensor), dim=1).cpu().numpy()[0]

            # predicts = predicts.cpu().numpy()
            className = self.classes[predicts.argmax()]
            # print(className, predicts)
            accuracy = predicts[predicts.argmax()]
            # print(accuracy)

            predict["className"] = className
            predict["accuracy"] = accuracy

        return predict


class MLP(nn.Module):

    def __init__(self, input_shape, classes):
        super(MLP, self).__init__()

        self.input_shape = input_shape

        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=len(classes))
        self.relu = nn.ReLU()
        # self.softmax = nn.LogSoftmax()
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.LogSigmoid()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fl(x)
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

    def __init__(self, gpu=False):
        self.model = Reader(["en"], gpu=gpu, verbose=False)
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
            elif text.isdigit() == 0:
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
    def __init__(self, src=None, type=None, format_directory="jpg", all_info=None, gpu=False):
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
        self.empty_frames = 0   # global variable which shows how many frames passed without a number
        self.model = EasyOcr(gpu=gpu)
        if type == "mp4":
            self.video = cv2.VideoCapture(src)
            print("video in ocr")
        elif type == "dir":
            self.counter = 0
            self.imgs = get_filelist(src, [format_directory])
            print("directory in ocr")
        elif type == "rtsp":
            print("rtsp in ocr")
        else:
            raise Exception("Not implemented reader type")

    def video_run(self, max_wait_iterations=20, cut_box=[(196, 400), (1235, 400), (1235, 1041), (196, 1041)],
                  watch=True):
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
            self.show_bbox(warped_img, results)
        elif self.empty_frames == max_wait_iterations:
            pif_paf = self.model.choose()
            if pif_paf:
                self.all_info[time.time()] = pif_paf
                print(self.empty_frames)
            self.empty_frames = 0
        if watch:
            show_image(warped_img, delay=0)

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
            self.show_bbox(warped_img, results)
        elif self.empty_frames == 1:
            pif_paf = self.model.choose()
            if pif_paf:
                self.all_info[time.time()] = pif_paf
                print(self.empty_frames)
            self.empty_frames = 0
        if watch:
            show_image(warped_img, delay=0)

    def main_ocr_run(self, local_src, max_wait_iterations=20):
        """
        > independent function
        > should be used in Data Processor
        if the number has not appeared yet returns None
        if the number has appeared
        """
        results = self.model.predict(local_src, 7)  # , draw_bbox
        print(len(results))
        if len(results) == 0 and len(self.model.buff) > 0:
            if self.empty_frames != max_wait_iterations:
                self.empty_frames += 1
                return results
        if len(results):
            self.model.saver(results)
            # cv2.imwrite(f"./data/results_of_backend/{time.ctime()} - mid1.jpg", local_src)
            self.show_bbox(local_src, results)
            self.empty_frames = 0
            return {"flag": "writing in the buffer now"}
        elif self.empty_frames >= max_wait_iterations:
            pif_paf = self.model.choose()
            if pif_paf:
                self.empty_frames = 0
                return pif_paf
                # print(self.empty_frames)
        else:
            return results

    def show_bbox(self, img, results):
        tl = results["bbox"][1]
        br = results["bbox"][2]
        text = results["number"]

        cv2.line(img, results["bbox"][0], results["bbox"][2], (222, 222, 222), 3)
        cv2.line(img, results["bbox"][2], results["bbox"][3], (222, 222, 222), 3)
        cv2.line(img, results["bbox"][3], results["bbox"][1], (222, 222, 222), 3)
        cv2.line(img, results["bbox"][1], results["bbox"][0], (222, 222, 222), 3)
        # cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def cleaner(self):
        self.empty_frames = 0

class FB_send:
    def __init__(self):
        self.q = multiprocessing.Queue(maxsize=1)
        self.DC = DataComposer()
        self.DC.CreateCurrentShift()
        self.proc = multiprocessing.Process(target=self.send_to_FB())
        self.proc.start()

    def send_to_process(self, event: dict):
        #TODO check it
        if self.q.empty():
            self.q.put(event)

    def send_to_FB(self):
        while True:
            try:
                out = self.q.get()
                if out is not None:
                    self.DC.AddEvent(out["time"],
                                     out["direction"],
                                     out["number"],
                                     out["trainID"],
                                     out["state"],
                                     out["event_frames"]
                                    )
            except:
                print(traceback.format_exc())

import multiprocessing
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from easyocr import Reader
import numpy as np
import time
from torchsummary import summary
from utils import warp_image, show_image, get_filelist, get_format_date
from Firebase import *
import os
import yaml
from yolov5 import OCR
import cv2
from utils import draw_bbox


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
        # TODO (вставка new осr)
        self.model = Reader(["en"], gpu=gpu, verbose=False)
        # TODO ------------------
        self.buff = {}
        self.ID = []

    def predict(self, image, minsize):
        """
function give information from the frame and sort it
        @param image: frame
        @param minsize: int - minimum numbers of the answer that would be used in the final choosing
        @return: dict - with keys (bbox - bounding box, prob - prediction, number - text, frame - picture)
        """
        # TODO (вставка new осr)
        results = self.model.readtext(
            image,
            contrast_ths=10,
            low_text=0.15,
        )
        # TODO ------------------
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
        """
function for saving frame with info (answer) into the EasyOcr buffer
        @param ans:
        """
        self.buff[ans["prob"]] = ans
        self.ID.append(ans["prob"])

    def choose(self):
        """
        @return: dict - with better information from the buffer
        """
        if self.ID:
            self.ID.sort()
            id = self.ID[-1]
            ans = self.buff[id]
            self.buff.clear()
            self.ID.clear()
            return ans

    def cleaner(self):
        """
function for cleaning buffer
        """
        self.buff.clear()
        self.ID.clear()


class Our_OCR:
    """class with EasyOcr and sorting algorithm"""

    def __init__(self, gpu=False):
        # TODO (вставка new осr)
        self.model = OCR.NumDetector(weights="./yolov5/runs/train/exp26/weights/best.pt",
                                     device="cpu",
                                     inp_shape=(320, 320)
                                     )
        # TODO ------------------
        self.buff = {}
        self.ID = []

    def predict(self, image, minsize):
        """
function give information from the frame and sort it
        @param image: frame
        @param minsize: int - minimum numbers of the answer that would be used in the final choosing
        @return: dict - with keys (bbox - bounding box, prob - prediction, number - text, frame - picture)
        """
        # TODO (вставка new осr)
        results = self.model.predict(image)
        # TODO ------------------
        all_items = {}
        clefs = []
        ans = {}
        for (bbox, num, avProb) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            res = {"bbox": (tr, tl, br, bl), "number": num, "prob": avProb, "frame": image}

            clef = tr[0] - tl[0]
            all_items[clef] = res
            if len(num) < minsize:
                continue
            elif num.isdigit() == 0:
                continue
            else:
                clefs.append(clef)
        clefs.sort()
        if len(clefs) > 0:
            ans = all_items[clefs[-1]]
        return ans

    def saver(self, ans):
        """
function for saving frame with info (answer) into the EasyOcr buffer
        @param ans:
        """
        self.buff[ans["prob"]] = ans
        self.ID.append(ans["prob"])

    def choose(self):
        """
        @return: dict - with better information from the buffer
        """
        if self.ID:
            self.ID.sort()
            id = self.ID[-1]
            ans = self.buff[id]
            self.buff.clear()
            self.ID.clear()
            return ans

    def cleaner(self):
        """
function for cleaning buffer
        """
        self.buff.clear()
        self.ID.clear()


class OCRReader:
    def __init__(self, src=None, type=None, format_directory="jpg", all_info=None, gpu=False, nn='OCR'):
        """
        ---this class should be used for comfortable reading text-information from frames---
        > There are three types ["mp4" or "rtsp", "img", "dir"] - video, image, directory.
        (format_directory - it is name-type of all images in the directory)
        > gpu - using gpu or not
        > all_info - dict, list, tuple or something else (structure where would be saved all info)
        > nn - 'OCR' or 'EasyOCR' (selection of nn type)
        """
        self.type = type
        self.src = src
        self.all_info = all_info
        self.video = None
        self.empty_frames = 0   # global variable which shows how many frames passed without a number
        #################################
        if nn == 'Our_OCR':
            self.model = EasyOcr(gpu=gpu)
        elif nn == 'EasyOCR':
            self.model = Our_OCR()
        else:
            print('none type of nn')
        ################################
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
        debug function
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
        debug function
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
function for analyzing information from frames and making better choice between same frames of one wagon
        @param local_src: frame
        @param max_wait_iterations: int - how many frames should be empty before making choosing better frame and number
        @return: dict
        """
        # TODO (вставка new осr)
        results = self.model.predict(local_src, 7)  # , draw_bbox
        # TODO ---------------
        if len(results) == 0 and len(self.model.buff) > 0:
            if self.empty_frames != max_wait_iterations:
                self.empty_frames += 1
                return results
        if len(results):
            self.model.saver(results)  #TODO-----------------------
            self.show_bbox(local_src, results)
            self.empty_frames = 0
            return {"flag": "writing in the buffer now"}
        elif self.empty_frames >= max_wait_iterations:
            pif_paf = self.model.choose()  #TODO---------------------
            if pif_paf:
                self.empty_frames = 0
                return pif_paf
        else:
            return results

    def show_bbox(self, img, results):
        """
function for drawing bbox and text on the frame (can be used only inside this class)
        @param img: frame
        @param results: dict
        """
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
    """class for creating collateral process for sending info into the Firebase"""
    def __init__(self):
        self.q = multiprocessing.Queue(maxsize=1)
        self.proc = multiprocessing.Process(target=self.send_to_FB)
        self.proc.start()

    def send_to_process(self, event: dict):
        """
function add information in queue, which is linked main process with collateral process
        @param event: dict with information
        """
        #TODO check it
        if self.q.empty():
            self.q.put(event)

    def send_to_FB(self):
        """
function for cycle reading from the queue and sending information in FireBase (with help Firebase module)
        """
        self.DC = DataComposer()
        # self.DC.CreateCurrentShift()
        try:
            while True:
                    if not(self.q.empty()):
                        out = self.q.get()
                        self.DC.AddEvent(out["time"],
                                         out["direction"],
                                         out["number"],
                                         out["trainID"],
                                         out["state"],
                                         out["event_frames"]
                                        )
                        #print('info has been sanded')
        except:
            print(traceback.format_exc())


class Config:
    def __init__(self, path: str):
        with open(eval(f'r"{path}"')) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)
        #############-Cameras-###########################
        self.mid1 = self.cfg['cameras']['mid1']
        self.top = self.cfg['cameras']['top']
        self.cameras = self.cfg['cameras']
        #############-OpticalFlow-#######################
        self.optical_params = self.cfg['OpticalFlow']
        self.cfg['OpticalFlow']['opticflow_param']['flags'] = eval(self.cfg['OpticalFlow']['opticflow_param']['flags'])
        #############-Fenn_full_empty-##############################
        self.fenn_all_fe = self.cfg['Fenn_full_empty']
        #############-Fenn_train-##############################
        self.fenn_all_tr = self.cfg['Fenn_train']
        #############-Osr_train-##############################
        self.gpu_ocr = self.cfg['Ocr']['gpu']
        self.type_ocr = self.cfg['Ocr']['type']
        #############-All-####################################
        self.src = self.cfg['Source']
        self.cut_cord = self.cfg['Cut_cord']
        self.max_wait_iteration = self.cfg['Max_wait_iteration']
        self.time_zone = self.cfg['Time_zone']
        self.nn_type = self.cfg['Ocr_type']
        #############-Image-##################################
        self.image_show = self.cfg['Image']['image_show']
        self.saving_results = self.cfg['Image']['saving_results']
        self.flag_4img = self.cfg['Image']['flag_4img']
        self.dir_for_save = self.cfg['Image']['dir_for_save']
        self.cfg['Image']['fontFace'] = eval(self.cfg['Image']['fontFace'])
        self.fontFace = self.cfg['Image']['fontFace']
        self.fontScale = self.cfg['Image']['fontScale']
        self.color = self.cfg['Image']['color']
        self.thickness = self.cfg['Image']['thickness']
        #####################################################

    def cfg_save(self, mid1=None, top=None, optical_params=None, fenn_dev_fe=None, input_shp_fe=None,
                 classes_fe=None, pathToWeights_fe=None, fenn_dev_tr=None, input_shp_tr=None, classes_tr=None,
                 pathToWeights_tr=None, src_ocr=None, gpu_osr=None, type_ocr=None, cut_cord=None, image_show=None,
                 saving_results=None, flag_4img=None, dir_for_save=None, fontFace=None, fontScale=None, color=None,
                 thickness=None):
        pass


def time_zone(tm=3):
    time = get_format_date(date_format="%Y-%m-%dT%H:%M:%S")
    hours = int(time[11:13]) + tm
    if hours >= 24:
        hours -= 24
    hours = str(hours)
    if len(hours) != 2:
        return time.replace(time[11:14], f'0{hours}:', 1)
    else:
        return time.replace(time[11:14], f'{hours}:', 1)

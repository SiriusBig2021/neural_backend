import pytesseract
import cv2
import numpy as np
import time
from models import EasyOcr
from utils import warp_image, show_image, get_filelist, draw_bbox


empty_frames = 0   # frames without numbers
cc = 0

imgs = get_filelist("/home/home/projects/neural_backend/samples/ocr", ["jpg"])
# img = cv2.imread("/home/home/projects/neural_backend/samples/numberOCR.jpg")
# img = cv2.imread("photo_2021-07-10_12-12-37.jpg")
# img = cv2.imread("photo_2021-07-10_12-18-24.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

v = "./samples/cutVideo.mp4"

cap = cv2.VideoCapture(v)

model = EasyOcr()
all_info = {}     # dict which is considered all frames with numbers and their optional information
while True:

    try:
        img = cv2.imread(imgs[cc])
    except:
        exit()
    cc += 1

    # _, img = cap.read()  #TODO
    # warped_img = img
    warped_img = warp_image(img, np.array(eval(str([(196, 400), (1235, 400), (1235, 1041), (196, 1041)])), dtype="float32"))
    results = model.predict(warped_img, 7)  #, draw_bbox
    if not(len(results)):
        empty_frames += 1
    if len(results):
        model.saver(results)
        # print("results - ", results)
        tl = results["bbox"][1]
        br = results["bbox"][2]
        text = results["number"]
        cv2.rectangle(warped_img, tl, br, (0, 255, 0), 2)
        cv2.putText(warped_img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # print(empty_frames)
    elif empty_frames == 20:
        pif_paf = model.choose()
        if pif_paf:
            all_info[time.time()] = pif_paf
        empty_frames = 0
        print(all_info)

    show_image(warped_img, delay=0)



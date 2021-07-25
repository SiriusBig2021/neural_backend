from yolov5 import wrappedModel
import cv2
from utils import draw_bbox

m = wrappedModel.NumDetector(weights="/home/sauce-chili/PycharmProjects/test/yolov5/runs/train/exp26/weights/best.pt",
                             device="cpu",
                             inp_shape=(320, 320))

img = cv2.imread("/home/sauce-chili/ocr_dataset_val/1ZGDIYTUV258ICKHNQZ02X.jpg")

predicts = m.predict(img)
print(predicts)

draw_bbox(img, predicts["bbox"],str(predicts["sumProb"]))
cv2.imshow('bbox',img)
cv2.waitKey(0)

import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import letterbox

import cv2
import numpy as np
import os


class NumDetector:

    def __init__(self, weights, device, inp_shape):

        os.environ["CUDA_VISIBLE_DEVICES"] = device

        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.inp_shape = inp_shape

        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.agnostic_nms = False
        self.max_det = 1000

    # # image pre processing # # should return img for your network input # #

    def __imgPreprocessing(self, img):
        img = letterbox(img, self.inp_shape[0], stride=int(self.model.stride.max()))[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)  # , dtype=np.float32
        img /= 255.0
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def __resultPostprocessing(self, pred, img, img0):

        results = []

        # Process detections
        for i, det in enumerate(pred):  # detections for image i

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  #   normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # x = int(xyxy[0]) / img0.shape[1]
                    # if x < 0.:
                    #     x = 0.001
                    # y = int(xyxy[1]) / img0.shape[0]
                    # if y < 0.:
                    #     y = 0.001
                    # w = (int(xyxy[2]) - int(xyxy[0])) / img0.shape[1]
                    # if w > 1.:
                    #     w = 0.99
                    # h = (int(xyxy[3]) - int(xyxy[1])) / img0.shape[0]
                    # if h > 1.:
                    #     h = 0.99
                    #
                    # xywh = [x, y, w, h]

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    results.append({'type': self.classes_names[int(cls)], 'prob': float(conf), "bbox": xywh})

        return results

    def __xcycwh_to_xyxy(self, imh, imw, bbox):
        """
        convert relative coordinates [x_center, y_center, width, height] to absolute [xmin, ymin, xmax, ymax]

        imh   -  image height
        imw   -  image width
        bbox  -  [xc,yc,w,h]
        """
        return [int((float(bbox[0]) - float(bbox[2]) / 2) * imw), int((float(bbox[1]) - float(bbox[3]) / 2) * imh),
                int((float(bbox[0]) + float(bbox[2]) / 2) * imw), int((float(bbox[1]) + float(bbox[3]) / 2) * imh)]

    def predict(self, image) -> dict:
        """
        @param image: input img
        @return: dict with recognized number and bbox number.
                *key num - recognized num in str type
                *key bbox - bbox around number.Contains 4 cord(x_min, y_min , x_max, y_max)
                *key sumProb - the sum of the probabilities of each number
        """
        h, w = image.shape[:2]
        processedImage = self.__imgPreprocessing(image)

        prediction = self.model(processedImage)[0]

        prediction = non_max_suppression(prediction, self.conf_thres, self.iou_thres)
        # torch.cuda.empty_cache()

        resultPredict = self.__resultPostprocessing(prediction, processedImage, image)


        number = ""
        sumProb = 0

        for res in resultPredict:
            # convert to absolute cord (x_min,y_min,x_max,y_max) for each bbox numbers
            res["bbox"] = self.__xcycwh_to_xyxy(h, w, res["bbox"])

        resultPredict = sorted(resultPredict, key=lambda var: var["bbox"][0])
        # print(resultPredict)

        for res in resultPredict:
            number += res["type"]
            sumProb += res["prob"]

        bbox_x_min = min([res["bbox"][0] for res in resultPredict])
        bbox_y_min = min([res["bbox"][1] for res in resultPredict])
        bbox_x_max = max([res["bbox"][2] for res in resultPredict])
        bbox_y_max = max([res["bbox"][3] for res in resultPredict])

        # cv2.circle(image, (bbox_x_min, bbox_y_min), 5, (0, 255, 0), 1)
        # cv2.circle(image, (bbox_x_max, bbox_y_max), 5, (0, 0, 255), 1)

        return {
            "num": number,
            "bbox": [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max],
            "avProb": (sumProb / len(number))
        }

# if __name__ == "__main__":
#     m = NumDetector(weights="/home/sauce-chili/PycharmProjects/yolov5/runs/train/exp26/weights/best.pt",
#                     device="cpu",
#                     inp_shape=(320, 320))
#
#     img = cv2.imread("/home/sauce-chili/ocr_dataset_val/1ZGDIYTUV258ICKHNQZ02X.jpg")
#
#     predicts = m.predict(img)
#
#     print(predicts)

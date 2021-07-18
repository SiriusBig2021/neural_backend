from models import FENN
import torch
import cv2

cfg = {
    "device": "cpu",  # "cpu" or "cuda:0" for gpu
    "input_shape": (3, 128, 128),  # ch, h, w
    "classes": ['Empty', 'Fill'],
    "pathToWeights": ""

}


def initCNN():
    return FENN(input_shape=cfg["input_shape"], classes=cfg["classes"], deviceType=cfg["device"])\
        .load_state_dict(torch.load(cfg["pathToWeights"]))


if __name__ == "__main__":

    model = initCNN()

    pathToImg = '/home/sauce-chili/Sirius/neural_backend/test_CNN/Кадр от top_11-07-2021_10:39:56_14-07-2021-03:00.mp4.png'

    print(model.predict(cv2.imread(pathToImg)))
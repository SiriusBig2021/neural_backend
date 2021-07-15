import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import os
import glob

config = {
    'path': None,  # folder to train img
    'fillClassifierClasses': {"Empty": 0, "Fill": 1},
    'wagonDetection': {"Train": 0, "None": 1},
    'imgSize': (64, 64),
}


class CustomDataset(Dataset):

    def __init__(self, metaFile, imagesPath, modelType=None, transform=None):

        with open(metaFile) as mf:
            data = mf.readlines()

        self.data = []
        self.transform = transform

        for i in data:
            image, label1, label2 = i.split()

            imagePath = os.path.join(imagesPath, image)
            if modelType == "fillClassifier":
                self.data.append([imagePath, label2])
                self.class_map = config['fillClassifierClasses']

            elif modelType == "wagonDetection":
                self.data.append([imagePath, label1])

        self.img_dim = config['imgSize']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim) / 255.
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1).float()
        # class_id = torch.tensor([class_id])

        # img_tensor = self.transform(img)
        # img_tensor = img_tensor.permute(2, 0, 1)
        # print(img_tensor, ' ', class_id)
        return img_tensor, class_id

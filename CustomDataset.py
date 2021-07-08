import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import glob

config = {
    'path': None,  # folder to train img
    'classes': {"train": 0, "none": 1},
    'imgSize': (720, 1280),
}



class CustomDataset(Dataset):

    def __init__(self, imgsFolder):
        self.imgs_path = imgsFolder
        root = glob.glob(self.imgs_path + "*/", )  # get only path
        print(root)
        self.data = []
        for class_path in root:
            path = class_path.split("/")
            while '' in path:
                path.remove('')
            class_name = path[-1]
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])

        print(self.data)
        self.class_map = config['classes']
        self.img_dim = config['imgSize']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        # print(img_tensor, ' ', class_id)
        return img_tensor, class_id

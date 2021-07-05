import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from datagenerator import *
from utils import *


if __name__ == "__main__":

    train_cfgs = [

        {

            "tag": "mnist_classifier",

            "model": "MLP",
            "device": "cuda:0",  # "cpu" or "cuda:0" for gpu

            "input_shape": (1, 28, 28),  # ch, h, w

            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],

            "epochs": 5,
            "init_lr": 0.001,
            "batch_size": 16,

            "loss": "categorical_cross_entropy",
            "optimizer": "adam",

            "save_to": "./tag.pt"

        }

    ]

    for cfg in train_cfgs:

        cfg["save_to"] = cfg["save_to"].replace("tag", cfg["tag"])
        device = torch.device(cfg["device"])

        model = None

        if cfg["model"] == "MLP":
            model = MLP(input_shape=cfg["input_shape"])

        if model is None:
            raise Exception("Not implemented model %s. Exit." % cfg["model"])

        model.to(device)

        summary(model, model.input_shape)

        model.load_state_dict(torch.load(cfg["save_to"]))


        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

        train_set = MNIST("./data", train=True, download=False, transform=transform)
        test_set = MNIST("./data", train=False, download=False, transform=transform)

        # train_set = CIFAR10("./data", train=True, download=False, transform=transform)
        # test_set = CIFAR10("./data", train=False, download=False, transform=transform)

        train_gen = torch.utils.data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=1)
        test_gen = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

        with torch.no_grad():

            for i, data in enumerate(test_gen):

                images, labels = data[0], data[1]

                inputs, labels = images.to(device), labels.to(device)

                # forward + backward + optimize
                predicts = torch.softmax(model(inputs), dim=1)
                predicts = predicts.cpu().numpy()[0]

                print("predict %s ; prob %.3f" % (cfg["classes"][predicts.argmax()],predicts[predicts.argmax()]))

                for im in images:
                    im = np.transpose(im.numpy(), (1, 2, 0))
                    show_image(im)


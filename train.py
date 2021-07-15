import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from CustomDataset import CustomDataset

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
            "start": False,

            "tag": "train_classifier",

            "model": "MLP",
            "device": "cpu",  # "cpu" or "cuda:0" for gpu
            # TODO уточнить кол-во каналов
            "input_shape": (1, 28, 28),  # ch, h, w

            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],

            "epochs": 9,
            "init_lr": 0.001,
            "batch_size": 16,

            "loss": "binary",
            "optimizer": "ReLU",

            "save_to": "./tag.pt"

        },

        {
            "start": True,

            "tag": "fill_classifier",

            "model": "MLP",
            "device": "cpu",  # "cpu" or "cuda:0" for gpu
            # TODO уточнить кол-во каналов
            "input_shape": (3, 64, 64),  # ch, h, w

            "classes": ['Empty', 'Fill'],

            "epochs": 9,
            "init_lr": 0.001,
            "batch_size": 16,

            "loss": "binary",
            "optimizer": "ReLU",

            "save_to": "./tag.pt"

        }

    ]

    for cfg in train_cfgs:

        if not cfg["start"]:
            continue

        cfg["save_to"] = cfg["save_to"].replace("tag", cfg["tag"])
        device = torch.device(cfg["device"])

        model = None

        if cfg["model"] == "MLP":
            model = MLP(input_shape=cfg["input_shape"], classes=cfg["classes"])

        if model is None:
            raise Exception("Not implemented model %s. Exit." % cfg["model"])

        model.to(device)

        summary(model, model.input_shape)

        loss_function = nn.CrossEntropyLoss(reduction='mean')
        # loss_function = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"])

        # TODO data generator

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            # transforms.GaussianBlur(kernel_size=25),
            # transforms.RandomGrayscale(p=0.5),
            # transforms.ColorJitter(brightness=(0.1, 1.5), contrast=(0, 4), saturation=(0, 4), hue=(-0.5, 0.5)),
        ])

        train_set = MNIST("./data", train=True, download=True, transform=transform)
        test_set = MNIST("./data", train=False, download=True, transform=transform)

        tr_set = CustomDataset(metaFile="./data/Dataset/trainMetaFile.classes",
                               imagesPath="./data/Dataset/SlicedImg",
                               modelType="fillClassifier",
                               transform=transform)

        # train_set = CIFAR10("./data", train=True, download=False, transform=transform)
        # test_set = CIFAR10("./data", train=False, download=False, transform=transform)

        train_gen = torch.utils.data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=1)
        test_gen = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

        tr_gen = torch.utils.data.DataLoader(tr_set, batch_size=4, shuffle=True, num_workers=1)

        for epoch in range(cfg["epochs"]):

            # # Train

            pbar = tqdm(enumerate(tr_gen), total=len(tr_gen))

            loss_train = 0.0

            for i, data in pbar:
                images, labels = data[0], data[1]

                # for j, im in enumerate(images):
                #     im = im.numpy()
                #     im = np.transpose(im, (1, 2, 0))
                #     print(labels[j].numpy())
                #     show_image(im)

                inputs, labels = images.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predicts = model(inputs)
                loss = loss_function(predicts, labels)
                loss.backward()
                optimizer.step()

                loss_train += loss.item()

                pbar.set_description(
                    "Epoch %s/%s ; TrainLoss %.3f  " % (epoch + 1, cfg["epochs"], loss_train / (i + 1)))

            continue
            # # Test

            # TODO add metrics
            # acc https://androidkt.com/calculate-total-loss-and-accuracy-at-every-epoch-and-plot-using-matplotlib-in-pytorch/

            pbar = tqdm(enumerate(test_gen), total=len(test_gen))

            loss_test = 0.0

            with torch.no_grad():

                for i, data in pbar:
                    images, labels = data[0], data[1]

                    # for im in images:
                    #     print(min(im), max(im))
                    #     im = np.transpose(im.numpy(), (1, 2, 0))
                    #     show_image(im)

                    inputs, labels = images.to(device), labels.to(device)

                    # forward + backward + optimize
                    predicts = model(inputs)
                    loss = loss_function(predicts, labels)

                    loss_test += loss.item()

                    pbar.set_description(
                        "Epoch %s/%s ; TestLoss  %.3f  " % (epoch + 1, cfg["epochs"], loss_test / (i + 1)))

            # # Save model
            print("Saving weights to %s" % cfg["save_to"])
            torch.save(model.state_dict(), cfg["save_to"])

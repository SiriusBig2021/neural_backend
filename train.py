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
# from datagenerator import *
from utils import *

if __name__ == "__main__":

    train_cfgs = [

        # {
        #     "start": False,
        #
        #     "tag": "train_classifier",
        #
        #     "model": "CNN",
        #     "device": "cpu",  # "cpu" or "cuda:0" for gpu
        #
        #     "input_shape": (3, 228, 228),  # ch, h, w
        #
        #     "classes": ['Train', 'None'],
        #
        #     "epochs": 9,
        #     "init_lr": 0.001,
        #     "batch_size": 16,
        #
        #     "loss": "binary",
        #     "optimizer": "ReLU",
        #
        #     "save_to": "./tag.pt"
        #
        # },

        {
            "start": True,

            "tag": "fill_classifier",

            "model": "CNN",

            "device": "cuda:0",  # "cpu" or "cuda:0" for gpu

            "input_shape": (3, 128, 128),  # ch, h, w

            "classes": ['Empty', 'Fill'],


            "epochs": 9,
            "init_lr": 0.0001,
            "batch_size": 4,

            "loss": "binary",
            "optimizer": "ReLU",

            "save_to": "./tag.pt",

            "threshold": 0.5

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

        elif cfg['model'] == 'CNN':
            # model = CNN(input_shape=cfg["input_shape"], classes=cfg["classes"])
            model = FENN(input_shape=cfg['input_shape'], classes=cfg["classes"], deviceType=cfg["device"])

        # if cfg["model"] == "CNN":
        #     model = FENN(input_shape=cfg["input_shape"])

        if model is None:
            raise Exception("Not implemented model %s. Exit." % cfg["model"])

        # model.to(device)

        # summary(model, model.input_shape)

        loss_function = nn.CrossEntropyLoss(reduction='mean')
        # loss_function = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"])



        transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=45),
            transforms.GaussianBlur(kernel_size=15),
            transforms.RandomGrayscale(p=0.5),
            # transforms.ColorJitter(brightness=(0.1, 1.5), contrast=(0, 4), saturation=(0, 4), hue=(-0.5, 0.5)),
        ])

<<<<<<< HEAD
        # train_set = MNIST("./data/datasets", train=True, download=False, transform=transform)
        # test_set = MNIST("./data/datasets", train=False, download=False, transform=transform)

        train_set = CIFAR10("./data/datasets", train=True, download=True, transform=transform)
        test_set = CIFAR10("./data/datasets", train=False, download=True, transform=transform)
=======
        train_set = CustomDataset(
            metaFile="/home/sauce-chili/Sirius/neural_backend/data/Dataset/Fill_or_Empry_Dataset/trainMetaFile.classes",
            imagesPath="/home/sauce-chili/Sirius/neural_backend/data/Dataset/Fill_or_Empry_Dataset/Imgs",
            imgSize=cfg["input_shape"][1:],
            modelType="fillClassifier",
            transform=transform)

        test_set = CustomDataset(
            metaFile='/home/sauce-chili/Sirius/neural_backend/data/Dataset/Fill_or_Empry_Dataset/testMetaFile.classes',
            imagesPath="/home/sauce-chili/Sirius/neural_backend/data/Dataset/Fill_or_Empry_Dataset/Imgs",
            imgSize=cfg["input_shape"][1:],
            modelType='fillClassifier',
            transform=transform
        )
>>>>>>> 67b5ccaf875e4893b6b183af3e13d05365150a21

        train_gen = torch.utils.data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=1)
        test_gen = torch.utils.data.DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=1)

        # train_gen = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)

        for epoch in range(cfg["epochs"]):

            # Train unit

            pbar = tqdm(enumerate(train_gen), total=len(train_gen))

            loss_train = 0.0
            correct = 0
            c = 1000
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for i, data in pbar:
                images, labels = data[0], data[1]

<<<<<<< HEAD
                for im in images:
                    im = np.transpose(im.numpy(), (1, 2, 0))
                    show_image(im)
=======
                # for j, im in enumerate(images):
                #     im = im.numpy()
                #     im = np.transpose(im, (1, 2, 0))
                #     print(labels[j].numpy())
                #     show_image(im)
>>>>>>> 67b5ccaf875e4893b6b183af3e13d05365150a21

                inputs, labels = images.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predicts = model(inputs)
                loss = loss_function(predicts, labels)
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad()

                loss_train += loss.item()

                # calc metrics
                predicts = predicts.detach().numpy()
                lbl = cfg["classes"][(labels.detach().numpy()[0])]
                pred = cfg["classes"][predicts[0].argmax()]
                prob = predicts[0][predicts[0].argmax()]

                correct += int(pred == lbl)
                tp += int((pred == lbl) and prob > cfg["threshold"])
                tn += int(((lbl != pred) and (lbl is None)) and prob < cfg["threshold"])
                fp += int((lbl != pred) and prob > cfg["threshold"])
                fn += int((pred == lbl) and prob < cfg["threshold"])

                if tp > 0 or fp > 0:
                    accuracy = (tp + tn) / (tn + fn + fp + tp)
                    precision = tp / (tp + fp)
                else:
                    accuracy = 0
                    precision = 0

                pbar.set_description(
                    "Epoch %s/%s ; TrainLoss %.3f  ; Acc %.3f ; Pre %.3f" % (
                        epoch + 1, cfg["epochs"], loss_train / (i + 1), accuracy, precision))

            # Test unit

            # acc https://androidkt.com/calculate-total-loss-and-accuracy-at-every-epoch-and-plot-using-matplotlib-in-pytorch/

            pbar = tqdm(enumerate(test_gen), total=len(test_gen))

            loss_test = 0.0
            loss_train = 0.0
            correct = 0
            c = 1000
            tp = 0
            tn = 0
            fp = 0
            fn = 0

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

                    # calc metrics
                    predicts = predicts.detach().numpy()
                    lbl = cfg["classes"][(labels.detach().numpy()[0])]
                    pred = cfg["classes"][predicts[0].argmax()]
                    prob = predicts[0][predicts[0].argmax()]

                    correct += int(pred == lbl)
                    tp += int((pred == lbl) and prob > cfg["threshold"])
                    tn += int(((lbl != pred) and (lbl is None)) and prob < cfg["threshold"])
                    fp += int((lbl != pred) and prob > cfg["threshold"])
                    fn += int((pred == lbl) and prob < cfg["threshold"])

                    if tp > 0 or fp > 0:
                        accuracy = (tp + tn) / (tn + fn + fp + tp)
                        precision = tp / (tp + fp)
                    else:
                        accuracy = 0
                        precision = 0

                    pbar.set_description(
                        "Epoch %s/%s ; TrainLoss %.3f  ; Acc %.3f ; Pre %.3f" % (
                            epoch + 1, cfg["epochs"], loss_train / (i + 1), accuracy, precision))

            # # Save model
            print("Saving weights to %s" % cfg["save_to"])
            torch.save(model.state_dict(), cfg["save_to"])

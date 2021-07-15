import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0],
                               out_channels=self.input_shape[1],
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.input_shape[1],
                               out_channels=self.input_shape[1]*2,
                               kernel_size=3,
                               padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(self.input_shape[1]*2*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.fl(x)
        # x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        prediction = self.fc2(x)
        return prediction


class MLP(nn.Module):

    def __init__(self, input_shape, classes):
        super(MLP, self).__init__()

        self.input_shape = input_shape

        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.input_shape[0]*self.input_shape[1]*self.input_shape[2], out_features=128)
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
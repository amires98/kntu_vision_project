import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as tf
import numpy as np
import cv2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model():
    my_model=torch.load('entiremodel11.pth')
    # my_model = torch.load('model/entiremodel10.pth')
    my_model.eval()
    return my_model


def my_prediction(batch, my_model):
    transf = tf.Compose([tf.ToPILImage(),
                         tf.Resize((42, 20)),
                         tf.ToTensor(),
                         tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    new_li=[]
    for img in batch:
        img = transf(img)
        img=img.unsqueeze(0)
        l=my_model(img)
        new_li.append(l)

    return new_li

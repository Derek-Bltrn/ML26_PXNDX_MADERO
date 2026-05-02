import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(history_length+1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, n_classes)

        # TODO : define layers of a convolutional neural network
        # ¿ how many images to use as input? (this is your input channels)
        # How many actions can your agent execute? (this is your output neurons)

    def forward(self, x):
        # TODO: implement forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def calc_out_size(self, w, h, kernel_size, padding, stride):
        # TODO: Implement the function to get the output size for a cnn given the parameters
        width = ((w + 2 * padding - kernel_size) // stride) + 1
        height = ((h + 2 * padding - kernel_size) // stride) + 1
        return width, height

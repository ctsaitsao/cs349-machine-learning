import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size[0], stride=stride[0])  # weights size [16, 3, 5, 5]
        self.conv2 = nn.Conv2d(16, 32, kernel_size[1], stride=stride[1])  # weights size [32, 16, 5, 5]
        self.fc = nn.Linear(32 * 13 * 13, 10)  # 13x13 from image dimension

    def forward(self, x):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        x = x.permute(0, 3, 1, 2)  # permute [1, 64, 64, 3] -> [1, 3, 64, 64]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # [1, 32, 13, 13] -> [1, 5408]
                                                   # without this incorrectly transforms to [416, 13]
                                                   # need to do this because conv layers are 4D and fc are 2D
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying
    synthesized images.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Third hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size[0], stride=stride[0])
        self.conv2 = nn.Conv2d(2, 4, kernel_size[1], stride=stride[1])
        self.conv3 = nn.Conv2d(4, 8, kernel_size[2], stride=stride[2])
        self.fc = nn.Linear(8 * 1 * 1, 2)

    def forward(self, x):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

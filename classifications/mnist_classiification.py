import torch
import torchvision

############################################################
##### Download and load dataset according to the model #####
############################################################

# load training data and labels
train_dataset = torchvision.datasets.MNIST('.', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                               ]))

#load test data and labels
test_dataset = torchvision.datasets.MNIST('.', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


#create dataloader for train and test data to suffle data and create batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

###################################
##### Create torch model here #####
###################################

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# create derived class f rom the pytorch nn.module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input : 32 x 32 x 3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # Input(with maxpool2d) : 15 x 15 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        # Input: flattern of 7 x 7 x 32 = 140+1470 = 1610
        self.fc1 = nn.Linear(1610, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # set as self.training because dropout only requires for training
        x = self.fc2(x)
        return F.softmax(x)

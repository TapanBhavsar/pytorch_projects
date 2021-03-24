import torch
import torchvision

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

print(len(train_loader))
print(len(test_loader))

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
        # Input(with maxpool2d) : 13 x 13 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        # Input: flattern of 5 x 5 x 32 = 800
        self.fc1 = nn.Linear(800, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # set as self.training because dropout only requires for training
        x = self.fc2(x)
        return F.softmax(x)

# initialize back propagation parameters
learning_rate = 0.01
momentum = 0.9
network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

######################################
##### Train the model #####
######################################

for epoch in range(10):
  epoch_loss = 0
  for batch_idx, (data, target) in enumerate(train_loader):
      # zero the gradient parameters because if it is not zerop the previous gradient accumulates with current gradient.
      optimizer.zero_grad()
      # prediction of batch.
      output = network(data)
      # find the loss value.
      loss = criterion(output, target)
      # start back propagation calculation.
      loss.backward()
      # run optimizer
      optimizer.step()

      epoch_loss += loss.item()
      if (batch_idx % 100 ==0 or batch_idx == 937):
        print("epoch: {}, batch size: {}, loss: {}".format(epoch+1, batch_idx + 1, epoch_loss / len(train_loader)))
        epoch_loss = 0


############################################
##### Save the trained model #####
############################################


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
                                 (0.0,), (1,))
                               ]))

#load test data and labels
test_dataset = torchvision.datasets.MNIST('.', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.0,), (1,))
                             ]))

#create dataloader for train and test data to suffle data and create batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
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
        # Input : 28 x 28 x 1
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
network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

######################################
##### Train the model #####
######################################

epochs = 10
for epoch in range(epochs):
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
      if (batch_idx % 100 ==0):
        print("epoch: {}, batch size: {}, loss: {}".format(epoch+1, batch_idx + 1, epoch_loss / len(data)))
        epoch_loss = 0


################################################
##### Save the entire/weight trained model #####
################################################

MODEL_PATH = "trained_model.pt"
# torch.save(network.state_dict(), MODEL_PATH)  # save trained weights only
torch.save(network, MODEL_PATH)  # save entire model with architecture as well

#################################################################
##### Load the model entirely as well as using only weights #####
#################################################################

MODEL_PATH = "trained_model.pt"

# load with weight saved model only.
'''
model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
'''

# load the entire model.
model = torch.load(MODEL_PATH)
model.eval()  # model.eval() must be called to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

###############################################
##### Test the loaded model #####
##############################################
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

###########################################
##### Predict a single image #####
###########################################
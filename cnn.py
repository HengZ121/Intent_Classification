import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data import *
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold

### Load Data
dataset = Dataset("Dataset.csv")
classes = list(set(dataset.label))

# Parameters:
batch_size = 5
filter_size = 5

### Define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_size := (dataset.max_len-filter_size)+1, filter_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(out_size2 := out_size, (out_size2-filter_size)+1, filter_size)
        self.fc1 = nn.Linear(3456, (num_classes := len(classes)))
        self.fc2 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### 10-fold
kf = KFold(n_splits=10)
fold = 1
for train_index, test_index in kf.split(dataset.sentence):
    print("Fold ", fold)

    train_sub = torch.utils.data.dataset.Subset(dataset, train_index)
    test_sub = torch.utils.data.dataset.Subset(dataset, test_index)
    

    trainloader = torch.utils.data.DataLoader(train_sub, batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_sub, batch_size=1, shuffle=False)

    cnn = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    print("Start Training")
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #reshape inputs
            inputs = torch.tensor(np.array(inputs).reshape(len(inputs),1,28,28))
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print("Training Complete")

    print("Start Testing")
    y_predict = []
    y_true = []
    for data in tqdm(testloader):
        input, label = data
        pred = list((cnn(inputs))[0])
        y_predict.append(int(pred.index(max(pred))))
        y_true.append(int(label))
    print("Testing Complete")
    print("Precision Score is: ", metrics.precision_score(y_true, y_predict, average = "macro"))
    print("Recall Score is: ", metrics.recall_score(y_true, y_predict, average = "macro"))
    print("Accuracy Score is: ", metrics.accuracy_score(y_true, y_predict))
    fold += 1
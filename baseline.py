import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data import *
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
from sklearn.model_selection import KFold
from Intent_classification_final import *

### Load Data
dataset = Dataset("Dataset.csv")
classes = list(set(dataset.label))

# Parameters:
batch_size = 10
filter_size = (3, 5)
learning_rate = 0.01

# ### Define CNN
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, filter_size, padding= 0, stride = 3)
#         self.pool1 = nn.MaxPool2d((2, 1))
#         self.fc1 = nn.Linear(40, 20)
#         self.fc2 = nn.Linear(20, len(classes))
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = self.dropout(self.fc2(x))
#         return x

### 10-fold
kf = KFold(n_splits=10)
fold = 1
p_scores = []
r_scores = []
a_scores = []
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])
model = load_model("model.h5")
for train_index, test_index in kf.split(dataset.vector):
    # print("Fold ", fold)

    # train_sub = torch.utils.data.dataset.Subset(dataset, train_index)
    test_sub = torch.utils.data.dataset.Subset(dataset, test_index)

    # trainloader = torch.utils.data.DataLoader(train_sub, batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_sub, batch_size=1, shuffle=False)

    # cnn = Net()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
    # print("Start Training")
    # for epoch in range(1, 51):  # loop over the dataset multiple times
    #     i = 0
    #     print("Epoche: ", epoch)
    #     for data in tqdm(trainloader):
    #         running_loss = 0.0
    #         i += 1
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels, _ = data

    #         #reshape inputs
    #         inputs = torch.tensor(np.array(inputs).reshape(len(inputs),1,28,5))
            
    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = cnn(inputs)
            
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #     print(f'[{epoch}, {i + 1:5d}] loss: {running_loss/len(inputs)}')
    #         # print(cnn.named_parameters)
    
    # print("Training Complete")
    # print("Start Testing")
    
    y_predict = []
    y_true = []
    for data in testloader:
        input, label, sent = data
        sent = ''.join(sent)
        pred = predictions(sent)
        y_predict.append(p := np.where(pred == np.max(pred))[0])
        y_true.append(int(label))
    print("Testing Complete")
    
    print("Precision Score is: ", p := metrics.precision_score(y_true, y_predict, average = "macro"))
    p_scores.append(p)
    print("Recall Score is: ", r := metrics.recall_score(y_true, y_predict, average = "macro"))
    r_scores.append(r)
    print("Accuracy Score is: ", a := metrics.accuracy_score(y_true, y_predict))
    a_scores.append(a)
    fold += 1

print("***************************")
print("*Average Precision: ", mean(p_scores))
print("*Average Recall: ", mean(r_scores))
print("*Average Accuracy: ", mean(a_scores))
print("*Quantity of Classses: ", len(classes))
print("***************************")
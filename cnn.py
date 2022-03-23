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

# 'How to register for a new meal pass',)  real label:  3  output label:  2
# ('i want to know your office address in punjab?',)  real label:  0  output label:  1
# ('How to apply?',)  real label:  2  output label:  3
# ('Sodexo moblie application procedure?',)  real label:  3  output label:  2
# ('can u tell me your Email id ?',)  real label:  0  output label:  9
# ('abcd company email address?',)  real label:  0  output label:  2
# ('The bills to be uploaded for proving the address?',)  real label:  1  output label:  0
# ('Time you need for giving me a loan approved?',)  real label:  4  output label:  2
# ('Is it possible to meet any one in bank',)  real label:  0  output label:  5
# ('Loan for new business?',)  real label:  6  output label:  2
# ('How to apply for a new pass',)  real label:  3  output label:  0
# ('Need a new meal pass how to apply',)  real label:  3  output label:  0
# ('Where can I use this money from loan?',)  real label:  9  output label:  8
# ('Firms which can apply for this loan?',)  real label:  7  output label:  9
# ('Can I take the business loan from you?',)  real label:  7  output label:  8
# ('What is the money I can borrow from this loan?',)  real label:  8  output label:  2
# ('What are the different purposes this borrowed amount is valid?',)  real label:  9  output label:  2
# ('Time needed for the loan approved?',)  real label:  4  output label:  0
# ('For what scenarios will this loan help?',)  real label:  9  output label:  7
# ('What form do i have to fill for enrolling into sodexo',)  real label:  3  output label:  0
# ('how to get loan',)  real label:  2  output label:  4
# ('Can I call tomorrow?',)  real label:  0  output label:  3
# ('Which business can take loan from you?',)  real label:  7  output label:  8
# ('For which functions will I use abcd company direct?',)  real label:  9  output label:  7
# ('What are the ways to apply for abcd company direct?',)  real label:  2  output label:  0
# ('Tell me the procedure of this loan application?',)  real label:  2  output label:  7
# ('I started my business started last month',)  real label:  6  output label:  0
# ('What is money I Can take as a loan?',)  real label:  8  output label:  7
# ('I want to know the procedure for loan',)  real label:  2  output label:  0
# ('Its a fresh company',)  real label:  6  output label:  0
# ('stock market trader i need loan',)  real label:  9  output label:  6
# ('Time you need for approval',)  real label:  4  output label:  2
# ('Which purposes can this borrowed amount be used for?',)  real label:  9  output label:  7
# ('How to start application for loan?',)  real label:  2  output label:  0
# ('How do I apply?',)  real label:  3  output label:  2
# ('How will get loan',)  real label:  2  output label:  4
# ('How can I take loan from your company?',)  real label:  2  output label:  7
# ('Please send information about the money I can have from this loan?',)  real label:  8  output label:  0
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:00<00:00, 1281.23it/s] 
# Testing Complete
# [3, 0, 2, 2, 3, 0, 3, 8, 3, 0, 5, 1, 0, 1, 0, 1, 0, 2, 4, 0, 1, 4, 2, 6, 2, 8, 3, 3, 9, 7, 9, 7, 2, 2, 3, 8, 9, 2, 7, 4, 9, 1, 2, 2, 4, 9, 3, 8, 2, 2, 0, 2, 0, 2, 1, 7, 0, 9, 2, 4, 3, 9, 2, 9, 2, 9, 6, 5, 8, 2, 9, 9, 8, 2, 6, 7, 9, 0, 4, 9, 9, 2, 9, 
# 8, 3, 0, 2, 2, 2, 2, 2, 8]
# [2, 1, 2, 3, 3, 0, 2, 8, 3, 9, 5, 1, 0, 1, 2, 0, 0, 2, 2, 5, 1, 4, 2, 2, 2, 8, 0, 0, 8, 9, 9, 8, 2, 2, 3, 2, 2, 2, 7, 0, 7, 1, 2, 2, 4, 9, 0, 8, 2, 2, 0, 4, 3, 2, 1, 8, 0, 7, 2, 4, 3, 9, 0, 9, 7, 9, 0, 5, 7, 2, 9, 9, 8, 0, 0, 7, 6, 0, 2, 9, 7, 0, 9, 
# 8, 2, 0, 2, 2, 4, 2, 7, 0]
# Precision Score is:  0.5502583527583528
# Recall Score is:  0.5621666666666666
# Accuracy Score is:  0.5869565217391305
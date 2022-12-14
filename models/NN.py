# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from typing import Counter

train = pd.read_csv("/content/drive/MyDrive/ML_project/pre-processed_train.csv")
test = pd.read_csv("/content/drive/MyDrive/ML_project/pre-processed_test.csv")

train.shape, test.shape

X_train = train.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)
y_train = train["isFraud"].to_numpy().astype(np.float64)
X_test = test.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)

X_train.shape, X_test.shape

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

pipeline = Pipeline(steps=[('o', over), ('u', under)])
train_X_re, train_y_re = pipeline.fit_resample(X_train, y_train)

train_X_re.shape

EPOCHS = 50
BATCH_SIZE = 60000
LEARNING_RATE = 0.0001

class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(train_X_re).cuda(), torch.FloatTensor(train_y_re).cuda())

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = TestData(torch.FloatTensor(X_test).cuda())

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

input_layer = 258
# hidden_1 = 129
hidden_1 = 64
hidden_2 = 10
output_layer = 2

model = nn.Sequential(nn.Linear(input_layer, hidden_1),
                      nn.ReLU(),
                      nn.Linear(hidden_1, hidden_2),
                      nn.ReLU(),
                      nn.Linear(hidden_2, output_layer),
                      nn.Softmax(dim=1))
model = model.cuda()
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        # X_batch, y_batch = X_batch, y_batch
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        y_batch = y_batch.to(torch.int64)
        loss = loss_function(y_pred, y_batch)
        # acc = optimizer(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

ypred_train = model(torch.FloatTensor(X_train).cuda())

ypred_train.shape

ypred_train

ypred_train = ypred_train.cpu().detach().numpy()
ypred_train = np.argmax(ypred_train, 1)

ypred_train

Counter(ypred_train).keys()
Counter(ypred_train).values()

sum=0
for i in range(y_train.size):
  if(y_train[i] == ypred_train[i]):
    sum += 1
acc = sum / y_train.size
print("Train acc: ", acc)

ypred_test = model(torch.FloatTensor(X_test).cuda())

ypred_test = ypred_test.cpu().detach().numpy()
ypred_test = np.argmax(ypred_test, 1)

ypred_test.shape

Counter(ypred_test).keys()
Counter(ypred_test).values()

test_results = pd.Series(ypred_test.astype('int32'), name="isFraud")
test_results.to_csv("submission_nn.csv")

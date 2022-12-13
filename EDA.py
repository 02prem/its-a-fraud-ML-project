# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fgk4t58zfri7fSp6p4EXhBXr964ENs1d
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from typing import Counter
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('train.csv')

"""### Function to find correlated features"""

def corr_features(correlation_matrix, thres):
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > thres:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print("Following are highly correlated features:", correlated_features)

"""# EDA

## Basic Information
"""

train.describe()

"""## Graphs

#### Transaction Amount
"""

train.plot(x='isFraud', y='TransactionAmt', kind='scatter')

"""The training data shows that maximun frauds happen upto transaction amount of 4000. There are less chances of fraud at higher amount but still it can happen at any amount, yet we can infer that the transactions of more than 4000 could be genuine but not with 100% surety !!

#### Billing address
"""

train.plot(x='isFraud', y='addr1', kind='scatter')
train.plot(x='isFraud', y='addr2', kind='scatter')
plt.show()

"""We can not say much about the transaction by just looking at the billing region (addr1) or billing country (addr2) as frauds are scattered all over the regions

#### Device Type
"""

train.loc[train['isFraud'] == 1]["DeviceType"].value_counts().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.title("Types of devices used in a fraud transaction")
plt.show()

"""We can clearly see that the device type does not say anything about the transaction. One can fraud from desktop as well as mobile. So this feature is not of much use.

#### Transaction ID
"""

print("Total data:", train.shape[0])
print("No. of unique values of transaction ID:", train["TransactionID"].nunique())

"""As we can clearly see that there is a different Transaction ID for each transaction, this feature does not make much sense. We can remove this feature.

#### TransactionDT

“TransactionDT "corresponds to the number of seconds in a day.<br>
This indicates a time delta from a given reference date and time<br>
It is clearly a time-related information like how many days since transaction, etc. Adding this to the model won't make much sense.

#### Product CD
"""

train["ProductCD"].value_counts().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.title("Number of Product Code associated with transactions")
plt.xlabel("Product Code")
plt.ylabel("Count")
plt.show()

train.loc[train['isFraud'] == 1]["ProductCD"].value_counts().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.title("Count of Product Code associated with Fraud transactions")
plt.xlabel("Product Code")
plt.ylabel("Count")
plt.show()

"""Although we can see that products with ProductCD = 'W' have the most number of frauds, this is clear that out of number of products with code 'C' it has maximum percentage of fraud transactions.

There are about 50,000 products with code 'C' and about 6000 of them are fraud which makes it around 12%

And the other codes roughly makes it around 5-6%

#### Card
"""

train["card6"].value_counts().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.title("Type of card")
plt.xlabel("card6")
plt.ylabel("Count")
plt.show()

train.loc[train['isFraud'] == 1]["card6"].value_counts().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.title("Type of card used in fraud")
plt.xlabel("card6")
plt.ylabel("Count")
plt.show()

"""If you look at the overall data then credit card is used verly less in compare to debit card, but when it comes to fraud credit card are used almost same as debit card.

So we can conclude that there are bigger chance of transaction being fraud when credit card is used (7%) than debit card (2.5%).

#### M Columns
"""

from sklearn.preprocessing import LabelEncoder

train[['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']] = train[['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']].apply(LabelEncoder().fit_transform)
M_features = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'isFraud']
corr_M = train[M_features].corr()

plt.subplots(figsize = (20, 20))
sns.heatmap(corr_M,annot=True, cmap = 'YlGnBu') 
sns.set(font_scale=1)

corr_features(corr_M, 0.9)

"""Here M2, M3, M8 and M9 are highly correlated<br>
So we can keep only one of them and remove the others.<br>
M2 is most correlated with isFraud among the four

#### D Columns
"""

D_features = [ 'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']

corr_D = train[D_features].corr()
plt.subplots(figsize = (20, 20))
sns.heatmap(corr_D,annot=True, cmap = 'YlGnBu') 
sns.set(font_scale=1)

corr_features(corr_D, 0.9)

train["D2"].nunique(), train["D7"].nunique(), train["D12"].nunique(), train["D6"].nunique()

"""Here D2, D7, D12 and D6 are highly correlated<br>
So we can keep only one of them and remove the others.<br>
As D6 has the most number of unique values we will keep D6.

#### C Columns
"""

C_features = [ 'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14'] 
corr_C = train[C_features].corr()
plt.subplots(figsize = (20, 20))
sns.heatmap(corr_C,annot=True, cmap = 'YlGnBu') 
sns.set(font_scale=1)

corr_features(corr_C, 0.9)

"""Here 'C12', 'C9', 'C6', 'C7', 'C2', 'C8', 'C4', 'C11', 'C14', 'C10' are highly correlated

#### id Columns

id_01 to id_11
"""

id_1to11_features = [ 'id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'] 
corr_id_1to11 = train[id_1to11_features].corr()
plt.subplots(figsize = (15, 15))
sns.heatmap(corr_id_1to11,annot=True, cmap = 'YlGnBu') 
sns.set(font_scale=1)

corr_features(corr_id_1to11, 0.9)

"""none of the following have any high correlation between them"""

id_rem_columns = ['id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_32']
corr_id_rem = train[id_rem_columns].corr()
plt.subplots(figsize = (15, 15))
sns.heatmap(corr_id_rem,annot=True, cmap = 'YlGnBu') 
sns.set(font_scale=1)

corr_features(corr_id_rem, 0.9)

"""None are highly correlated

#### Device info
"""

train['device_name'] = train['DeviceInfo'].str.split('/', expand=True)[0]
train.loc[train['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
train.loc[train['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
train.loc[train['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung' 
train.loc[train['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
train.loc[train['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
train.loc[train['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
train.loc[train['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
train.loc[train['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
train.loc[train['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
train.loc[train['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
train.loc[train['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
train.loc[train['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
train.loc[train['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
train.loc[train['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
train.loc[train['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
train.loc[train['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
train.loc[train['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
train.loc[train.device_name.isin(train.device_name.value_counts()[train.device_name.value_counts() < 200].index), 'device_name'] = 'Others'

train["device_name"].value_counts().plot(kind='bar')
plt.title("Device Name")
plt.xlabel("device_name")
plt.ylabel("Count")
plt.show()

train.loc[train['isFraud'] == 1]["device_name"].value_counts().plot(kind='bar')
plt.title("Names of devices used in a fraud")
plt.xlabel("device_name")
plt.ylabel("Count")
plt.show()
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from typing import Counter
from sklearn.model_selection import GridSearchCV

train_standard = pd.read_csv('pre-processed_train.csv')
test_standard = pd.read_csv('pre-processed_test.csv')

X_train = train_standard.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)
y_train = train_standard["isFraud"].to_numpy().astype(np.float64)
X_test = test_standard.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)

"""# Re-sampling"""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

pipeline = Pipeline(steps=[('o', over), ('u', under)])
train_X_re, train_y_re = pipeline.fit_resample(X_train, y_train)

"""# Logistic regression model"""

from sklearn.linear_model import LogisticRegression

logistic_regression_model = LogisticRegression(max_iter=10000)

logistic_regression_model.fit(train_X_re, train_y_re)

ypred_logistic = logistic_regression_model.predict(X_test)

test_results_2 = pd.Series(ypred_logistic.astype('int32'), name="isFraud")
test_results_2.to_csv("submission_1.csv")